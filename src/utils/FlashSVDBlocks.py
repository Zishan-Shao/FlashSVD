import os
import sys
import time
import itertools
import torch
import torch.nn as nn
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
# from evaluate import load as load_metric
from typing import Callable, Tuple
import math
import torch.nn.functional as F

import functools

# we need to access this directory first
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.kernels.flashsvdattn import flash_svd_attention
from src.kernels.flashsvdffnv2 import flashsvd_ffn
from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
from src.utils.svd_helpers import build_plain_svd_helpers



class BertFlashSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn, rank_ff, svd_per_head, svd_low_rank, rank_wo):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        
        # factor Q/K/V
        WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1,H,1,dh)
        WkT, bk = cfg.key.weight.data.t(),   cfg.key.bias.data.view(1,H,1,dh)
        WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1,H,1,dh)
        self.Pq, self.Vq = map(nn.Parameter, svd_per_head(WqT, rank_attn))
        self.Pk, self.Vk = map(nn.Parameter, svd_per_head(WkT, rank_attn))
        self.Pv, self.Vv = map(nn.Parameter, svd_per_head(WvT, rank_attn))
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq,bk,bv))

        # factor FFN
        Wi, bi   = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
        WoT, bo2 = hf_layer.output.dense.weight.data.t(),      hf_layer.output.dense.bias.data
        self.U1, self.V1 = map(nn.Parameter, svd_low_rank(Wi,   rank_ff))
        self.U2, self.V2 = map(nn.Parameter, svd_low_rank(WoT, rank_ff))
        self.b1, self.b2 = map(nn.Parameter, (bi, bo2))

        # output projection (attn)
        Wo_full  = hf_layer.attention.output.dense.weight.data
        bo_attn  = hf_layer.attention.output.dense.bias.data
        self.Uo, self.Vo = map(nn.Parameter, svd_low_rank(Wo_full.t(), rank_wo))
        self.bo_attn    = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)

        Vq_full = self.Vq.expand(B,H,R,dh)
        Vk_full = self.Vk.expand(B,H,R,dh)
        Vv_full = self.Vv.expand(B,H,R,dh)
        bq_full = self.bq.expand(B,H,1,dh).squeeze(2)
        bk_full = self.bk.expand(B,H,1,dh).squeeze(2)
        bv_full = self.bv.expand(B,H,1,dh).squeeze(2)

        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()
        
        attn = attn_out.view(B,H,M,dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        mid = x1 @ self.U1 
        # flashsvdffn v2
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        out = self.ln2(x1 + y)
        return out 





################### FlashFWSVD BERT ####################
class BertFlashFWSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 fwsvd_per_head: Callable, fwsvd_low_rank:  Callable, rank_wo: int=768,):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features
        rank_wo = rank_wo or rank_attn  # default same as attention rank
        
        # ——— 1) factor Q/K/V ———
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

        Uq,Vq = fwsvd_per_head(WqT, rank_attn)
        Uk,Vk = fwsvd_per_head(WkT, rank_attn)
        Uv,Vv = fwsvd_per_head(WvT, rank_attn)

        # ——— 2) factor Wᵢ and Wₒ for FFN ———
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
        bo2  = hf_layer.output.dense.bias.data

        U1,V1 = fwsvd_low_rank(Wi,    rank_ff)
        U2,V2 = fwsvd_low_rank(WoT,   rank_ff)

        # ——— 3) factor attention-output projection Wₒ ———
        Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
        bo_attn = hf_layer.attention.output.dense.bias.data
        Uo, Vo = fwsvd_low_rank(Wo_full.t(), rank_wo)
        
        # ——— stash everything ———
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        self.U1, self.V1, self.b1    = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2    = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):  # x: [B,M,dm]
        B, M, dm = x.shape
        H, R = self.Pq.shape[1], self.Pq.shape[-1]
        dh   = dm // H
        scale = 1.0 / math.sqrt(dh)

        # ——— 1) project into low-rank Q/K/V ———
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq[0]).contiguous()
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk[0]).contiguous()
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv[0]).contiguous()

        # expand V and biases
        Vq_full = self.Vq.expand(B, H, R, dh)
        Vk_full = self.Vk.expand(B, H, R, dh)
        Vv_full = self.Vv.expand(B, H, R, dh)
        bq_full = self.bq.expand(B, H, 1, dh).contiguous().squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).contiguous().squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).contiguous().squeeze(2)

        # flash-SVD attention
        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4,
            block_m=32,
            block_r=R,
        )  # [B,H,M,dh]
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        attn = attn_out.view(B, H, M, dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        # ——— flash-SVD FFN ———
        mid = x1 @ self.U1              # [B,M,rank_ff]                             # [B,M,dm]
        # y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        return self.ln2(x1 + y)









