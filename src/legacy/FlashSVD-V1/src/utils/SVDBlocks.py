# this contains all SVDBlock classes for SVD inference/training
# all SVD Blocks are non-rank aware (as their prior implementation)
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
from src.kernels.flash_attn_triton import flash_attn_triton


class BertSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 svd_per_head: Callable, svd_low_rank:  Callable, rank_wo: int=768,):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab & factor Q/K/V_transpose
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

        Uq,Vq = svd_per_head(WqT, rank_attn)
        Uk,Vk = svd_per_head(WkT, rank_attn)
        Uv,Vv = svd_per_head(WvT, rank_attn)

        # 2) grab & factor FFN transpose
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
        bo2  = hf_layer.output.dense.bias.data

        U1,V1 = svd_low_rank(Wi,    rank_ff)
        U2,V2 = svd_low_rank(WoT,   rank_ff)
        
        # ——— 3) factor attention-output projection Wₒ ———
        Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
        bo_attn = hf_layer.attention.output.dense.bias.data
        Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        
        # ——— stash everything ———
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        
        # stash everything
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.U1, self.V1, self.b1= nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2= nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2       = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H
        scale = 1.0 / math.sqrt(dh)

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()
        
        # FlashAttention
        if mask is not None:
            # assume mask: [B, M], 1 for valid tokens
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            # no padding: everything valid
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        del Q, K, V
        torch.cuda.empty_cache()
        
        # # raw attention scores
        # logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale

        # if mask is not None:
        #     # mask shape [B, M] → [B,1,1,M]
        #     m = mask.view(B, 1, 1, M).to(torch.bool)
        #     logits = logits.masked_fill(~m, float("-1e9"))

        # A = torch.softmax(logits, dim=-1)
        # attn = torch.einsum("bhmn,bhnd->bhmd", A, V)
        # del Q, K, V, A
        # torch.cuda.empty_cache()

        # back to [B,M,dm]
        attn = attn.transpose(1,2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) 
        
        # FFN
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        out = self.ln2(x1 + y)
         
        return out





################ FWSVD BERT ################
class BertFWSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 fwsvd_per_head: Callable, fwsvd_low_rank:  Callable, rank_wo: int=768,):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab & factor Q/K/V_transpose
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

        Uq,Vq = fwsvd_per_head(WqT, rank_attn)
        Uk,Vk = fwsvd_per_head(WkT, rank_attn)
        Uv,Vv = fwsvd_per_head(WvT, rank_attn)

        # 2) grab & factor FFN transpose
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
        bo2  = hf_layer.output.dense.bias.data

        U1,V1 = fwsvd_low_rank(Wi,    rank_ff)
        U2,V2 = fwsvd_low_rank(WoT,   rank_ff)
        
        # ——— 3) factor attention-output projection Wₒ ———
        Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
        bo_attn = hf_layer.attention.output.dense.bias.data
        # factor its transpose so WoT_attn: [dm,dm] -> Uo:[dm,rank], Vo:[rank,dm]
        # print(Wo_full.shape)
        # Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        Uo, Vo = fwsvd_low_rank(Wo_full.t(), rank_wo)
        
        # # ——— stash everything ———
        # self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        # self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        # self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        

        # stash everything
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        #self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        #self.Wo, self.bo         = hf_layer.attention.output.dense.weight.data, hf_layer.attention.output.dense.bias.data
        self.U1, self.V1, self.b1= nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2= nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2       = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B,M,dm = x.shape
        _,H,_,R = self.Pq.shape
        dh      = dm//H
        scale   = 1.0/math.sqrt(dh)

        def project(x, P, V, b):                 # P:(H,dm,R)  V:(H,R,dh)  b:(1,H,1,dh)
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)      # (B,H,M,R)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b    # broadcast OK

        # ───────── inside forward ─────────
        Q = project(x, self.Pq[0], self.Vq[0], self.bq)   # self.bq is already (1,H,1,dh)
        K = project(x, self.Pk[0], self.Vk[0], self.bk)
        V = project(x, self.Pv[0], self.Vv[0], self.bv)
        
        
        # # FlashAttention
        # # build a [B,H,1,M] boolean mask (1=keep, 0=pad)
        # if mask is not None:
        #     # assume mask: [B, M], 1 for valid tokens
        #     mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        # else:
        #     # no padding: everything valid
        #     mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # # Flash-attn returns [B, H, M, dh] float32
        # attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)


        # raw attention scores
        logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale

        if mask is not None:
            # mask shape [B, M] → [B,1,1,M]
            m = mask.view(B, 1, 1, M).to(torch.bool)
            logits = logits.masked_fill(~m, float("-1e9"))

        A = torch.softmax(logits, dim=-1)
        attn = torch.einsum("bhmn,bhnd->bhmd", A, V)
        
        
        
        # back to [B,M,dm]
        attn = attn.transpose(1,2).reshape(B, M, dm)

        mid_o    = attn @ self.Uo       # [B,M,rank_wo]
        x1 = self.ln1(x + mid_o @ self.Vo + self.bo_attn)    

        # 2) FFN: add b1 *before* relu
        mid  = x1 @ self.U1               # [B,M,rank_ff]
        midV = mid @ self.V1             # [B,M,d_ff]
        midA = F.gelu(midV + self.b1)           # GELU, not ReLU
        y    = (midA @ self.U2) @ self.V2 + self.b2
        return self.ln2(x1 + y)






############################### RoBERTa Blocks #############################
class RobertaSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 svd_per_head: Callable, svd_low_rank:  Callable, rank_wo: int=768,):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab & factor Q/K/V_transpose
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

        Uq,Vq = svd_per_head(WqT, rank_attn)
        Uk,Vk = svd_per_head(WkT, rank_attn)
        Uv,Vv = svd_per_head(WvT, rank_attn)

        # 2) grab & factor FFN transpose
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
        bo2  = hf_layer.output.dense.bias.data

        U1,V1 = svd_low_rank(Wi,    rank_ff)
        U2,V2 = svd_low_rank(WoT,   rank_ff)
        
        # ——— 3) factor attention-output projection Wₒ ———
        Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
        bo_attn = hf_layer.attention.output.dense.bias.data
        # factor its transpose so WoT_attn: [dm,dm] -> Uo:[dm,rank], Vo:[rank,dm]
        # print(Wo_full.shape)
        # Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        
        # ——— stash everything ———
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        

        # stash everything
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        #self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        #self.Wo, self.bo         = hf_layer.attention.output.dense.weight.data, hf_layer.attention.output.dense.bias.data
        self.U1, self.V1, self.b1= nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2= nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2       = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H
        scale = 1.0 / math.sqrt(dh)

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()
        
        # FlashAttention
        # FlashAttention wants:
        #   Q,K,V: [B, H, M, dh]
        #   key_padding_mask: Optional[torch.BoolTensor] of shape [B, M]
        #   causal:          whether to apply a causal mask
        #   softmax_scale:   your 1/sqrt(dh) factor
        #
        # So if you have a BERT-style mask of 0=pad,1=keep in [B,M], just
        # convert it to bool and pass it in:
        # --- replace with FlashAttention via Triton wrapper ---
        # build a [B,H,1,M] boolean mask (1=keep, 0=pad)
        if mask is not None:
            # assume mask: [B, M], 1 for valid tokens
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            # no padding: everything valid
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        
        
        # # raw attention scores
        # logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale

        # if mask is not None:
        #     # mask shape [B, M] → [B,1,1,M]
        #     m = mask.view(B, 1, 1, M).to(torch.bool)
        #     logits = logits.masked_fill(~m, float("-1e9"))

        # A = torch.softmax(logits, dim=-1)
        # attn = torch.einsum("bhmn,bhnd->bhmd", A, V)



        # back to [B,M,dm]
        attn = attn.transpose(1,2).reshape(B, M, dm)
        
        # output projection + LayerNorm
        mid_o = attn @ self.Uo
        #print("mid_o shape:", mid_o.shape)
        x1 = self.ln1(x + mid_o @ self.Vo + self.bo_attn)

        # FFN
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        return self.ln2(x1 + y)







class RobertaFWSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 fwsvd_per_head: Callable, fwsvd_low_rank:  Callable, rank_wo: int=768,):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab & factor Q/K/V_transpose
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1,H,1,dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1,H,1,dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1,H,1,dh)

        Uq,Vq = fwsvd_per_head(WqT, rank_attn)
        Uk,Vk = fwsvd_per_head(WkT, rank_attn)
        Uv,Vv = fwsvd_per_head(WvT, rank_attn)

        # 2) grab & factor FFN transpose
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # [dm,d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()         # [d_ff,dm]
        bo2  = hf_layer.output.dense.bias.data

        U1,V1 = fwsvd_low_rank(Wi,    rank_ff)
        U2,V2 = fwsvd_low_rank(WoT,   rank_ff)
        
        # ——— 3) factor attention-output projection Wₒ ———
        Wo_full = hf_layer.attention.output.dense.weight.data  # [dm,dm]
        bo_attn = hf_layer.attention.output.dense.bias.data
        # factor its transpose so WoT_attn: [dm,dm] -> Uo:[dm,rank], Vo:[rank,dm]
        # print(Wo_full.shape)
        # Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        Uo, Vo = fwsvd_low_rank(Wo_full.t(), rank_wo)
        
        # ——— stash everything ———
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        

        # stash everything
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        #self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        #self.Wo, self.bo         = hf_layer.attention.output.dense.weight.data, hf_layer.attention.output.dense.bias.data
        self.U1, self.V1, self.b1= nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2= nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2       = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H
        scale = 1.0 / math.sqrt(dh)

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()
        
        # FlashAttention
        # FlashAttention wants:
        #   Q,K,V: [B, H, M, dh]
        #   key_padding_mask: Optional[torch.BoolTensor] of shape [B, M]
        #   causal:          whether to apply a causal mask
        #   softmax_scale:   your 1/sqrt(dh) factor
        #
        # So if you have a BERT-style mask of 0=pad,1=keep in [B,M], just
        # convert it to bool and pass it in:
        # --- replace with FlashAttention via Triton wrapper ---
        # build a [B,H,1,M] boolean mask (1=keep, 0=pad)
        if mask is not None:
            # assume mask: [B, M], 1 for valid tokens
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            # no padding: everything valid
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        
        
        # # raw attention scores
        # logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale

        # if mask is not None:
        #     # mask shape [B, M] → [B,1,1,M]
        #     m = mask.view(B, 1, 1, M).to(torch.bool)
        #     logits = logits.masked_fill(~m, float("-1e9"))

        # A = torch.softmax(logits, dim=-1)
        # attn = torch.einsum("bhmn,bhnd->bhmd", A, V)



        # back to [B,M,dm]
        attn = attn.transpose(1,2).reshape(B, M, dm)
        
        # output projection + LayerNorm
        mid_o = attn @ self.Uo
        #print("mid_o shape:", mid_o.shape)
        x1 = self.ln1(x + mid_o @ self.Vo + self.bo_attn)

        # FFN
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        return self.ln2(x1 + y)









