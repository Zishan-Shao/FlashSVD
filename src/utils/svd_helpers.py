# utility file for SVD perform
import torch 
import torch.nn as nn

# # we need to access this directory first
# THIS_FILE = os.path.abspath(__file__)
# REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)
from .fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert_with_attention,
)


# Helpers for BERT SVD decomposition 
def build_plain_svd_helpers(model):
    def svd_per_head(Wt: torch.Tensor, rank: int):
        d_model, _ = Wt.shape
        H          = model.config.num_attention_heads
        dh         = d_model // H
        Wt3        = Wt.view(d_model, H, dh)
        Us, Vs     = [], []
        for h in range(H):
            Wh = Wt3[:, h, :].float()  # to float32 for SVD
            U32, S32, Vh32 = torch.linalg.svd(Wh, full_matrices=False)
            U = (U32[:, :rank] * S32[:rank]).to(Wt.dtype)
            V = Vh32[:rank, :].to(Wt.dtype)
            Us.append(U)
            Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf = W.float()
        U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
        U = (U32[:, :rank] * S32[:rank]).to(W.dtype)
        V = Vh32[:rank, :].to(W.dtype)
        return U, V

    return svd_per_head, svd_low_rank


# BERT LayerShim
class BertLayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:,0,0,:] == 0)
        return (self.block(hidden_states, raw_mask),)





############################ FWSVD Helpers Here ############################
def build_fwsvd_helpers(model, dataloader, device="cuda", eps=1e-6):
    model = model.to(device).train()
    # compute all five fisher vectors
    fisher_q, fisher_k, fisher_v, fisher_int, fisher_out = \
        estimate_fisher_weights_bert_with_attention(
            model, dataloader, compute_full=False, device=device
        )

    H  = model.config.num_attention_heads
    dh = model.config.hidden_size // H
    fw_map = {}

    for i, layer in enumerate(model.bert.encoder.layer):
        # Q/K/V
        for Wt, fisher in [
            (layer.attention.self.query.weight.data.t(),   fisher_q[i]),
            (layer.attention.self.key  .weight.data.t(),   fisher_k[i]),
            (layer.attention.self.value.weight.data.t(),   fisher_v[i]),
        ]:
            fw_map[Wt.data_ptr()] = fisher + eps

        # FFN intermediate + output
        for Wt, fisher in [
            (layer.intermediate.dense.weight.data.t(),     fisher_int[i]),
            (layer.output.dense.weight.data.t(),           fisher_out[i]),
        ]:
            fw_map[Wt.data_ptr()] = fisher + eps

        # attention-output projection
        WoT = layer.attention.output.dense.weight.data.t()
        # you can choose any fisher vector here; adding eps just in case
        fw_map[WoT.data_ptr()] = fisher_int[i] + eps

    def fwsvd_per_head(Wt: torch.Tensor, rank: int):
        ptr = Wt.data_ptr()
        w   = fw_map[ptr]           # shape [d_model]
        d_model, total = Wt.shape
        H   = model.config.num_attention_heads
        dh  = d_model // H
        Wh  = Wt.view(d_model, H, dh)
        Us, Vs = [], []
        for h in range(H):
            U, V = compute_row_sum_svd_decomposition(
                Wh[:, h, :], weights=w, rank=rank
            )
            Us.append(U); Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def fwsvd_low_rank(W: torch.Tensor, rank: int):
        ptr = W.data_ptr()
        w   = fw_map[ptr]
        return compute_row_sum_svd_decomposition(W, weights=w, rank=rank)

    return fwsvd_per_head, fwsvd_low_rank




class BertFWLayerShim(nn.Module):
    """
    A shim giving our custom block the same forward signature as BertLayer.
    Only uses the first `hidden_states` argument; ignores everything else.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    #def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:,0,0,:] == 0)
        return (self.block(hidden_states, raw_mask),)






