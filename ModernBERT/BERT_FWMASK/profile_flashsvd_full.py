import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# ---------- helpers ----------

@torch.no_grad()
def svd_factorize_for_A_V(Wt: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For a linear y = x @ W^T + b where W^T ∈ R[in, out],
    we want W^T ≈ A^T @ V with A ∈ R[rank, in], V ∈ R[rank, out].
    Input Wt is W^T (shape [in, out]).
    """
    U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)  # Wt = U @ diag(S) @ Vh
    r = min(rank, S.numel())
    Ur = U[:, :r]                                 # [in, r]
    SrVh = (S[:r].unsqueeze(0) * Vh[:r, :])       # [r, out]
    A = Ur.T.contiguous()                         # [r, in]
    V = SrVh.contiguous()                         # [r, out]
    return A, V

@torch.no_grad()
def svd_factorize_for_U_V(Wt: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For a linear y = x @ W^T + b where W^T ∈ R[in, out],
    we want W^T ≈ U @ V with U ∈ R[in, rank], V ∈ R[rank, out].
    """
    U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
    r = min(rank, S.numel())
    Ur = U[:, :r]                                 # [in, r]
    SrVh = (S[:r].unsqueeze(0) * Vh[:r, :])       # [r, out]
    return Ur.contiguous(), SrVh.contiguous()     # U, V

@dataclass
class QKVFactors:
    Pq: torch.Tensor
    Pk: torch.Tensor
    Pv: torch.Tensor
    Vq: torch.Tensor
    Vk: torch.Tensor
    Vv: torch.Tensor
    bq: Optional[torch.Tensor]
    bk: Optional[torch.Tensor]
    bv: Optional[torch.Tensor]

# ---------- FFN wrapper using your fused kernel ----------

class FlashSVDGeGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, inter_size: int, *,
                 r1: int, r2: int, gelu_approx: str = "tanh",
                 blocks: Dict[str, int] = None, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.r1 = r1
        self.r2 = r2
        self.gelu_approx = gelu_approx

        # A1: [r1, hidden], V1: [r1, 2*inter], U2: [inter, r2], V2: [r2, hidden]
        self.A1 = nn.Parameter(torch.empty(r1, hidden_size, device=device, dtype=dtype))
        self.V1 = nn.Parameter(torch.empty(r1, 2 * inter_size, device=device, dtype=dtype))
        self.U2 = nn.Parameter(torch.empty(inter_size, r2, device=device, dtype=dtype))
        self.V2 = nn.Parameter(torch.empty(r2, hidden_size, device=device, dtype=dtype))
        self.b1 = nn.Parameter(torch.zeros(2 * inter_size, device=device, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))

        # launch block sizes for your kernel (can be tuned)
        blocks = blocks or {}
        self.BL = blocks.get("BL", 64)
        self.BD = blocks.get("BD", 128)
        self.BR1 = blocks.get("BR1", 64)
        self.BH = blocks.get("BH", 128)
        self.BR2 = blocks.get("BR2", 64)

    @torch.no_grad()
    def load_from_dense(self, dense1: nn.Linear, dense2: nn.Linear, r1: int = None, r2: int = None):
        r1 = self.r1 if r1 is None else r1
        r2 = self.r2 if r2 is None else r2

        # dense1: in=hidden, out=2*inter  (GeGLU pre-activation)
        W1t = dense1.weight.data.T.contiguous()  # [hidden, 2*inter]
        A1, V1 = svd_factorize_for_A_V(W1t, r1)
        self.A1.copy_(A1.to(self.A1.dtype))
        self.V1.copy_(V1.to(self.V1.dtype))
        if dense1.bias is not None:
            self.b1.copy_(dense1.bias.data.to(self.b1.dtype))

        # dense2: in=inter, out=hidden
        W2t = dense2.weight.data.T.contiguous()  # [inter, hidden]
        U2, V2 = svd_factorize_for_U_V(W2t, r2)
        self.U2.copy_(U2.to(self.U2.dtype))
        self.V2.copy_(V2.to(self.V2.dtype))
        if dense2.bias is not None:
            self.b2.copy_(dense2.bias.data.to(self.b2.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        # precompute P = x @ A1^T ∈ [B, L, r1]
        P = x.matmul(self.A1.t())
        # fused phase (P, V1, U2, V2, b1, b2) → [B, L, H]
        y = flashsvd_ffn_geglu_fused(
            P, self.V1, self.U2, self.V2, self.b1, self.b2,
            BL=self.BL, BD=self.BD, BR1=self.BR1, BH=self.BH, BR2=self.BR2,
            gelu_approx=self.gelu_approx,
        )
        return y

# ---------- Attention wrapper using your RoPE kernel ----------

class SVDRoPEMHA(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rotary_emb,
                 rq: int, rk: int, rv: int, ro: int,
                 bm=128, bn=128, bdh=None, br=64, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # input projections (A* terms for precompute P*)
        self.Aq = nn.Parameter(torch.empty(rq, hidden_size, device=device, dtype=dtype))
        self.Ak = nn.Parameter(torch.empty(rk, hidden_size, device=device, dtype=dtype))
        self.Av = nn.Parameter(torch.empty(rv, hidden_size, device=device, dtype=dtype))
        # post A* factors (V* terms) map rank->(H*dh)
        self.Vq = nn.Parameter(torch.empty(rq, num_heads * head_dim, device=device, dtype=dtype))
        self.Vk = nn.Parameter(torch.empty(rk, num_heads * head_dim, device=device, dtype=dtype))
        self.Vv = nn.Parameter(torch.empty(rv, num_heads * head_dim, device=device, dtype=dtype))
        self.bq = nn.Parameter(torch.zeros(num_heads * head_dim, device=device, dtype=dtype))
        self.bk = nn.Parameter(torch.zeros(num_heads * head_dim, device=device, dtype=dtype))
        self.bv = nn.Parameter(torch.zeros(num_heads * head_dim, device=device, dtype=dtype))

        # output projection (post-SDPA)
        self.Uo = nn.Parameter(torch.empty(num_heads * head_dim, ro, device=device, dtype=dtype))
        self.Vo = nn.Parameter(torch.empty(ro, hidden_size, device=device, dtype=dtype))
        self.bo = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))

        self.flash = FlashSVDRoPEAttention(num_heads, head_dim, rotary_emb,
                                           bm=bm, bn=bn, bdh=bdh, br=br)

    @torch.no_grad()
    def load_from_dense_qkv_o(self, q_lin: nn.Linear, k_lin: nn.Linear, v_lin: nn.Linear, o_lin: nn.Linear,
                              rq: int, rk: int, rv: int, ro: int):
        # Each linear is in=hidden, out=H*dh; we factorize on W^T ∈ [hidden, H*dh].
        for (A, V, b, lin, r) in [
            (self.Aq, self.Vq, self.bq, q_lin, rq),
            (self.Ak, self.Vk, self.bk, k_lin, rk),
            (self.Av, self.Vv, self.bv, v_lin, rv),
        ]:
            Wt = lin.weight.data.T.contiguous()
            Ai, Vi = svd_factorize_for_A_V(Wt, r)
            A.copy_(Ai.to(A.dtype)); V.copy_(Vi.to(V.dtype))
            if lin.bias is not None:
                b.copy_(lin.bias.data.to(b.dtype))

        # Out projection: y @ W_o^T with W_o^T ∈ [H*dh, hidden] → factorize as Uo@Vo
        Wot = o_lin.weight.data.T.contiguous()
        Uo, Vo = svd_factorize_for_U_V(Wot, ro)
        self.Uo.copy_(Uo.to(self.Uo.dtype)); self.Vo.copy_(Vo.to(self.Vo.dtype))
        if o_lin.bias is not None:
            self.bo.copy_(o_lin.bias.data.to(self.bo.dtype))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor],
                position_ids: torch.Tensor, sliding_window_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, M, H]
        Pq = x.matmul(self.Aq.t())  # [B, M, rq]
        Pk = x.matmul(self.Ak.t())
        Pv = x.matmul(self.Av.t())

        qkv = QKVFactors(
            Pq=Pq, Pk=Pk, Pv=Pv,
            Vq=self.Vq, Vk=self.Vk, Vv=self.Vv,
            bq=self.bq, bk=self.bk, bv=self.bv,
        )

        O = self.flash(qkv, attention_mask, position_ids, sliding_window_mask)  # [B,H,M,dh]
        B, H, M, dh = O.shape
        O = O.permute(0, 2, 1, 3).reshape(B, M, H * dh)  # [B,M,H*dh]

        # output projection via factors: (O @ Uo) @ Vo + bo
        y = O.matmul(self.Uo).matmul(self.Vo)
        return y + self.bo

# ---------- ModernBERT patcher ----------

def instrument_modernbert_with_flashsvd(model, *,
                                        r_ffn1=128, r_ffn2=128,
                                        r_q=64, r_k=64, r_v=64, r_o=64,
                                        gelu_approx="tanh"):
    """
    Replaces ModernBERT's per-layer FFN and MHA with SVD-factored versions
    that call your Triton kernels. Works for inference.
    """
    cfg = model.config
    H = cfg.hidden_size
    num_layers = getattr(cfg, "num_hidden_layers", None) or len(getattr(model.encoder, "layer", []))
    n_heads = cfg.num_attention_heads
    head_dim = H // n_heads

    # --- find encoder layers (works for most BERT-like impls) ---
    # ModernBERT typically: model.encoder.layer[i]
    layers = None
    for cand in ["encoder.layer", "encoder.layers", "model.layers"]:
        mod = model
        ok = True
        for name in cand.split("."):
            if hasattr(mod, name):
                mod = getattr(mod, name)
            else:
                ok = False
                break
        if ok:
            layers = mod
            break
    if layers is None:
        raise RuntimeError("Could not locate encoder layers on this ModernBERT variant.")

    # Rotary embedding object (grab from the first layer's attention if present)
    # Fallback: build from config if needed.
    rotary_emb = None
    try:
        att0 = layers[0].attention
        rotary_emb = getattr(att0, "rotary_emb", None) or getattr(att0.self, "rotary_emb", None)
    except Exception:
        pass
    if rotary_emb is None:
        raise RuntimeError("Could not find rotary_emb on ModernBERT attention; expose and pass it in if needed.")

    # --- iterate layers and replace ---
    for i, layer in enumerate(layers):
        # ----- locate vanilla modules -----
        # Attention pieces
        att = getattr(layer, "attention", None) or getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if att is None:
            raise RuntimeError(f"Layer {i}: attention module not found.")
        # HF patterns
        q_lin = getattr(att, "q_proj", None) or getattr(att, "query", None) or getattr(att, "self", None).query
        k_lin = getattr(att, "k_proj", None) or getattr(att, "key", None)   or getattr(att, "self", None).key
        v_lin = getattr(att, "v_proj", None) or getattr(att, "value", None) or getattr(att, "self", None).value
        o_lin = getattr(att, "out_proj", None) or getattr(att, "output", None).dense

        # FFN pieces (GeGLU)
        ffn = getattr(layer, "intermediate", None) or getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
        ffn_out = getattr(layer, "output", None) or getattr(layer, "mlp_out", None)
        if ffn is None or ffn_out is None:
            raise RuntimeError(f"Layer {i}: FFN modules not found.")

        # dense1 produces 2*inter_size for GeGLU; dense2 projects back to hidden
        dense1 = getattr(ffn, "dense", None) or getattr(ffn, "fc1", None) or getattr(ffn, "gate_proj", None)
        dense2 = getattr(ffn_out, "dense", None) or getattr(ffn_out, "fc2", None) or getattr(ffn_out, "down_proj", None)
        if dense1 is None or dense2 is None:
            raise RuntimeError(f"Layer {i}: FFN dense modules not found.")

        # ----- build replacements -----
        inter = getattr(cfg, "intermediate_size", getattr(dense2, "in_features", None))
        device = dense1.weight.device; dtype = dense1.weight.dtype

        # attention
        mha = SVDRoPEMHA(H, n_heads, head_dim, rotary_emb,
                         rq=r_q, rk=r_k, rv=r_v, ro=r_o,
                         device=device, dtype=dtype)
        mha.load_from_dense_qkv_o(q_lin, k_lin, v_lin, o_lin, r_q, r_k, r_v, r_o)

        # ffn
        ffn_new = FlashSVDGeGLUFFN(H, inter, r1=r_ffn1, r2=r_ffn2,
                                   gelu_approx=gelu_approx, device=device, dtype=dtype)
        ffn_new.load_from_dense(dense1, dense2, r1=r_ffn1, r2=r_ffn2)

        # ----- drop them in -----
        # Keep layernorms and dropouts/residuals owned by the original layer.
        # Replace call-sites: we replace the "attention" submodule with a tiny shim
        # that matches the original forward signature (hidden_states, attn_mask, etc.)
        class _AttnShim(nn.Module):
            def __init__(self, core: SVDRoPEMHA, orig_att):
                super().__init__()
                self.core = core
                # preserve layernorm and dropout if attention block owns them
                self.LayerNorm = getattr(orig_att, "LayerNorm", None) or getattr(layer, "attention_output", None)
                self.dropout = getattr(orig_att, "dropout", None)
                self.is_decoder = getattr(orig_att, "is_decoder", False)

            def forward(self, hidden_states, attention_mask=None, head_mask=None,
                        encoder_hidden_states=None, encoder_attention_mask=None,
                        past_key_value=None, output_attentions=False,
                        position_ids=None, sliding_window_mask=None, **kwargs):
                # position_ids is required for RoPE kernel; if missing, build 0..M-1
                if position_ids is None:
                    B, M, _ = hidden_states.shape
                    position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, -1)
                y = self.core(hidden_states, attention_mask, position_ids, sliding_window_mask)
                if self.dropout is not None:
                    y = self.dropout(y)
                return (y, None) if output_attentions else (y,)

        class _FFNShim(nn.Module):
            def __init__(self, core: FlashSVDGeGLUFFN, orig_layer):
                super().__init__()
                self.core = core
                # preserve dropout and LayerNorm in the parent layer (ModernBERT usually keeps them outside)
                self.dropout = getattr(orig_layer.output, "dropout", None)

            def forward(self, hidden_states):
                y = self.core(hidden_states)
                if self.dropout is not None:
                    y = self.dropout(y)
                return y

        setattr(layer, "attention", _AttnShim(mha, att))
        # Replace the FFN’s two-linears with a single fused core behind a shim:
        setattr(layer, "fused_ffn_core", _FFNShim(ffn_new, layer))

        # monkeypatch the layer forward to call our fused ffn if necessary
        if not hasattr(layer, "_flashsvd_patched"):
            orig_forward = layer.forward
            def patched_forward(*args, **kwargs):
                # call original to get its standard residual/norm plumbing,
                # but swap out the FFN call by temporarily hijacking modules
                # Easiest: rely on ModernBERT’s pattern (attn → add+norm → ffn → add+norm).
                out = orig_forward(*args, **kwargs)
                return out
            layer._flashsvd_patched = True  # marker (we keep original forward; attention was replaced)
        # For many ModernBERT variants, just replacing `attention` and the two dense modules is enough:
        setattr(ffn, "dense", nn.Identity())  # avoid double compute
        setattr(ffn_out, "dense", nn.Identity())
        # Route FFN through our fused core by patching the layer's call site if needed:
        if hasattr(layer, "mlp"):
            layer.mlp = layer.fused_ffn_core
        elif hasattr(layer, "intermediate") and hasattr(layer, "output"):
            # If the layer calls: x = intermediate(x); x = output(x)
            # we can wrap by replacing `intermediate` with identity and `output` with our fused core
            layer.intermediate = nn.Identity()
            layer.output = layer.fused_ffn_core

    return model




import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
model.eval()

# >>> IMPORTANT <<<
# Make sure your two kernels are imported/defined in this file:
# - flashsvd_ffn_geglu_fused(...)
# - class FlashSVDRoPEAttention(nn.Module): ...

# Patch the model in-place (choose your ranks)
instrument_modernbert_with_flashsvd(
    model,
    r_ffn1=128, r_ffn2=128,
    r_q=64, r_k=64, r_v=64, r_o=64,
    gelu_approx="tanh",
)

inp = tok("hello world", return_tensors="pt")
with torch.no_grad():
    out = model(**inp)
