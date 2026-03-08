#coding:utf8
import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.model_utils import find_layers
from flashsvd_component.svd_llama import (
    SVD_LlamaAttention,
    SVD_LlamaMLP,
    enable_flashsvd_llama_layer_tail_cuda_graph,
)
from flashsvd_component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from flashsvd_component.svd_opt import SVDOPTDecoderLayer

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
enable_flashsvd_llama_layer_tail_cuda_graph()


def _compat_enabled(key: str, default: bool = False) -> bool:
    """Check compatibility flags from environment.

    Supports SVDLLM_COMPAT_ALL=1 to enable everything, or per-feature toggles:
    SVDLLM_COMPAT_WHITENING, _RANKS, _ATTENTION.
    """
    if os.getenv('SVDLLM_COMPAT_ALL', '0') != '0':
        return True
    return os.getenv(f'SVDLLM_COMPAT_{key.upper()}', '1' if default else '0') != '0'


def _sqrtm_svd_spd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetric matrix square-root via eigendecomposition with eigenvalue clipping."""
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(mat.shape)}")
    mat = (mat + mat.t()) * 0.5
    w, Q = torch.linalg.eigh(mat)
    w = torch.clamp(w, min=float(eps))
    return (Q * torch.sqrt(w)) @ Q.t()


def truncated_svd(W: torch.Tensor, rank: int):
    """Compute full (economy) SVD, then truncate to the top-k singular values."""
    if rank <= 0:
        raise ValueError("rank must be positive.")
    U, S, VT = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], VT[:k, :]


def randomized_svd(W: torch.Tensor, rank: int, niter: int = 2, oversample: int = 5):
    """Approximate top-k SVD via randomized low-rank factorization."""
    if rank <= 0:
        raise ValueError("rank must be positive.")
    q = min(rank + max(0, int(oversample)), min(W.shape))
    try:
        U, S, V = torch.svd_lowrank(W, q=q, niter=max(0, int(niter)))
    except Exception as e:
        raise RuntimeError(f"torch.svd_lowrank failed: {e}")
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], V[:, :k].T


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    """SVD-LLM v2 style profiling: compute whitening factor via symmetric sqrt (no Cholesky)."""
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model name for profiling: {name}")

    model = model.to(dev)
    prev_cache = getattr(model.config, 'use_cache', False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    compat_whitening = _compat_enabled('whitening', False)
    msg = "Start obtaining the whitening matrix (raw XTX, official compat)..." if compat_whitening else "Start obtaining the whitening matrix (centered covariance)..."
    print(msg)

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            x = x.view(-1, x.shape[-1])
        x = x.to(dtype=torch.float64, device=dev)
        if compat_whitening:
            module._acc += x.t().matmul(x)
        else:
            module._second += x.t().matmul(x)
            module._mean += x.sum(dim=0)
            module._count += x.shape[0]
        del x, output
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    handles = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            if compat_whitening:
                module._acc = torch.zeros((in_f, in_f), dtype=torch.float64, device=dev)
            else:
                module._second = torch.zeros((in_f, in_f), dtype=torch.float64, device=dev)
                module._mean = torch.zeros((in_f,), dtype=torch.float64, device=dev)
                module._count = 0
            handles.append(module.register_forward_hook(hook))

    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)

    for h in handles:
        h.remove()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()

    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for n in subset:
            if compat_whitening and hasattr(subset[n], "_acc"):
                subset[n]._acc = subset[n]._acc.cpu()
            elif hasattr(subset[n], "_second"):
                subset[n]._second = subset[n]._second.cpu()
                subset[n]._mean = subset[n]._mean.cpu()

    profiling_mat = {}
    print("Start SVD sqrt factorization (no Cholesky)...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for n in subset:
            if compat_whitening:
                if not hasattr(subset[n], "_acc"):
                    continue
                mat = subset[n]._acc.to(dev)
                dmean = mat.diag().abs().mean().item()
                eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                scaling = _sqrtm_svd_spd(mat, eps=eps)
                subset[n]._acc = None
            else:
                if not hasattr(subset[n], "_second"):
                    continue
                second = subset[n]._second.to(dev)
                mean = subset[n]._mean.to(dev)
                count = max(int(subset[n]._count), 1)
                cov = second / count - torch.outer(mean / count, mean / count)
                cov = (cov + cov.t()) * 0.5
                dmean = cov.diag().abs().mean().item()
                eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                scaling = _sqrtm_svd_spd(cov, eps=eps)
                subset[n]._second = None
                subset[n]._mean = None
                subset[n]._count = 0

            layer_profile[n] = scaling.cpu()
            del scaling
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    return profiling_mat


@torch.no_grad()
def whitening_hetero(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    attn_ratio: float = None,
    mlp_ratio: float = None,
    svd_method: str = "full",
    svd_niter: int = 2,
    svd_oversample: int = 5,
):
    """Whitening + SVD compression with heterogeneous ratios for attention vs MLP."""
    model.eval()
    attn_ratio = float(ratio) if attn_ratio is None else float(attn_ratio)
    mlp_ratio = float(ratio) if mlp_ratio is None else float(mlp_ratio)
    svd_method = (svd_method or "full").lower()

    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print(
        f"Start SVD decomposition after whitening (hetero): attn_ratio={attn_ratio}, mlp_ratio={mlp_ratio} ..."
    )
    compat_ranks = _compat_enabled('ranks', False)
    compat_attn = _compat_enabled('attention', False)
    svd_time_total = 0.0

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)

        # Replace Attn, MLP modules with different ratios
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(
                config=model.config, ratio=attn_ratio, compat_ranks=compat_ranks, compat_attention=compat_attn
            )
            svd_mlp = SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=mlp_ratio,
                compat_ranks=compat_ranks,
            )
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=attn_ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=mlp_ratio)
        elif 'opt' in model_name:
            # OPT has a single ratio in the decoder layer class; approximate with max of both.
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=max(attn_ratio, mlp_ratio))
        else:
            raise ValueError(f"Unsupported model name for whitening_hetero: {model_name}")

        for n in subset:
            orig_dtype = subset[n].weight.dtype
            W = subset[n].weight.data.to(dev, dtype=torch.float32)
            dtype = orig_dtype
            scaling_diag_matrix = profiling_mat[i][n].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(
                    scaling_diag_matrix.shape[0], device=dev, dtype=scaling_diag_matrix.dtype
                )
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)

            # Choose local ratio per module
            ln = str(n)
            if ("q_proj" in ln) or ("k_proj" in ln) or ("v_proj" in ln) or ("o_proj" in ln) or ("out_proj" in ln):
                local_ratio = attn_ratio
            elif ("gate_proj" in ln) or ("down_proj" in ln) or ("up_proj" in ln) or ("fc1" in ln) or ("fc2" in ln):
                local_ratio = mlp_ratio
            else:
                local_ratio = float(ratio)

            local_ratio = min(1.0, max(0.0, float(local_ratio)))
            max_rank = min(W.shape[0], W.shape[1])
            use_official_rank = compat_ranks or ("mistral" in model_name) or ("opt" in model_name)
            if use_official_rank:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * local_ratio / (W.shape[0] + W.shape[1]))
            else:
                num_s_after_trunc = int(max_rank * local_ratio)
            num_s_after_trunc = max(1, min(num_s_after_trunc, max_rank))

            t_svd_start = time.perf_counter()
            if svd_method == "randomized":
                U, S, VT = randomized_svd(W_scale, rank=num_s_after_trunc, niter=svd_niter, oversample=svd_oversample)
            elif svd_method == "truncated":
                U, S, VT = truncated_svd(W_scale, rank=num_s_after_trunc)
            else:
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            svd_time_total += time.perf_counter() - t_svd_start

            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            if 'opt' in model_name:
                if "q_proj" in n:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.q_u_proj.bias.data = prev_b.bias.data
                elif "k_proj" in n:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.k_u_proj.bias.data = prev_b.bias.data
                elif "v_proj" in n:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.v_u_proj.bias.data = prev_b.bias.data
                elif "out_proj" in n:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.out_u_proj.bias.data = prev_b.bias.data
                elif "fc1" in n:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    prev_fc1 = getattr(layer, 'fc1', None)
                    if prev_fc1 is None:
                        prev_fc1 = getattr(layer, 'fc1_u_proj', None)
                    if prev_fc1 is not None and getattr(prev_fc1, 'bias', None) is not None:
                        svd_decoder.fc1_u_proj.bias.data = prev_fc1.bias.data
                elif "fc2" in n:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    prev_fc2 = getattr(layer, 'fc2', None)
                    if prev_fc2 is None:
                        prev_fc2 = getattr(layer, 'fc2_u_proj', None)
                    if prev_fc2 is not None and getattr(prev_fc2, 'bias', None) is not None:
                        svd_decoder.fc2_u_proj.bias.data = prev_fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in n:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in n:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in n:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in n:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in n:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in n:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in n:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp

            # Keep Linear metadata in sync with the new weight shapes
            def _sync_linear_meta(lin_mod: nn.Linear):
                if not isinstance(lin_mod, nn.Linear):
                    return
                lin_mod.in_features = lin_mod.weight.shape[1]
                lin_mod.out_features = lin_mod.weight.shape[0]
                if lin_mod.bias is not None and lin_mod.bias.numel() != lin_mod.out_features:
                    new_bias = lin_mod.bias.new_zeros(lin_mod.out_features)
                    sz = min(lin_mod.bias.numel(), lin_mod.out_features)
                    new_bias[:sz] = lin_mod.bias.data[:sz]
                    lin_mod.bias = nn.Parameter(new_bias, requires_grad=lin_mod.bias.requires_grad)

            _sync_linear_meta(subset[n])
            try:
                if 'opt' not in model_name:
                    if 'q_proj' in n:
                        _sync_linear_meta(svd_attn.q_u_proj)
                        _sync_linear_meta(svd_attn.q_v_proj)
                    elif 'k_proj' in n:
                        _sync_linear_meta(svd_attn.k_u_proj)
                        _sync_linear_meta(svd_attn.k_v_proj)
                    elif 'v_proj' in n:
                        _sync_linear_meta(svd_attn.v_u_proj)
                        _sync_linear_meta(svd_attn.v_v_proj)
                    elif 'o_proj' in n:
                        _sync_linear_meta(svd_attn.o_u_proj)
                        _sync_linear_meta(svd_attn.o_v_proj)
                    elif 'gate_proj' in n:
                        _sync_linear_meta(svd_mlp.gate_u_proj)
                        _sync_linear_meta(svd_mlp.gate_v_proj)
                    elif 'down_proj' in n:
                        _sync_linear_meta(svd_mlp.down_u_proj)
                        _sync_linear_meta(svd_mlp.down_v_proj)
                    elif 'up_proj' in n:
                        _sync_linear_meta(svd_mlp.up_u_proj)
                        _sync_linear_meta(svd_mlp.up_v_proj)
            except Exception:
                pass

            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

        del layer
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> int:
    """Delegate runtime decode benchmarking to the shared FlashSVD benchmark entrypoint.

    This keeps the v2 compression utilities in this file while exposing a simple
    CLI for llama-7b checkpoint inference-speed tests:

        python SVDLLM_v2_flashsvd.py --checkpoint <ckpt> ...
    """

    from bench_flashsvd_vs_svd_decode import main as bench_main

    # This file imports and enables the HF decoder-layer graph patch eagerly.
    # Keep the wrapper conservative: default to the MLP-only graph scope unless
    # the caller explicitly overrides it.
    if "--mlp_cuda_graph_scope" not in sys.argv:
        sys.argv.extend(["--mlp_cuda_graph_scope", "mlp"])

    return int(bench_main())


if __name__ == "__main__":
    raise SystemExit(main())
