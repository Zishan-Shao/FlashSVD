#coding:utf8
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.model_utils import find_layers
from models.llama import (
    SVD_LlamaAttention,
    SVD_LlamaMLP,
    enable_flashsvd_llama_layer_tail_cuda_graph,
)
from models.mistral import SVD_MistralAttention, SVD_MistralMLP
from models.opt import SVDOPTDecoderLayer
enable_flashsvd_llama_layer_tail_cuda_graph()


def _sqrtm_svd_spd(mat: torch.Tensor, eps: float) -> torch.Tensor:
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(mat.shape)}")
    mat = (mat + mat.t()) * 0.5
    w, Q = torch.linalg.eigh(mat)
    w = torch.clamp(w, min=float(eps))
    return (Q * torch.sqrt(w)) @ Q.t()


def truncated_svd(W: torch.Tensor, rank: int):
    if rank <= 0:
        raise ValueError("rank must be positive.")
    U, S, VT = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], VT[:k, :]


def randomized_svd(W: torch.Tensor, rank: int, niter: int = 2, oversample: int = 5):
    if rank <= 0:
        raise ValueError("rank must be positive.")
    q = min(rank + max(0, int(oversample)), min(W.shape))
    try:
        U, S, V = torch.svd_lowrank(W, q=q, niter=max(0, int(niter)))
    except Exception as e:
        raise RuntimeError(f"torch.svd_lowrank failed: {e}")
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], V[:, :k].T


def _get_layers(model_name: str, model):
    if "opt" in model_name:
        return model.model.decoder.layers
    return model.model.layers


def _weight_type_key(name: str) -> Optional[str]:
    for key in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
    ):
        if key in name:
            return key
    return None


def _is_attn_weight(name: str) -> bool:
    return _weight_type_key(name) in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}


def _is_mlp_weight(name: str) -> bool:
    return _weight_type_key(name) in {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"}


def _rank_from_keep_ratio(out_features: int, in_features: int, keep_ratio: float) -> int:
    keep_ratio = min(1.0, max(0.0, float(keep_ratio)))
    k = int(out_features * in_features * keep_ratio / (out_features + in_features))
    return max(1, min(k, min(out_features, in_features)))


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


def _safe_log_score(loss: float, strict_paper_formula: bool) -> float:
    if loss <= 0.0:
        if strict_paper_formula:
            raise ValueError(
                f"[strict_paper_formula] Encountered non-positive Lmin={loss}. "
                "Paper does not specify a repair for this case."
            )
        loss = 1e-12
    lg = math.log(loss)
    if abs(lg) < 1e-12:
        if strict_paper_formula:
            raise ValueError(
                f"[strict_paper_formula] Encountered log(Lmin)=0 for Lmin={loss}. "
                "Paper does not specify a repair for this case."
            )
        lg = 1e-12
    return 1.0 / lg


def _select_ctor_ratio(subset, layer_keep: Dict[str, float], default_keep_ratio: float, family: str) -> float:
    vals = []
    for n in subset:
        if family == "attn" and _is_attn_weight(str(n)):
            vals.append(float(layer_keep.get(n, default_keep_ratio)))
        elif family == "mlp" and _is_mlp_weight(str(n)):
            vals.append(float(layer_keep.get(n, default_keep_ratio)))
    return max(vals) if vals else float(default_keep_ratio)


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev, *, raw_xtx: bool = True):
    """Profile whitening factors for SVD-LLM v2 hetero compression.

    By default this follows the paper-style raw X^T X branch instead of the
    centered covariance path used by the current engineering v2 script.
    """
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model name for profiling: {name}")

    model = model.to(dev)
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    msg = (
        "Start obtaining the whitening matrix (raw XTX, paper-style)..."
        if raw_xtx
        else "Start obtaining the whitening matrix (centered covariance, engineering fallback)..."
    )
    print(msg)

    def hook(module, input, output):
        x = input[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() != 2:
            x = x.view(-1, x.shape[-1])
        x = x.to(dtype=torch.float64, device=dev)
        if raw_xtx:
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
            if raw_xtx:
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
            if raw_xtx and hasattr(subset[n], "_acc"):
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
            if raw_xtx:
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


profile_svdllm = profle_svdllm


@torch.no_grad()
def allocate_svdllm_v2_adaptive_keep_ratios(
    model_name,
    model,
    profiling_mat,
    target_reduction_ratio: float,
    dev,
    strict_paper_formula: bool = True,
):
    """Allocate per-module keep ratios using a paper-faithful v2-style heuristic.

    `target_reduction_ratio` follows paper semantics: 0.20 means 20% parameter
    reduction. The returned keep ratios use engineering semantics: kept_params /
    original_params.
    """
    target_reduction_ratio = float(target_reduction_ratio)
    if not (0.0 < target_reduction_ratio < 1.0):
        raise ValueError(
            f"target_reduction_ratio must be in (0, 1), got {target_reduction_ratio}"
        )

    global_keep_ratio = 1.0 - target_reduction_ratio
    layers = _get_layers(model_name, model)
    grouped_losses = defaultdict(list)
    module_lmin = {}

    print(
        f"Start SVD-LLM v2 adaptive ratio allocation under paper compression ratio "
        f"R={target_reduction_ratio:.6f} ..."
    )

    for i in tqdm(range(len(layers))):
        subset = find_layers(layers[i])
        for n, mod in subset.items():
            wt_key = _weight_type_key(str(n))
            if wt_key is None or i not in profiling_mat or n not in profiling_mat[i]:
                continue

            W = mod.weight.detach().to(dev, dtype=torch.float32)
            S = profiling_mat[i][n].to(dev, dtype=torch.float32)
            D = W @ S
            sing_vals = torch.linalg.svdvals(D)
            k = _rank_from_keep_ratio(W.shape[0], W.shape[1], global_keep_ratio)

            if k >= sing_vals.numel():
                lmin = 0.0
            else:
                lmin = torch.linalg.vector_norm(sing_vals[k:], ord=2).item()

            module_lmin[(i, n)] = lmin
            grouped_losses[wt_key].append((i, n, lmin))

            del W, S, D, sing_vals
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

    module_reduce_ratios = {}
    module_keep_ratios = {}

    for wt_key, items in grouped_losses.items():
        scores = [_safe_log_score(float(lmin), strict_paper_formula) for _, _, lmin in items]
        denom = sum(scores)
        if abs(denom) < 1e-12:
            if strict_paper_formula:
                raise ValueError(
                    f"[strict_paper_formula] Sum of 1/log(Lmin) is ~0 in group={wt_key}. "
                    "Paper does not specify a repair for this case."
                )
            scores = [1.0] * len(items)
            denom = float(len(items))

        for (i, n, lmin), score in zip(items, scores):
            reduce_ratio = len(items) * target_reduction_ratio * score / denom
            if strict_paper_formula:
                if not (0.0 < reduce_ratio < 1.0):
                    raise ValueError(
                        f"[strict_paper_formula] Invalid allocated reduction ratio={reduce_ratio:.6f} "
                        f"for layer={i}, name={n}, group={wt_key}, Lmin={lmin:.6e}. "
                        "The paper formula produced an out-of-range value."
                    )
            else:
                reduce_ratio = min(max(float(reduce_ratio), 1e-6), 1.0 - 1e-6)

            keep_ratio = 1.0 - float(reduce_ratio)
            module_reduce_ratios[(i, n)] = float(reduce_ratio)
            module_keep_ratios[(i, n)] = float(keep_ratio)

    return module_keep_ratios, module_reduce_ratios, module_lmin


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
    module_keep_ratios: Optional[Dict[Tuple[int, str], float]] = None,
    force_param_count_rank: bool = True,
):
    """Whitening + SVD compression with per-module keep ratios.

    `ratio` is the fallback homogeneous keep ratio. When `module_keep_ratios` is
    provided, it takes priority over the coarse attn/mlp ratios.
    """
    model.eval()
    default_keep_ratio = float(ratio)
    attn_ratio = default_keep_ratio if attn_ratio is None else float(attn_ratio)
    mlp_ratio = default_keep_ratio if mlp_ratio is None else float(mlp_ratio)
    svd_method = (svd_method or "full").lower()
    layers = _get_layers(model_name, model)
    compat_attn = os.getenv("SVDLLM_COMPAT_ATTENTION", "0") != "0"
    svd_time_total = 0.0

    print("Start SVD decomposition after whitening (paper-faithful adaptive rank)...")

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)

        layer_keep = {}
        for n in subset:
            if module_keep_ratios is not None and (i, n) in module_keep_ratios:
                layer_keep[n] = float(module_keep_ratios[(i, n)])
            elif _is_attn_weight(str(n)):
                layer_keep[n] = attn_ratio
            elif _is_mlp_weight(str(n)):
                layer_keep[n] = mlp_ratio
            else:
                layer_keep[n] = default_keep_ratio

        attn_ctor_ratio = _select_ctor_ratio(subset, layer_keep, default_keep_ratio, "attn")
        mlp_ctor_ratio = _select_ctor_ratio(subset, layer_keep, default_keep_ratio, "mlp")

        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(
                config=model.config,
                ratio=attn_ctor_ratio,
                compat_ranks=force_param_count_rank,
                compat_attention=compat_attn,
            )
            svd_mlp = SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=mlp_ctor_ratio,
                compat_ranks=force_param_count_rank,
            )
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=attn_ctor_ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=mlp_ctor_ratio)
        elif "opt" in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=max(attn_ctor_ratio, mlp_ctor_ratio))
        else:
            raise ValueError(f"Unsupported model name for whitening_hetero: {model_name}")

        for n in subset:
            orig_dtype = subset[n].weight.dtype
            W = subset[n].weight.data.to(dev, dtype=torch.float32)
            dtype = orig_dtype
            scaling_diag_matrix = profiling_mat[i][n].to(dev, dtype=torch.float32)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception:
                scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(
                    scaling_diag_matrix.shape[0], device=dev, dtype=scaling_diag_matrix.dtype
                )
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

            W_scale = torch.matmul(W, scaling_diag_matrix)
            local_keep_ratio = min(1.0, max(0.0, float(layer_keep[n])))
            if force_param_count_rank:
                num_s_after_trunc = _rank_from_keep_ratio(W.shape[0], W.shape[1], local_keep_ratio)
            else:
                max_rank = min(W.shape[0], W.shape[1])
                num_s_after_trunc = max(1, min(int(max_rank * local_keep_ratio), max_rank))

            t_svd_start = time.perf_counter()
            if svd_method == "randomized":
                U, S, VT = randomized_svd(
                    W_scale,
                    rank=num_s_after_trunc,
                    niter=svd_niter,
                    oversample=svd_oversample,
                )
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

            if "opt" in model_name:
                if "q_proj" in n:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "q_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "q_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.q_u_proj.bias.data = prev_b.bias.data
                elif "k_proj" in n:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "k_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "k_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.k_u_proj.bias.data = prev_b.bias.data
                elif "v_proj" in n:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "v_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "v_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.v_u_proj.bias.data = prev_b.bias.data
                elif "out_proj" in n:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, "self_attn", layer), "out_proj", None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, "self_attn", layer), "out_u_proj", None)
                    if prev_b is not None and getattr(prev_b, "bias", None) is not None:
                        svd_decoder.self_attn.out_u_proj.bias.data = prev_b.bias.data
                elif "fc1" in n:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    prev_fc1 = getattr(layer, "fc1", None)
                    if prev_fc1 is None:
                        prev_fc1 = getattr(layer, "fc1_u_proj", None)
                    if prev_fc1 is not None and getattr(prev_fc1, "bias", None) is not None:
                        svd_decoder.fc1_u_proj.bias.data = prev_fc1.bias.data
                elif "fc2" in n:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    prev_fc2 = getattr(layer, "fc2", None)
                    if prev_fc2 is None:
                        prev_fc2 = getattr(layer, "fc2_u_proj", None)
                    if prev_fc2 is not None and getattr(prev_fc2, "bias", None) is not None:
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

            _sync_linear_meta(subset[n])
            try:
                if "opt" not in model_name:
                    if "q_proj" in n:
                        _sync_linear_meta(svd_attn.q_u_proj)
                        _sync_linear_meta(svd_attn.q_v_proj)
                    elif "k_proj" in n:
                        _sync_linear_meta(svd_attn.k_u_proj)
                        _sync_linear_meta(svd_attn.k_v_proj)
                    elif "v_proj" in n:
                        _sync_linear_meta(svd_attn.v_u_proj)
                        _sync_linear_meta(svd_attn.v_v_proj)
                    elif "o_proj" in n:
                        _sync_linear_meta(svd_attn.o_u_proj)
                        _sync_linear_meta(svd_attn.o_v_proj)
                    elif "gate_proj" in n:
                        _sync_linear_meta(svd_mlp.gate_u_proj)
                        _sync_linear_meta(svd_mlp.gate_v_proj)
                    elif "down_proj" in n:
                        _sync_linear_meta(svd_mlp.down_u_proj)
                        _sync_linear_meta(svd_mlp.down_v_proj)
                    elif "up_proj" in n:
                        _sync_linear_meta(svd_mlp.up_u_proj)
                        _sync_linear_meta(svd_mlp.up_v_proj)
            except Exception:
                pass

            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = None
            truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

        del layer
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Done SVD decomposition, total SVD time: {svd_time_total:.2f}s")
    return model


@torch.no_grad()
def compress_model_adaptive(
    model_name,
    model,
    calib_loader,
    target_reduction_ratio: float,
    dev,
    strict_paper_formula: bool = True,
    raw_xtx: bool = True,
    svd_method: str = "full",
    svd_niter: int = 2,
    svd_oversample: int = 5,
):
    """One-shot helper for paper-style v2 heterogeneous compression."""
    profiling_mat = profle_svdllm(model_name, model, calib_loader, dev, raw_xtx=raw_xtx)
    module_keep_ratios, module_reduce_ratios, module_lmin = allocate_svdllm_v2_adaptive_keep_ratios(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        target_reduction_ratio=target_reduction_ratio,
        dev=dev,
        strict_paper_formula=strict_paper_formula,
    )
    whitening_hetero(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        ratio=1.0 - float(target_reduction_ratio),
        dev=dev,
        svd_method=svd_method,
        svd_niter=svd_niter,
        svd_oversample=svd_oversample,
        module_keep_ratios=module_keep_ratios,
        force_param_count_rank=True,
    )
    return module_keep_ratios, module_reduce_ratios, module_lmin, profiling_mat
