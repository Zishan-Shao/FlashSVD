#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import (
    SVD_LlamaAttention,
    SVD_LlamaMLP,
    enable_flashsvd_llama_layer_tail_cuda_graph,
)
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
enable_flashsvd_llama_layer_tail_cuda_graph()


def _compat_enabled(key: str, default: bool = False) -> bool:
    """Check compatibility flags from environment.

    Supports SVDLLM_COMPAT_ALL=1 to enable everything, or
    per-feature toggles: SVDLLM_COMPAT_WHITENING, _RANKS, _ATTENTION.
    """
    if os.getenv('SVDLLM_COMPAT_ALL', '0') != '0':
        return True
    return os.getenv(f'SVDLLM_COMPAT_{key.upper()}', '1' if default else '0') != '0'


def _ensure_tokenizer(tokenizer_obj, model_id: str, hf_token: str = None):
    """Return a callable HF tokenizer. If the provided object is invalid, reload it.

    Some environments or legacy checkpoints may produce a non-callable placeholder
    (e.g., a bool) for the tokenizer field. This helper ensures we always pass a
    proper tokenizer to data utilities.
    """
    try:
        if callable(tokenizer_obj):
            return tokenizer_obj
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True, token=hf_token
        )
        return tok
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False, token=hf_token
        )
        return tok



'''
# Baseline (original model): (reduce sequence_length to make inference faster)
# --model_path original loads the unmodified HF model and evaluates PPL on wikitext2. It’s your reference to compare against whitening/compressed checkpoints.
CUDA_VISIBLE_DEVICES=2 python SVDLLM.py --model openlm-research/open_llama_7b --model_path original --step 4 --model_seq_len 2048 --eval_batch_size 2


# Whitened/compressed:
CUDA_VISIBLE_DEVICES=6 python SVDLLM.py --model openlm-research/open_llama_7b --model_path "./openlm_research_open_llama_7b_whitening_only_0.8.pt" --step 4 --model_seq_len 2048 --eval_batch_size 4


# in 20% compression case
# Baseline, 08:14 min:sec
Baseline PPL: {'wikitext2': 6.607561713533838}
Weight Memory: 26778.193359375 MiB

# SVD-LLM, 07:57 min:sec 
PPL after pruning: {'wikitext2': 8.652321061319611}
Weight Memory: 20995.66 MiB
Peak Memory (allocated): 27885.11 MiB
Peak Memory (reserved):  30610.00 MiB

# with FlashAttention: 06:16
PPL after pruning: {'wikitext2': 8.65232312419395}
Weight Memory: 20995.66 MiB
Peak Memory (allocated): 24909.11 MiB
Peak Memory (reserved):  26794.00 MiB


# 40% compressed (reduce by 60%):
CUDA_VISIBLE_DEVICES=7 python SVDLLM.py --model openlm-research/open_llama_7b --model_path ./jeffwan_llama_7b_hf_whitening_only_0.4.pt --step 4 --model_seq_len 2048 --eval_batch_size 4


PPL after pruning: {'wikitext2': 2903.5490535154536}
Weight Memory: 10945.34 MiB
Peak Memory (allocated): 14908.80 MiB
Peak Memory (reserved):  17276.00 MiB



'''


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    # Disable KV cache during profiling to avoid HF Cache interactions with custom layers
    prev_cache = getattr(model.config, 'use_cache', False)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    compat_whitening = _compat_enabled('whitening', False)
    print("Start obtaining the whitening matrix (centered covariance)..." if not compat_whitening else "Start obtaining the whitening matrix (raw XTX, official compat)...")
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
        del x
        torch.cuda.empty_cache()
    handles = []
    for mod_name, module in model.named_modules():
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
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            # Move stats to CPU for factorization
            if compat_whitening and hasattr(subset[name], "_acc"):
                subset[name]._acc = subset[name]._acc.cpu()
            elif hasattr(subset[name], "_second"):
                subset[name]._second = subset[name]._second.cpu()
                subset[name]._mean = subset[name]._mean.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            if compat_whitening:
                if not hasattr(subset[name], "_acc"):
                    continue
                raw_xtx = subset[name]._acc.to(dev)
                try:
                    scaling = torch.linalg.cholesky(raw_xtx)
                except Exception:
                    # Official fix: shift by (-min_eig + 1e-6) I
                    evals = torch.linalg.eigvalsh(raw_xtx)
                    min_e = float(evals.min().item())
                    shift = (-min_e + 1e-6) if min_e < 0 else 1e-6
                    scaling = torch.linalg.cholesky(raw_xtx + shift * torch.eye(raw_xtx.shape[0], device=dev, dtype=raw_xtx.dtype))
            else:
                if not hasattr(subset[name], "_second"):
                    continue
                second = subset[name]._second.to(dev)
                mean = subset[name]._mean.to(dev)
                count = max(int(subset[name]._count), 1)
                cov = second / count - torch.outer(mean / count, mean / count)
                cov = (cov + cov.t()) * 0.5
                try:
                    scaling = torch.linalg.cholesky(cov)
                except Exception:
                    # First try scale-aware diagonal jitter
                    dmean = cov.diag().abs().mean().item()
                    eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                    evals = torch.linalg.eigvalsh(cov)
                    min_e = float(evals.min().item())
                    shift = max(eps - min_e, 0.0)
                    cov_j = cov + shift * torch.eye(cov.shape[0], device=dev, dtype=cov.dtype)
                    try:
                        scaling = torch.linalg.cholesky(cov_j)
                    except Exception:
                        # Fallback: eigenvalue clipping to the nearest SPD, then Cholesky
                        w, Q = torch.linalg.eigh(cov)
                        w = torch.clamp(w, min=eps)
                        cov_spd = (Q * w) @ Q.T
                        scaling = torch.linalg.cholesky(cov_spd)
            layer_profile[name] = scaling.cpu()
            # cleanup
            if compat_whitening:
                subset[name]._acc = None
            else:
                subset[name]._second = None
                subset[name]._mean = None
                subset[name]._count = 0
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    # Restore original cache setting
    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(
    model_name,
    model,
    calib_loader,
    dev,
    *,
    stats_device: str = "auto",        # auto|cpu|cuda
    stats_dtype: str = "fp32",         # fp32|fp64
    store_dtype: str = "fp16",         # fp16|bf16|fp32
    microbatch: int = 1,
):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    # Resolve stats device/dtypes
    stats_device = (stats_device or "auto").lower()
    stats_dtype = (stats_dtype or "fp32").lower()
    store_dtype = (store_dtype or "fp16").lower()
    stats_t = torch.float64 if stats_dtype == "fp64" else torch.float32
    store_t = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(store_dtype, torch.float16)
    use_gpu_stats = str(dev).startswith("cuda") and stats_device in ("auto", "cuda")

    dtype = next(iter(model.parameters())).dtype
    # Use CPU-pinned buffers for activations to avoid large GPU allocations
    # and move slices to GPU on-the-fly when running each layer.
    use_cpu_buffers = True if str(dev).startswith('cuda') else False
    buf_device = torch.device('cpu') if use_cpu_buffers else dev
    pin = True if use_cpu_buffers else False
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=buf_device,
        pin_memory=pin,
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # Keep tensors on the same device as 'inps' (CPU-pinned if enabled)
            inps[cache['i']] = inp.to(buf_device, non_blocking=True)
            cache['i'] += 1
            # Capture attention mask if present; leave as None otherwise
            am = kwargs.get('attention_mask', None)
            if am is not None:
                am = am.to(buf_device, non_blocking=True)
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = am
                else:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], am), dim=0)
            # Capture position ids (non-OPT) if present
            if "opt" not in model_name and 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                pid = kwargs['position_ids']
                if cache['position_ids'] is None:
                    cache['position_ids'] = pid.to(buf_device, non_blocking=True)
                else:
                    cache['position_ids'] = torch.cat((cache['position_ids'], pid.to(buf_device, non_blocking=True)), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        compat_whitening = _compat_enabled('whitening', False)
        cov_cache = {}
        def hook(module, input, output):
            orig_x = input[0].detach()
            if orig_x.dim() == 3:
                orig_x2d = orig_x.reshape(-1, orig_x.shape[-1])
            elif orig_x.dim() == 2:
                orig_x2d = orig_x
            else:
                orig_x2d = orig_x.view(-1, orig_x.shape[-1])

            # Cache X^T X + sum(X) per unique input tensor in this forward call
            key = (int(orig_x2d.data_ptr()), int(orig_x2d.shape[0]), int(orig_x2d.shape[1]))
            cached = cov_cache.get(key, None)
            if cached is None:
                if compat_whitening and hasattr(module, "_acc"):
                    stats_dev = module._acc.device
                elif hasattr(module, "_second"):
                    stats_dev = module._second.device
                else:
                    stats_dev = dev
                x = orig_x2d.to(device=stats_dev, dtype=stats_t)
                xtx = x.t().matmul(x)
                xsum = x.sum(dim=0)
                nrow = int(x.shape[0])
                cov_cache[key] = (xtx, xsum, nrow)
            else:
                xtx, xsum, nrow = cached

            if compat_whitening:
                module._acc += xtx
            else:
                module._second += xtx
                module._mean += xsum
                module._count += nrow
            del output
        handles = []
        for name in subset:
            if isinstance(subset[name], nn.Linear):
                in_f = subset[name].in_features
                # Prefer accumulating whitening statistics on GPU for speed; fall back to CPU on OOM.
                stats_dev = dev if use_gpu_stats else torch.device("cpu")
                try:
                    if compat_whitening:
                        subset[name]._acc = torch.zeros((in_f, in_f), dtype=stats_t, device=stats_dev)
                    else:
                        subset[name]._second = torch.zeros((in_f, in_f), dtype=stats_t, device=stats_dev)
                        subset[name]._mean = torch.zeros((in_f,), dtype=stats_t, device=stats_dev)
                        subset[name]._count = 0
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and str(stats_dev).startswith("cuda"):
                        torch.cuda.empty_cache()
                        if compat_whitening:
                            subset[name]._acc = torch.zeros((in_f, in_f), dtype=stats_t, device="cpu")
                        else:
                            subset[name]._second = torch.zeros((in_f, in_f), dtype=stats_t, device="cpu")
                            subset[name]._mean = torch.zeros((in_f,), dtype=stats_t, device="cpu")
                            subset[name]._count = 0
                    else:
                        raise
            handles.append(subset[name].register_forward_hook(hook))
        mb = max(int(microbatch), 1)
        n = int(inps.shape[0])
        for j0 in range(0, n, mb):
            j1 = min(n, j0 + mb)
            cov_cache.clear()

            amj = None if attention_masks is None else attention_masks[j0:j1]
            if amj is not None and amj.device != dev:
                amj = amj.to(dev, non_blocking=True)

            xj = inps[j0:j1].to(dev, non_blocking=True)
            if "opt" not in model_name:
                pidsj = None if cache.get('position_ids', None) is None else position_ids[j0:j1].to(dev, non_blocking=True)
                pos_emb = None
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                        pos_emb = model.model.rotary_emb(xj, pidsj)
                except Exception:
                    pos_emb = None
                try:
                    yj = layer(xj, attention_mask=amj, position_ids=pidsj, position_embeddings=pos_emb)[0]
                except TypeError:
                    yj = layer(xj, attention_mask=amj, position_ids=pidsj)[0]
            else:
                yj = layer(xj, attention_mask=amj)[0]

            # Overwrite activation buffer in-place to save memory.
            inps[j0:j1] = yj.to(buf_device, non_blocking=True)
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            compat_whitening = _compat_enabled('whitening', False)
            if compat_whitening:
                if not hasattr(subset[name], "_acc"):
                    continue
                raw_xtx = subset[name]._acc
                try:
                    scaling = torch.linalg.cholesky(raw_xtx)
                except Exception:
                    evals = torch.linalg.eigvalsh(raw_xtx)
                    min_e = float(evals.min().item())
                    shift = (-min_e + 1e-6) if min_e < 0 else 1e-6
                    scaling = torch.linalg.cholesky(
                        raw_xtx + shift * torch.eye(raw_xtx.shape[0], device=raw_xtx.device, dtype=raw_xtx.dtype)
                    )
                layer_profile[name] = scaling.to(dtype=store_t).cpu()
                subset[name]._acc = None
            else:
                if not hasattr(subset[name], "_second"):
                    continue
                # Keep computations on the device where stats live (CPU if offloaded)
                second = subset[name]._second
                mean = subset[name]._mean
                count = max(int(subset[name]._count), 1)
                cov = second / count - torch.outer(mean / count, mean / count)
                cov = (cov + cov.t()) * 0.5
                try:
                    scaling = torch.linalg.cholesky(cov)
                except Exception:
                    # First try scale-aware diagonal jitter
                    dmean = cov.diag().abs().mean().item()
                    eps = 1e-6 * (dmean if dmean > 0 else 1.0)
                    evals = torch.linalg.eigvalsh(cov)
                    min_e = float(evals.min().item())
                    shift = max(eps - min_e, 0.0)
                    cov_j = cov + shift * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
                    try:
                        scaling = torch.linalg.cholesky(cov_j)
                    except Exception:
                        # Fallback: eigenvalue clipping to the nearest SPD, then Cholesky
                        w, Q = torch.linalg.eigh(cov)
                        w = torch.clamp(w, min=eps)
                        cov_spd = (Q * w) @ Q.T
                        scaling = torch.linalg.cholesky(cov_spd)
                layer_profile[name] = scaling.to(dtype=store_t).cpu()
                # cleanup
                subset[name]._second = None
                subset[name]._mean = None
                subset[name]._count = 0
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        torch.cuda.empty_cache()
    return profiling_mat
     
 
@torch.no_grad()
def whitening(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    *,
    svd_lowrank: bool = False,
    svd_oversample: int = 64,
    svd_niter: int = 2,
):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            compat_ranks = _compat_enabled('ranks', False)
            compat_attn = _compat_enabled('attention', False)
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, compat_ranks=compat_ranks, compat_attention=compat_attn)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, compat_ranks=compat_ranks)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        #### Replace Attn, MLP ####
        for name in subset:
            orig_dtype = subset[name].weight.dtype
            # Compute SVD in float32 for stability, cast back to original dtype for storage.
            W = subset[name].weight.data.to(dev, dtype=torch.float32)
            dtype = orig_dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev, dtype=torch.float32)
            # Small diagonal jitter for numerical safety (matches common whitening practice).
            try:
                W_scale = torch.matmul(W, scaling_diag_matrix)
            except Exception:
                eps = 1e-6
                scaling_diag_matrix = scaling_diag_matrix + eps * torch.eye(
                    scaling_diag_matrix.shape[0], device=scaling_diag_matrix.device, dtype=scaling_diag_matrix.dtype
                )
                W_scale = torch.matmul(W, scaling_diag_matrix)
            # Official SVD-LLM rank mapping: keep_ratio ~= params_ratio
            # For W in R^{m x n}, choose k such that k(m+n) ≈ ratio * (m*n).
            max_rank = min(W.shape[0], W.shape[1])
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, min(num_s_after_trunc, max_rank))

            if svd_lowrank and num_s_after_trunc < max_rank:
                q = min(max_rank, int(num_s_after_trunc) + int(max(svd_oversample, 0)))
                U_lr, S_lr, V_lr = torch.svd_lowrank(W_scale, q=q, niter=int(max(svd_niter, 0)))
                truc_u = U_lr[:, :num_s_after_trunc]
                truc_s = S_lr[:num_s_after_trunc]
                VT_k = V_lr[:, :num_s_after_trunc].t().contiguous()  # (k, n)
            else:
                U_full, S_full, VT_full = torch.linalg.svd(W_scale, full_matrices=False)
                truc_u = U_full[:, :num_s_after_trunc]
                truc_s = S_full[:num_s_after_trunc]
                VT_k = VT_full[:num_s_after_trunc, :].contiguous()

            # Avoid forming inv(L): solve right-triangular system for VT_k @ inv(L)
            # We want X = VT_k @ inv(L) where L = scaling_diag_matrix (lower-triangular).
            # Equivalent: (X L)^T = VT_k^T -> L^T X^T = VT_k^T.
            truc_v = torch.linalg.solve_triangular(scaling_diag_matrix.t(), VT_k.t(), upper=True).t()
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    # copy bias from either HF layer (q_proj) or existing SVD layer (q_u_proj)
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.q_u_proj.bias.data = prev_b.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.k_u_proj.bias.data = prev_b.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.v_u_proj.bias.data = prev_b.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.out_u_proj.bias.data = prev_b.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    prev_fc1 = getattr(layer, 'fc1', None)
                    if prev_fc1 is None:
                        prev_fc1 = getattr(layer, 'fc1_u_proj', None)
                    if prev_fc1 is not None and getattr(prev_fc1, 'bias', None) is not None:
                        svd_decoder.fc1_u_proj.bias.data = prev_fc1.bias.data
                elif "fc2" in name:
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
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    # Preserve HF layer index for Cache objects (transformers>=4.4x).
                    try:
                        svd_attn.layer_idx = int(getattr(layer.self_attn, "layer_idx", i))
                    except Exception:
                        svd_attn.layer_idx = int(i)
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
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

            # Sync original HF linear (subset[name]) metadata
            _sync_linear_meta(subset[name])
            # Also sync SVD modules if they exist (important for LoRA wrappers)
            try:
                if 'opt' not in model_name:
                    if 'q_proj' in name:
                        _sync_linear_meta(svd_attn.q_u_proj)
                        _sync_linear_meta(svd_attn.q_v_proj)
                    elif 'k_proj' in name:
                        _sync_linear_meta(svd_attn.k_u_proj)
                        _sync_linear_meta(svd_attn.k_v_proj)
                    elif 'v_proj' in name:
                        _sync_linear_meta(svd_attn.v_u_proj)
                        _sync_linear_meta(svd_attn.v_v_proj)
                    elif 'o_proj' in name:
                        _sync_linear_meta(svd_attn.o_u_proj)
                        _sync_linear_meta(svd_attn.o_v_proj)
                    elif 'gate_proj' in name:
                        _sync_linear_meta(svd_mlp.gate_u_proj)
                        _sync_linear_meta(svd_mlp.gate_v_proj)
                    elif 'down_proj' in name:
                        _sync_linear_meta(svd_mlp.down_u_proj)
                        _sync_linear_meta(svd_mlp.down_v_proj)
                    elif 'up_proj' in name:
                        _sync_linear_meta(svd_mlp.up_u_proj)
                        _sync_linear_meta(svd_mlp.up_v_proj)
            except Exception:
                pass
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False, update_us=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    # Only LLaMA/Mistral-like models have top-level model.norm
    if "opt" not in model_name:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(dev)
            cache['i'] += 1
            am = kwargs.get('attention_mask', None)
            if am is not None:
                am = am.to(dev)
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = am
                else:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], am), dim=0)
            if "opt" not in model_name and 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                if cache['position_ids'] is None:
                    cache['position_ids'] = kwargs['position_ids'].to(dev)
                else:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].to(dev)), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        layer_err = 0.0
        layer_upd_err = 0.0
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update, update_us=update_us)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            # attention_masks can be None; OPT layer will handle internal causal mask
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'q_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.q_u_proj.bias.data = prev_b.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'k_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.k_u_proj.bias.data = prev_b.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'v_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.v_u_proj.bias.data = prev_b.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_proj', None)
                    if prev_b is None:
                        prev_b = getattr(getattr(layer, 'self_attn', layer), 'out_u_proj', None)
                    if prev_b is not None and getattr(prev_b, 'bias', None) is not None:
                        svd_decoder.self_attn.out_u_proj.bias.data = prev_b.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    prev_fc1 = getattr(layer, 'fc1', None)
                    if prev_fc1 is None:
                        prev_fc1 = getattr(layer, 'fc1_u_proj', None)
                    if prev_fc1 is not None and getattr(prev_fc1, 'bias', None) is not None:
                        svd_decoder.fc1_u_proj.bias.data = prev_fc1.bias.data
                elif "fc2" in name:
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
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            try:
                layer_err += gpts[name].error
                layer_upd_err += gpts[name].updated_error
            except Exception:
                pass
        # For OPT we materialize a new SVD layer; others are updated in-place
        layer_for_fwd = svd_decoder if "opt" in model_name else layer
        layer_for_fwd = layer_for_fwd.to(dev)
        if "opt" not in model_name:
            outs = layer_for_fwd(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer_for_fwd(inps, attention_mask=attention_masks)[0]
        layers[i] = layer_for_fwd.cpu()
        try:
            print(f"[LocalUpdate] layer {i}: err={layer_err:.6e} -> {layer_upd_err:.6e}")
        except Exception:
            pass
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False, update_us=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.update_us = update_us
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        # Accept 2D (BT x H) or 3D (B x T x H) tensors
        if inp.dim() == 3:
            inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        else:
            inps = inp
        if out.dim() == 3:
            outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        else:
            outs = out
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        denom = torch.norm(outs, p='fro').item() + 1e-12
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / denom
        if self.update_us:
            # Update both U and Sigma by solving for H^T with V fixed
            Z = torch.matmul(inps, self.truc_v.T)
            Ht = torch.linalg.lstsq(Z, outs).solution  # (r x d_out)
            H = Ht.t().contiguous()                   # (d_out x r)
            # SVD on H to get updated U and Sigma
            Uh, Sh, VTh = torch.linalg.svd(H, full_matrices=False)
            # Save for fasterprune
            self.Uh = Uh
            self.Sh = Sh
            updated_output = Z.matmul(Ht)
        else:
            # Update U only with Sigma fixed
            x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            self.updated_uT = torch.linalg.lstsq(x, outs).solution
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / denom
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        if self.update_us and hasattr(self, 'Uh'):
            sqrtS = torch.diag(torch.sqrt(self.Sh))
            self.appendU = self.Uh.matmul(sqrtS)
            self.appendV = sqrtS.matmul(self.truc_v)
        else:
            sqrtSigma = torch.sqrt(self.truc_sigma)
            self.appendU = self.updated_uT.t().matmul(sqrtSigma)
            self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face access token (optional)')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    # Speed / memory knobs for step 1/2
    parser.add_argument('--whitening_stats_device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Where to accumulate whitening XTX stats (auto=cuda when DEV is cuda)')
    parser.add_argument('--whitening_stats_dtype', type=str, default='fp32', choices=['fp32', 'fp64'], help='dtype for whitening stats accumulation')
    parser.add_argument('--whitening_store_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='dtype to store Cholesky whitening matrices in profiling_mat (saves RAM/disk)')
    parser.add_argument('--whitening_microbatch', type=int, default=1, help='microbatch size when iterating cached activations per layer (higher = faster, more GPU memory)')
    parser.add_argument('--svd_lowrank', action='store_true', help='Use randomized truncated SVD (torch.svd_lowrank) instead of full SVD for speed')
    parser.add_argument('--svd_oversample', type=int, default=64, help='Oversampling for randomized SVD (only when --svd_lowrank)')
    parser.add_argument('--svd_niter', type=int, default=2, help='Power iterations for randomized SVD (only when --svd_lowrank)')
    # Official-compat toggles
    parser.add_argument('--svdllm_compat_all', action='store_true', help='Enable all official-compat behaviors (whitening XTX, official ranks, explicit attention math).')
    parser.add_argument('--svdllm_compat_whitening', action='store_true', help='Use original whitening accumulation (raw X^T X without centering).')
    parser.add_argument('--svdllm_compat_ranks', action='store_true', help='Use original SVD rank formulas for attention/MLP modules.')
    parser.add_argument('--svdllm_compat_attention', action='store_true', help='Force explicit attention (matmul+softmax) and 3-value return like HF.')
    
    args = parser.parse_args()
    # Apply compat flags via environment for downstream modules
    if args.svdllm_compat_all:
        os.environ['SVDLLM_COMPAT_ALL'] = '1'
    if args.svdllm_compat_whitening:
        os.environ['SVDLLM_COMPAT_WHITENING'] = '1'
    if args.svdllm_compat_ranks:
        os.environ['SVDLLM_COMPAT_RANKS'] = '1'
    if args.svdllm_compat_attention:
        os.environ['SVDLLM_COMPAT_ATTENTION'] = '1'
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, hf_token=args.hf_token)
        tokenizer = _ensure_tokenizer(tokenizer, args.model, args.hf_token)
        model = model.eval()
        model.seqlen = args.model_seq_len
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(
                args.model,
                model,
                cali_white_data,
                args.DEV,
                stats_device=args.whitening_stats_device,
                stats_dtype=args.whitening_stats_dtype,
                store_dtype=args.whitening_store_dtype,
                microbatch=args.whitening_microbatch,
            )
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening(
            args.model,
            model,
            profiling_mat,
            args.ratio,
            args.DEV,
            svd_lowrank=bool(args.svd_lowrank),
            svd_oversample=int(args.svd_oversample),
            svd_niter=int(args.svd_niter),
        )
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, hf_token=args.hf_token)
        tokenizer = _ensure_tokenizer(tokenizer, args.model, args.hf_token)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model.seqlen = args.model_seq_len
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(
                args.model,
                model,
                cali_white_data,
                args.DEV,
                stats_device=args.whitening_stats_device,
                stats_dtype=args.whitening_stats_dtype,
                store_dtype=args.whitening_store_dtype,
                microbatch=args.whitening_microbatch,
            )
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
        tokenizer = _ensure_tokenizer(tokenizer, args.model, args.hf_token)
        model = model.eval()
        model.seqlen = args.model_seq_len
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.seqlen = args.model_seq_len
        model.eval()
        # Optional dtype override for evaluation to control GPU memory
        # Default behavior preserved (float32) when no override is set.
        try:
            from argparse import SUPPRESS as _SUPPRESS
        except Exception:
            _SUPPRESS = None
        # Backward compatible: allow --dtype from CLI to select eval dtype if provided
        eval_dtype = None
        try:
            # argparse may include args.dtype if our caller provided it
            eval_dtype = getattr(args, 'dtype', None)
        except Exception:
            eval_dtype = None
        if isinstance(eval_dtype, str) and eval_dtype:
            _map = {
                'float16': torch.float16, 'fp16': torch.float16,
                'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
                'float32': torch.float32, 'fp32': torch.float32,
            }
            tgt = _map.get(eval_dtype.lower(), None)
            if tgt is not None:
                model = model.to(dtype=tgt)
        else:
            # Preserve original default behavior
            model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            label = 'Baseline PPL' if args.model_path == 'original' else 'PPL after pruning'
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV, label=label)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
