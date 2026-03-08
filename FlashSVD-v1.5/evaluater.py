from __future__ import annotations
import os
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None

from utils.model_utils import measure_param_bytes, mib


def _iter_texts(dataset_name: str, split: str) -> Iterable[str]:
    if load_dataset is None:
        raise ImportError("datasets is required for evaluation; please `pip install datasets`.")

    name = dataset_name.lower()
    if name in {"wikitext2", "wikitext-2", "wiki2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        for ex in ds:
            txt = ex.get("text", "")
            if txt:
                yield txt
        return
    if name in {"ptb", "penn_treebank"}:
        ds = load_dataset("ptb_text_only", split=split)
        for ex in ds:
            txt = ex.get("sentence", ex.get("text", ""))
            if txt:
                yield txt
        return
    if name in {"c4", "c4_en"}:
        ds = load_dataset("c4", "en", split=split, streaming=True)
        for ex in ds:
            txt = ex.get("text", "")
            if txt:
                yield txt
        return
    raise ValueError(f"Unsupported dataset: {dataset_name} (supported: wikitext2, ptb, c4)")


def _token_stream(dataset_name: str, tokenizer, split: str, *, max_texts: Optional[int] = None) -> List[int]:
    eos = tokenizer.eos_token_id
    ids: List[int] = []
    n = 0
    for txt in _iter_texts(dataset_name, split):
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        if eos is not None:
            ids.append(int(eos))
        n += 1
        if max_texts is not None and n >= max_texts:
            break
    return ids


@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    *,
    datasets: Sequence[str] = ("wikitext2",),
    model_seq_len: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    label: str = "PPL",
    max_texts: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate perplexity + report memory + throughput.

    This matches the SVD-LLM scripts' expectations (prints Weight/Peak memory),
    while also reporting wall-time and tokens/s so we can measure end-to-end speedup.
    """
    model.eval()
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    weight_mib = mib(measure_param_bytes(model))
    results: Dict[str, float] = {}

    t_global0 = time.perf_counter()
    for ds_name in datasets:
        token_ids = _token_stream(ds_name, tokenizer, split="test", max_texts=max_texts)
        if len(token_ids) < model_seq_len + 2:
            raise ValueError(f"Not enough tokens in {ds_name} test split for seqlen={model_seq_len}.")

        # Drop remainder to make full batches of fixed seq len.
        n_seq = (len(token_ids) - 1) // model_seq_len
        usable = n_seq * model_seq_len + 1
        token_ids = token_ids[:usable]

        flat = torch.tensor(token_ids, dtype=torch.long, device=device)
        x = flat[:-1].view(n_seq, model_seq_len)

        total_loss = 0.0
        total_tokens = 0

        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for i in range(0, n_seq, batch_size):
            xb = x[i : i + batch_size]
            attn = torch.ones_like(xb, dtype=torch.long, device=device)

            out = model(input_ids=xb, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B,S,V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = xb[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int(shift_labels.numel())

        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
        results[ds_name] = float(ppl)

        tok_per_s = (total_tokens / dt) if dt > 0 else float("inf")
        ms_per_batch = (dt * 1000.0) / max(1, math.ceil(n_seq / batch_size))
        print(f"[{ds_name}] tokens: {total_tokens} | time: {dt:.2f}s | {tok_per_s:,.0f} tok/s | {ms_per_batch:.2f} ms/batch")

    t_global = time.perf_counter() - t_global0

    print(f"{label}: {results}")
    if torch.cuda.is_available() and "cuda" in str(device):
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_rsrv = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        peak_alloc = peak_rsrv = 0.0

    print(f"Weight Memory: {weight_mib:.2f} MiB")
    print(f"Peak Memory (allocated): {peak_alloc:.2f} MiB")
    print(f"Peak Memory (reserved):  {peak_rsrv:.2f} MiB")
    print(f"Total eval time: {t_global:.2f}s")
    return results


@torch.no_grad()
def eff_eval(
    model,
    tokenizer,
    *,
    generated_len: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 50,
) -> None:
    """Simple throughput benchmark (no KV-cache)."""
    model.eval()
    vocab = int(getattr(model.config, "vocab_size", 32000))
    x = torch.randint(0, vocab, (batch_size, generated_len), device=device, dtype=torch.long)
    attn = torch.ones_like(x, dtype=torch.long, device=device)

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(int(warmup)):
        _ = model(input_ids=x, attention_mask=attn, use_cache=False)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(int(iters)):
        _ = model(input_ids=x, attention_mask=attn, use_cache=False)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens = int(batch_size) * int(generated_len) * int(iters)
    tok_s = tokens / dt if dt > 0 else float("inf")
    ms = (dt * 1000.0) / max(1, int(iters))

    if torch.cuda.is_available() and "cuda" in str(device):
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_rsrv = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        peak_alloc = peak_rsrv = 0.0

    print("=== Efficiency Eval (no KV cache) ===")
    print(f"mean {ms:.2f} ms/iter | {tok_s:,.0f} tok/s")
    print(f"Peak Memory (allocated): {peak_alloc:.2f} MiB")
    print(f"Peak Memory (reserved):  {peak_rsrv:.2f} MiB")


@torch.no_grad()
def decode_kvcache_eval(
    model,
    *,
    prompt_len: int = 2048,
    new_tokens: int = 128,
    batch_size: int = 1,
    device: str = "cuda",
    warmup: int = 5,
    max_cache_len: Optional[int] = None,
    lowrank_cache: bool = False,
    flashsvd_dense_cache: bool = False,
    baseline_dense_kvcache: bool = False,
    profile_decode: bool = False,
    profile_decode_steps: int = 20,
) -> dict[str, float | int | bool]:
    """Decode benchmark using KV-cache (per-token incremental forward).

    Runs:
      1) Prefill: forward(prompt_len) with cache writes enabled.
      2) Decode: repeat forward(1) for `new_tokens` steps using the cache.

    Notes:
    - Uses `StaticCache` to avoid dynamic `torch.cat` overhead.
    - Requires transformers>=4.4x Cache API (works with LLaMA in 4.53).
    """
    from transformers.cache_utils import StaticCache

    model.eval()
    if max_cache_len is None:
        extra = int(profile_decode_steps) if profile_decode else 0
        max_cache_len = int(prompt_len) + int(max(0, warmup)) + int(new_tokens) + extra + 1

    vocab = int(getattr(model.config, "vocab_size", 32000))
    dtype = next(iter(model.parameters())).dtype
    dev = torch.device(device)

    input_ids = torch.randint(0, vocab, (int(batch_size), int(prompt_len)), device=dev, dtype=torch.long)

    def _proj_out_features(proj) -> int:
        """Best-effort infer projection out_features across wrapper modules.

        We prefer `.out_features` when present (nn.Linear), and fall back to
        `.weight.shape[0]` for common wrappers / custom modules.
        """
        if proj is None:
            return 0
        try:
            out = getattr(proj, "out_features", None)
            if out is not None:
                return int(out)
        except Exception:
            pass
        try:
            w = getattr(proj, "weight", None)
            if w is not None and torch.is_tensor(w):
                return int(w.shape[0])
        except Exception:
            pass
        if torch.is_tensor(proj):
            try:
                return int(proj.shape[0])
            except Exception:
                return 0
        for child_name in ("base_layer", "linear", "proj", "wrapped", "module"):
            child = getattr(proj, child_name, None)
            if child is not None and child is not proj:
                r = _proj_out_features(child)
                if r:
                    return r
        return 0

    # If we enable the fine-grained FlashSVD attention decode profiler, avoid
    # perturbing the main timed decode benchmark by running it only during the
    # `profile_decode` phase (which is explicitly for profiling).
    attn_prof_prev = os.environ.get("FLASH_SVD_PROFILE_ATTN_DECODE", None)
    postpone_attn_prof = bool(profile_decode) and attn_prof_prev is not None and attn_prof_prev != "0"
    if postpone_attn_prof:
        os.environ.pop("FLASH_SVD_PROFILE_ATTN_DECODE", None)

    if lowrank_cache and flashsvd_dense_cache:
        raise ValueError("lowrank_cache and flashsvd_dense_cache are mutually exclusive.")
    if lowrank_cache and baseline_dense_kvcache:
        raise ValueError("lowrank_cache and baseline_dense_kvcache are mutually exclusive.")
    if flashsvd_dense_cache and baseline_dense_kvcache:
        raise ValueError("flashsvd_dense_cache and baseline_dense_kvcache are mutually exclusive.")

    if lowrank_cache:
        from flashsvd_component.lowrank_cache import LowRankKVCache

        # NOTE: Some FlashSVD checkpoints use different ranks for Q/K/V and/or
        # vary ranks by layer. Avoid inferring a single global rank here; the
        # cache lazily allocates per-layer buffers on the first `update()`.
        cache = LowRankKVCache(
            model.config,
            max_batch_size=int(batch_size),
            max_cache_len=int(max_cache_len),
            device=dev,
            dtype=dtype,
        )
    elif flashsvd_dense_cache or baseline_dense_kvcache:
        from flashsvd_component.dense_cache import FlashSVDDenseKVCache

        cache = FlashSVDDenseKVCache(
            model.config,
            max_batch_size=int(batch_size),
            max_cache_len=int(max_cache_len),
            device=dev,
            dtype=dtype,
        )
    else:
        cache = StaticCache(
            model.config,
            max_batch_size=int(batch_size),
            max_cache_len=int(max_cache_len),
            device=dev,
            dtype=dtype,
        )

    # Print key attention/cache metadata once (helps interpret decode results).
    try:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None and len(layers) > 0:
            attn0 = getattr(layers[0], "self_attn", None)
            if attn0 is not None:
                # Transformers may not expose `num_heads`/`num_key_value_heads` as attributes
                # on the attention module (newer versions rely on `attn.config` instead).
                cfg = getattr(attn0, "config", None)
                H = getattr(attn0, "num_heads", None)
                if H is None:
                    H = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
                if H is None:
                    H = getattr(model.config, "num_attention_heads", 0)
                H = int(H or 0)

                Hk = getattr(attn0, "num_key_value_heads", None)
                if Hk is None:
                    Hk = getattr(cfg, "num_key_value_heads", None) if cfg is not None else None
                if Hk is None:
                    Hk = getattr(model.config, "num_key_value_heads", H)
                Hk = int(Hk or H)
                Dh = int(getattr(attn0, "head_dim", 0) or 0)
                Rq = _proj_out_features(getattr(attn0, "q_v_proj", None))
                Rk = _proj_out_features(getattr(attn0, "k_v_proj", None))
                Rv = _proj_out_features(getattr(attn0, "v_v_proj", None))
                R = int(Rk or Rv or Rq)
                rep = int(H // Hk) if Hk else 0
                if lowrank_cache:
                    baseline_env = os.environ.get("FLASH_SVD_BASELINE_LR_KVCACHE", "0") != "0"
                    force_kernel_env = os.environ.get("FLASH_SVD_FORCE_ATTENTION_KERNEL", "0") != "0"
                    auto_mha_sdpa_env = os.environ.get("FLASH_SVD_AUTO_MHA_SDPA", "0") != "0"
                    auto_mha_stream_env = os.environ.get("FLASH_SVD_AUTO_MHA_STREAM", "0") != "0"
                    mha_stream_env = os.environ.get("FLASH_SVD_DECODE_MHA_STREAM", "0") != "0"
                    if baseline_env:
                        tag = "LowRankKVCacheBaseline"
                    else:
                        try:
                            thr_sdpa = int(os.environ.get("FLASH_SVD_MHA_SDPA_R_THRESHOLD", "512"))
                        except Exception:
                            thr_sdpa = 512
                        try:
                            thr_stream = int(os.environ.get("FLASH_SVD_MHA_STREAM_R_THRESHOLD", "768"))
                        except Exception:
                            thr_stream = 768
                        use_mha_stream = bool(
                            rep == 1
                            and H == Hk
                            and (
                                mha_stream_env
                                or (auto_mha_stream_env and R >= thr_stream)
                            )
                        )
                        if use_mha_stream:
                            tag = "LowRankKVCacheMHAStream"
                        elif auto_mha_sdpa_env and (not force_kernel_env) and rep == 1 and R >= thr_sdpa:
                            tag = "LowRankKVCacheAutoSDPA"
                        elif force_kernel_env:
                            tag = "LowRankKVCacheForcedKernel"
                        else:
                            tag = "LowRankKVCache"
                elif flashsvd_dense_cache:
                    tag = "FlashSVDDenseKVCache"
                elif baseline_dense_kvcache:
                    tag = "DenseKVCacheBaseline"
                else:
                    tag = "StaticCache"
                extra = " (REP=1, non-GQA)" if rep == 1 else ""
                rank_extra = ""
                if (Rq or Rk or Rv) and not (Rq == Rk == Rv):
                    rank_extra = f" (Rq={Rq}, Rk={Rk}, Rv={Rv})"
                print(f"[{tag}] H={H} Hk={Hk} REP={rep} Dh={Dh} R={R} max_cache_len={max_cache_len}{extra}{rank_extra}")
    except Exception:
        pass

    def _sync():
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()

    # Prefill
    cache_pos = torch.arange(int(prompt_len), device=dev, dtype=torch.long)
    _sync()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True, past_key_values=cache, cache_position=cache_pos)
    _sync()
    t_prefill = time.perf_counter() - t0

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    try:
        # Decode warmup
        for t in range(int(max(0, warmup))):
            cache_pos = torch.tensor([int(prompt_len) + t], device=dev, dtype=torch.long)
            out = model(input_ids=next_token, use_cache=True, past_key_values=cache, cache_position=cache_pos)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Timed decode
        _sync()
        t1 = time.perf_counter()
        for t in range(int(new_tokens)):
            cache_pos = torch.tensor([int(prompt_len) + int(warmup) + t], device=dev, dtype=torch.long)
            out = model(input_ids=next_token, use_cache=True, past_key_values=cache, cache_position=cache_pos)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync()
        t_decode = time.perf_counter() - t1
    finally:
        if postpone_attn_prof and attn_prof_prev is not None:
            os.environ["FLASH_SVD_PROFILE_ATTN_DECODE"] = attn_prof_prev

    prefill_tok = int(batch_size) * int(prompt_len)
    decode_tok = int(batch_size) * int(new_tokens)

    prefill_tok_s = prefill_tok / t_prefill if t_prefill > 0 else float("inf")
    decode_tok_s = decode_tok / t_decode if t_decode > 0 else float("inf")
    decode_ms = (t_decode * 1000.0) / max(1, int(new_tokens))

    print("=== KV-Cache Prefill/Decode Benchmark ===")
    print(f"Prefill: prompt_len={prompt_len} | time {t_prefill:.3f}s | {prefill_tok_s:,.0f} tok/s")
    print(f"Decode : new_tokens={new_tokens} | time {t_decode:.3f}s | {decode_ms:.3f} ms/token | {decode_tok_s:,.0f} tok/s")

    result = {
        "prompt_len": int(prompt_len),
        "new_tokens": int(new_tokens),
        "batch_size": int(batch_size),
        "prefill_time_s": float(t_prefill),
        "decode_time_s": float(t_decode),
        "prefill_tok_s": float(prefill_tok_s),
        "decode_tok_s": float(decode_tok_s),
        "decode_ms_per_token": float(decode_ms),
        "lowrank_cache": bool(lowrank_cache),
        "flashsvd_dense_cache": bool(flashsvd_dense_cache),
        "baseline_dense_kvcache": bool(baseline_dense_kvcache),
    }

    if not profile_decode:
        return result

    steps = int(max(0, profile_decode_steps))
    if steps <= 0:
        print("[profile_decode] skipped (profile_decode_steps<=0)")
        return
    if not (torch.cuda.is_available() and "cuda" in str(device)):
        print("[profile_decode] skipped (CUDA not available)")
        return

    # ---- Decode latency breakdown (module-level CUDA time) ----
    # Goal: identify whether decode is dominated by attention, MLP, layernorms, lm_head, etc.
    # We time selected modules using CUDA events via forward hooks.
    class _CudaEventTimer:
        def __init__(self, name: str):
            self.name = name
            self._stack = []
            self.pairs = []  # list[(start_event, end_event)]

        def pre(self, module, args):  # noqa: ARG002
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._stack.append((start, end))

        def post(self, module, args, output):  # noqa: ARG002
            if not self._stack:
                return
            start, end = self._stack.pop()
            end.record()
            self.pairs.append((start, end))

        def pre_kw(self, module, args, kwargs):  # noqa: ARG002
            return self.pre(module, args)

        def post_kw(self, module, args, kwargs, output):  # noqa: ARG002
            return self.post(module, args, output)

        def total_ms(self) -> float:
            total = 0.0
            for s, e in self.pairs:
                total += float(s.elapsed_time(e))
            return total

    def _add_timer(name: str, module, timers, handles):
        if module is None:
            return
        tmr = _CudaEventTimer(name)
        timers.append((name, tmr))
        # Be robust across PyTorch versions: prefer with_kwargs hooks when available.
        try:
            handles.append(module.register_forward_pre_hook(tmr.pre_kw, with_kwargs=True))
            handles.append(module.register_forward_hook(tmr.post_kw, with_kwargs=True))
        except TypeError:
            handles.append(module.register_forward_pre_hook(tmr.pre))
            handles.append(module.register_forward_hook(tmr.post))

    # Collect modules to time (best-effort across llama/opt-like models).
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)

    timers = []
    handles = []
    try:
        # Embedding + final norm + lm_head
        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed is None:
            embed = getattr(getattr(getattr(model, "model", None), "decoder", None), "embed_tokens", None)
        _add_timer("embed_tokens", embed, timers, handles)

        final_norm = getattr(getattr(model, "model", None), "norm", None)
        if final_norm is None:
            final_norm = getattr(getattr(getattr(model, "model", None), "decoder", None), "final_layer_norm", None)
        _add_timer("final_norm", final_norm, timers, handles)

        _add_timer("lm_head", getattr(model, "lm_head", None), timers, handles)

        # Per-layer modules
        if layers is not None:
            for i, layer in enumerate(layers):
                _add_timer(f"layer{i}.ln1", getattr(layer, "input_layernorm", None), timers, handles)
                _add_timer(f"layer{i}.attn", getattr(layer, "self_attn", None), timers, handles)
                _add_timer(f"layer{i}.ln2", getattr(layer, "post_attention_layernorm", None), timers, handles)
                _add_timer(f"layer{i}.mlp", getattr(layer, "mlp", None), timers, handles)

        # Total model forward timer (per-step)
        total_pairs = []
        start_pos = int(prompt_len) + int(max(0, warmup)) + int(new_tokens)
        for t in range(steps):
            cache_pos = torch.tensor([start_pos + t], device=dev, dtype=torch.long)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = model(input_ids=next_token, use_cache=True, past_key_values=cache, cache_position=cache_pos)
            ev1.record()
            total_pairs.append((ev0, ev1))
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        _sync()

        total_ms = sum(float(s.elapsed_time(e)) for s, e in total_pairs)
        total_ms_per_tok = total_ms / float(steps)

        # Aggregate by category
        per_mod = {name: tmr.total_ms() / float(steps) for name, tmr in timers}

        def _sum_suffix(suffix: str) -> float:
            return sum(v for k, v in per_mod.items() if k.endswith(suffix))

        breakdown = {
            "attn_total": _sum_suffix(".attn"),
            "mlp_total": _sum_suffix(".mlp"),
            "ln1_total": _sum_suffix(".ln1"),
            "ln2_total": _sum_suffix(".ln2"),
            "embed_tokens": per_mod.get("embed_tokens", 0.0),
            "final_norm": per_mod.get("final_norm", 0.0),
            "lm_head": per_mod.get("lm_head", 0.0),
        }
        known = sum(breakdown.values())
        other = max(0.0, float(total_ms_per_tok) - float(known))

        print("\n=== Decode Profile (module CUDA time) ===")
        print(f"Avg over {steps} steps @ seq≈{start_pos} | total_forward {total_ms_per_tok:.3f} ms/token")
        for k, v in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True):
            pct = (100.0 * v / total_ms_per_tok) if total_ms_per_tok > 0 else 0.0
            print(f"  {k:12s}: {v:8.3f} ms ({pct:5.1f}%)")
        pct_other = (100.0 * other / total_ms_per_tok) if total_ms_per_tok > 0 else 0.0
        print(f"  {'other':12s}: {other:8.3f} ms ({pct_other:5.1f}%)")

        # Show top individual modules for quick pinpointing.
        topk = 10
        top = sorted(((k, v) for k, v in per_mod.items() if v > 0), key=lambda kv: kv[1], reverse=True)[:topk]
        if top:
            print(f"Top {len(top)}/{topk} modules:")
            for name, ms in top:
                pct = (100.0 * ms / total_ms_per_tok) if total_ms_per_tok > 0 else 0.0
                print(f"  {name:16s}: {ms:8.3f} ms ({pct:5.1f}%)")
        print("")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
    return result
