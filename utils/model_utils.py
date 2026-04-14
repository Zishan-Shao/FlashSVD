from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def find_layers(
    module: nn.Module,
    layers: Tuple[Type[nn.Module], ...] = (nn.Linear,),
    *,
    prefix: str = "",
) -> Dict[str, nn.Module]:
    """Recursively collect layers of given types.

    The returned keys match `module.named_modules()` names, which is what the
    SVD-LLM scripts expect (e.g. "self_attn.q_proj", "mlp.gate_proj").
    """
    found: Dict[str, nn.Module] = {}
    for name, child in module.named_modules():
        if name == "":
            continue
        if isinstance(child, layers):
            found[prefix + name] = child
    return found


def _token_arg(hf_token: Optional[str]):
    # transformers>=4.35 uses `token=`; older versions use `use_auth_token=`.
    return {"token": hf_token} if hf_token else {}


def _ensure_writable_hf_cache() -> None:
    """Point HF/transformers caches at a writable workspace path when unset."""
    root = Path(__file__).resolve().parents[2]
    hf_home = Path(os.environ.get("HF_HOME", root / ".cache" / "huggingface"))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    modules_cache = Path(os.environ.get("HF_MODULES_CACHE", hf_home / "modules"))
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", hf_home / "transformers"))

    for path in (hf_home, hub_cache, modules_cache, transformers_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hf_export_cache_root() -> Path:
    root = Path(os.environ.get("FLASHSVD_HF_EXPORT_CACHE", _repo_root() / "checkpoints" / "hf_exports"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _looks_like_explicit_local_path(raw: str) -> bool:
    value = str(raw).strip()
    if not value:
        return False
    if value.startswith(("/", "./", "../", "~/")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    return False


def _strip_hf_source_prefix(raw: str) -> str:
    value = str(raw).strip()
    for prefix in ("hf://", "huggingface://"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _parse_hf_repo_subfolder_source(source: str) -> Optional[tuple[str, str]]:
    raw = _strip_hf_source_prefix(source)
    if not raw or raw.startswith(("http://", "https://")):
        return None
    if _looks_like_explicit_local_path(raw):
        return None

    if "::" in raw:
        repo_id, subfolder = raw.split("::", 1)
        repo_id = repo_id.strip().strip("/")
        subfolder = subfolder.strip().strip("/")
        if repo_id and subfolder:
            return repo_id, subfolder
        return None

    parts = [part for part in raw.split("/") if part]
    if len(parts) < 3:
        return None
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return repo_id, subfolder


def _download_hf_repo_subfolder(repo_id: str, subfolder: str, *, hf_token: Optional[str]) -> Path:
    _ensure_writable_hf_cache()
    from huggingface_hub import snapshot_download

    repo_id = str(repo_id).strip().strip("/")
    subfolder = str(subfolder).strip().strip("/")
    if not repo_id or not subfolder:
        raise ValueError(f"Invalid HF repo/subfolder source: repo_id={repo_id!r}, subfolder={subfolder!r}")

    repo_root = _hf_export_cache_root() / Path(repo_id)
    local_path = repo_root / Path(subfolder)

    def _has_model_weights(path: Path) -> bool:
        candidates = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        return any((path / name).is_file() for name in candidates) or any(path.glob("*.safetensors")) or any(
            path.glob("pytorch_model-*.bin")
        )

    def _looks_complete(path: Path) -> bool:
        if not path.is_dir():
            return False
        if not (path / "config.json").is_file():
            return False
        if not _has_model_weights(path):
            return False
        return True

    if _looks_complete(local_path):
        print("[FlashSVD] Reusing cached HF export from local cache.")
        return local_path

    print("[FlashSVD] Downloading HF export into local cache.")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{subfolder}/**", f"{subfolder}/*"],
        local_dir=str(repo_root),
        **_token_arg(hf_token),
    )
    if not _looks_complete(local_path):
        raise FileNotFoundError(
            "Downloaded a Hugging Face export but could not find the requested subfolder "
            f"under the local export cache: {repo_root}"
        )
    return local_path


def _read_local_config(path: Path) -> Dict[str, object]:
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _prepare_local_dynamic_module_dir(path: Path, cfg: Dict[str, object]) -> Path:
    """Create a sanitized symlink path for local dirs with custom HF code."""
    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict) or not auto_map:
        return path

    sanitized_name = re.sub(r"[^A-Za-z0-9_-]", "_", path.name)
    if sanitized_name == path.name:
        return path

    alias_root = path.parent / ".flashsvd_model_aliases"
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_path = alias_root / sanitized_name
    if alias_path.exists() or alias_path.is_symlink():
        try:
            if alias_path.resolve() == path.resolve():
                return alias_path
        except Exception:
            pass
        raise RuntimeError(
            f"Refusing to reuse local model alias path with unexpected target: {alias_path}"
        )

    os.symlink(path, alias_path, target_is_directory=True)
    return alias_path


def _is_basis_sharing_config(cfg: Dict[str, object]) -> bool:
    archs = cfg.get("architectures")
    if not isinstance(archs, list):
        return False
    return any("sharellama" in str(arch).lower() for arch in archs)


def _torch_dtype_from_raw(value: object) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    raw = str(value).strip().lower()
    mapping = {
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(raw)


def _replace_module_weight(module: nn.Module, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor weight, got: {type(tensor)!r}")
    module.weight = nn.Parameter(tensor)
    if isinstance(module, nn.Linear):
        _sync_linear_meta(module)
    elif isinstance(module, nn.Embedding):
        module.num_embeddings = int(tensor.shape[0])
        module.embedding_dim = int(tensor.shape[1])


def _iter_pytorch_bin_shards(path: Path) -> Iterable[Path]:
    index_path = path / "pytorch_model.bin.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid HF shard index at: {index_path}")
        shard_names = list(dict.fromkeys(str(name) for name in weight_map.values()))
        for shard_name in shard_names:
            shard_path = path / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard referenced by index: {shard_path}")
            yield shard_path
        return

    single_path = path / "pytorch_model.bin"
    if single_path.is_file():
        yield single_path
        return

    shard_paths = sorted(path.glob("pytorch_model-*.bin"))
    if shard_paths:
        for shard_path in shard_paths:
            yield shard_path
        return

    raise FileNotFoundError(f"Could not find HF PyTorch checkpoint shards under: {path}")


def _expected_basis_sharing_keys(num_hidden_layers: int) -> set[str]:
    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer_idx in range(int(num_hidden_layers)):
        prefix = f"model.layers.{layer_idx}"
        keys.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_basis.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_basis.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_basis.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.o_basis.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.gate_basis.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.up_basis.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp.down_basis.weight",
            }
        )
    return keys


def _assign_basis_sharing_tensor(model: nn.Module, key: str, tensor: torch.Tensor) -> bool:
    if key == "model.embed_tokens.weight":
        _replace_module_weight(model.model.embed_tokens, tensor)
        return True
    if key == "model.norm.weight":
        _replace_module_weight(model.model.norm, tensor)
        return True
    if key == "lm_head.weight":
        _replace_module_weight(model.lm_head, tensor)
        return True

    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return False

    layer_idx = int(parts[2])
    layer = model.model.layers[layer_idx]
    block = parts[3]

    if block == "input_layernorm" and parts[4] == "weight":
        _replace_module_weight(layer.input_layernorm, tensor)
        return True
    if block == "post_attention_layernorm" and parts[4] == "weight":
        _replace_module_weight(layer.post_attention_layernorm, tensor)
        return True

    if len(parts) != 6 or parts[5] != "weight":
        return False

    if block == "self_attn":
        attn_map = {
            "q_proj": "q_u_proj",
            "q_basis": "q_v_proj",
            "k_proj": "k_u_proj",
            "k_basis": "k_v_proj",
            "v_proj": "v_u_proj",
            "v_basis": "v_v_proj",
            "o_proj": "o_u_proj",
            "o_basis": "o_v_proj",
        }
        dst_name = attn_map.get(parts[4])
        if dst_name is None:
            return False
        _replace_module_weight(getattr(layer.self_attn, dst_name), tensor)
        return True

    if block == "mlp":
        mlp_map = {
            "gate_proj": "gate_u_proj",
            "gate_basis": "gate_v_proj",
            "up_proj": "up_u_proj",
            "up_basis": "up_v_proj",
            "down_proj": "down_u_proj",
            "down_basis": "down_v_proj",
        }
        dst_name = mlp_map.get(parts[4])
        if dst_name is None:
            return False
        _replace_module_weight(getattr(layer.mlp, dst_name), tensor)
        return True

    return False


def _tie_basis_sharing_layer_groups(model: nn.Module, cfg: Dict[str, object]) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    group_specs = {
        "q_groups": lambda layer: layer.self_attn.q_v_proj,
        "k_groups": lambda layer: layer.self_attn.k_v_proj,
        "v_groups": lambda layer: layer.self_attn.v_v_proj,
        "o_groups": lambda layer: layer.self_attn.o_v_proj,
        "gate_groups": lambda layer: layer.mlp.gate_v_proj,
        "up_groups": lambda layer: layer.mlp.up_v_proj,
        "down_groups": lambda layer: layer.mlp.down_v_proj,
    }

    for group_name, getter in group_specs.items():
        raw_groups = cfg.get(group_name)
        if not isinstance(raw_groups, list):
            continue
        for group_idx, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, list) or len(raw_group) <= 1:
                continue
            group = [int(layer_idx) for layer_idx in raw_group]
            ref_linear = getter(layers[group[0]])
            ref_weight = ref_linear.weight
            for layer_idx in group[1:]:
                cur_linear = getter(layers[layer_idx])
                cur_weight = cur_linear.weight
                if tuple(cur_weight.shape) != tuple(ref_weight.shape):
                    raise ValueError(
                        f"Basis_Sharing group {group_name}[{group_idx}] has mismatched shapes: "
                        f"{tuple(ref_weight.shape)} vs {tuple(cur_weight.shape)}"
                    )
                if not torch.equal(cur_weight, ref_weight):
                    max_diff = float((cur_weight.float() - ref_weight.float()).abs().max().item())
                    raise ValueError(
                        f"Basis_Sharing group {group_name}[{group_idx}] is not actually shared "
                        f"(layers {group[0]} and {layer_idx}, max diff {max_diff:.6f})."
                    )
                cur_linear.weight = ref_weight
                _sync_linear_meta(cur_linear)


def _finalize_basis_sharing_flashsvd_model(model: nn.Module) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    from models.llama import LlamaRotaryEmbedding
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    model.model.rotary_emb = HFLlamaRotaryEmbedding(config=model.config)

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        q_rank = int(attn.q_v_proj.weight.shape[0])
        k_rank = int(attn.k_v_proj.weight.shape[0])
        v_rank = int(attn.v_v_proj.weight.shape[0])
        if len({q_rank, k_rank, v_rank}) != 1:
            raise ValueError(
                f"Basis_Sharing attention layer {layer_idx} must have shared q/k/v rank; "
                f"got {q_rank}/{k_rank}/{v_rank}."
            )

        attn.layer_idx = int(layer_idx)
        attn.low_rank = int(q_rank)
        attn.ratio = float(2.0 * q_rank / max(1, attn.hidden_size))
        attn.use_lowrank_cache = True
        attn.num_key_value_heads = int(attn.k_u_proj.weight.shape[0] // max(1, attn.head_dim))
        attn.num_key_value_groups = int(attn.num_heads // max(1, attn.num_key_value_heads))
        setattr(attn.config, "flash_svd_lowrank_rank", int(q_rank))
        setattr(attn.config, "flash_svd_use_lowrank_cache", True)
        rope_theta = float(getattr(attn.config, "rope_theta", 10000.0))
        attn.rotary_emb = LlamaRotaryEmbedding(
            attn.head_dim,
            max_position_embeddings=attn.max_position_embeddings,
            base=rope_theta,
        )
        attn._decode_Vq = None
        attn._decode_Vk = None
        attn._decode_Vv = None
        attn._decode_ptrs = None
        attn._dense_decode_prepacked_cache = {}

        gate_rank = int(mlp.gate_v_proj.weight.shape[0])
        up_rank = int(mlp.up_v_proj.weight.shape[0])
        if gate_rank != up_rank:
            raise ValueError(
                f"Basis_Sharing MLP layer {layer_idx} must have shared gate/up rank; "
                f"got {gate_rank}/{up_rank}."
            )
        hidden_size = int(mlp.up_v_proj.weight.shape[1])
        intermediate_size = int(mlp.gate_u_proj.weight.shape[0])
        mlp.ratio = float(
            gate_rank * (float(hidden_size) + float(intermediate_size))
            / max(1.0, float(hidden_size) * float(intermediate_size))
        )


def _load_basis_sharing_model_from_local_dir(path: Path, cfg: Dict[str, object]):
    _ensure_writable_hf_cache()
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.llama import SVD_LlamaAttention, SVD_LlamaMLP

    tok = _load_tokenizer(str(path), hf_token=None)

    config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    if hasattr(config, "architectures"):
        try:
            config.architectures = ["LlamaForCausalLM"]
        except Exception:
            pass
    torch_dtype = _torch_dtype_from_raw(
        getattr(config, "torch_dtype", None) or cfg.get("torch_dtype") or cfg.get("dtype")
    )
    if torch_dtype is not None:
        setattr(config, "torch_dtype", torch_dtype)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        for layer in model.model.layers:
            layer.self_attn = SVD_LlamaAttention(config=model.config, ratio=0.5)
            layer.mlp = SVD_LlamaMLP(
                hidden_size=int(model.config.hidden_size),
                intermediate_size=int(model.config.intermediate_size),
                hidden_act=str(model.config.hidden_act),
                ratio=0.5,
            )

    expected_keys = _expected_basis_sharing_keys(int(getattr(model.config, "num_hidden_layers", 0)))
    seen_keys: set[str] = set()
    for shard_path in _iter_pytorch_bin_shards(path):
        shard = torch.load(str(shard_path), map_location="cpu")
        if not isinstance(shard, dict):
            raise TypeError(f"Unexpected shard type for {shard_path}: {type(shard)!r}")
        for key, tensor in shard.items():
            if _assign_basis_sharing_tensor(model, str(key), tensor):
                seen_keys.add(str(key))
        del shard

    missing_keys = sorted(expected_keys - seen_keys)
    if missing_keys:
        sample = ", ".join(missing_keys[:8])
        raise KeyError(
            f"Basis_Sharing load is missing {len(missing_keys)} required tensors from {path}. "
            f"First missing keys: {sample}"
        )

    _tie_basis_sharing_layer_groups(model, cfg)
    _finalize_basis_sharing_flashsvd_model(model)
    ensure_transformers_config_compat(model)
    ensure_transformers_layer_idx(model)
    _maybe_enable_flashsvd_llama_runtime_patches(model)
    model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
    return model, tok


def _is_uv_factor_linear(module: object) -> bool:
    if module is None:
        return False
    u_proj = getattr(module, "u_proj", None)
    v_proj = getattr(module, "v_proj", None)
    return isinstance(u_proj, nn.Linear) and isinstance(v_proj, nn.Linear)


def _is_dobi_factor_linear(module: object) -> bool:
    if module is None:
        return False
    a_linear = getattr(module, "ALinear", None)
    b_linear = getattr(module, "BLinear", None)
    return isinstance(a_linear, nn.Linear) and isinstance(b_linear, nn.Linear)


def _sync_linear_meta(linear: nn.Linear) -> None:
    if not isinstance(linear, nn.Linear):
        return
    linear.in_features = int(linear.weight.shape[1])
    linear.out_features = int(linear.weight.shape[0])
    bias = getattr(linear, "bias", None)
    if bias is not None and bias.numel() != linear.out_features:
        new_bias = bias.new_zeros(linear.out_features)
        keep = min(int(bias.numel()), int(linear.out_features))
        new_bias[:keep] = bias.data[:keep]
        linear.bias = nn.Parameter(new_bias, requires_grad=bias.requires_grad)


def _copy_uv_factor_pair(dst_u: nn.Linear, dst_v: nn.Linear, src) -> int:
    src_u = getattr(src, "u_proj", None)
    src_v = getattr(src, "v_proj", None)
    if not isinstance(src_u, nn.Linear) or not isinstance(src_v, nn.Linear):
        src_u = getattr(src, "BLinear", None)
        src_v = getattr(src, "ALinear", None)
    if not isinstance(src_u, nn.Linear) or not isinstance(src_v, nn.Linear):
        raise TypeError(
            "Expected a source module with nn.Linear u_proj/v_proj or BLinear/ALinear factors."
        )

    dst_u.weight.data = src_u.weight.data
    dst_v.weight.data = src_v.weight.data
    if getattr(src_u, "bias", None) is not None and getattr(dst_u, "bias", None) is not None:
        dst_u.bias.data = src_u.bias.data
    if getattr(src_v, "bias", None) is not None and getattr(dst_v, "bias", None) is not None:
        dst_v.bias.data = src_v.bias.data
    _sync_linear_meta(dst_u)
    _sync_linear_meta(dst_v)
    return int(dst_v.out_features)


def _module_rank(module) -> int:
    v_proj = getattr(module, "v_proj", None)
    if isinstance(v_proj, nn.Linear):
        return int(v_proj.out_features)
    a_linear = getattr(module, "ALinear", None)
    if isinstance(a_linear, nn.Linear):
        return int(a_linear.out_features)
    return 0


def _maybe_convert_uv_wrapped_llama_to_flashsvd(model: nn.Module) -> nn.Module:
    """Convert wrapped factorized LLaMA blocks into native FlashSVD modules."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return model

    try:
        from models.llama import SVD_LlamaAttention, SVD_LlamaMLP
    except Exception:
        return model

    converted = 0
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attn is None or mlp is None:
            continue
        if hasattr(attn, "q_u_proj") and hasattr(mlp, "gate_u_proj"):
            continue

        q_proj = getattr(attn, "q_proj", None)
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if not all(
            (_is_uv_factor_linear(mod) or _is_dobi_factor_linear(mod))
            for mod in (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        ):
            continue

        q_rank = _module_rank(q_proj)
        k_rank = _module_rank(k_proj)
        v_rank = _module_rank(v_proj)
        o_rank = _module_rank(o_proj)
        gate_rank = _module_rank(gate_proj)
        up_rank = _module_rank(up_proj)
        down_rank = _module_rank(down_proj)
        if min(q_rank, k_rank, v_rank, o_rank, gate_rank, up_rank, down_rank) <= 0:
            raise ValueError(f"FlashSVD native conversion found an invalid low-rank projection in layer {layer_idx}.")

        svd_attn = SVD_LlamaAttention(config=model.config, ratio=0.5)
        svd_mlp = SVD_LlamaMLP(
            hidden_size=int(layer.hidden_size),
            intermediate_size=int(model.config.intermediate_size),
            hidden_act=str(model.config.hidden_act),
            ratio=0.5,
        )

        _copy_uv_factor_pair(svd_attn.q_u_proj, svd_attn.q_v_proj, q_proj)
        _copy_uv_factor_pair(svd_attn.k_u_proj, svd_attn.k_v_proj, k_proj)
        _copy_uv_factor_pair(svd_attn.v_u_proj, svd_attn.v_v_proj, v_proj)
        _copy_uv_factor_pair(svd_attn.o_u_proj, svd_attn.o_v_proj, o_proj)
        _copy_uv_factor_pair(svd_mlp.gate_u_proj, svd_mlp.gate_v_proj, gate_proj)
        _copy_uv_factor_pair(svd_mlp.up_u_proj, svd_mlp.up_v_proj, up_proj)
        _copy_uv_factor_pair(svd_mlp.down_u_proj, svd_mlp.down_v_proj, down_proj)

        svd_attn.layer_idx = int(getattr(attn, "layer_idx", layer_idx))
        svd_attn.low_rank = int(q_rank)
        svd_attn.ratio = float(2.0 * q_rank / max(1, svd_attn.hidden_size))
        svd_attn.use_lowrank_cache = True
        setattr(svd_attn.config, "flash_svd_lowrank_rank", int(q_rank))
        setattr(svd_attn.config, "flash_svd_use_lowrank_cache", True)
        svd_mlp.ratio = float(
            0.5 * (gate_rank + up_rank)
            * (float(layer.hidden_size) + float(model.config.intermediate_size))
            / max(1.0, float(layer.hidden_size) * float(model.config.intermediate_size))
        )

        layer.self_attn = svd_attn
        layer.mlp = svd_mlp
        converted += 1

    if converted:
        try:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

            llama_model = getattr(model, "model", None)
            if llama_model is not None and not hasattr(llama_model, "rotary_emb"):
                llama_model.rotary_emb = HFLlamaRotaryEmbedding(config=model.config)
        except Exception:
            pass
        ensure_transformers_layer_idx(model)
    return model


def _is_tokenizer_like(obj) -> bool:
    if obj is None or isinstance(obj, bool):
        return False
    # HF tokenizers are callable and implement encode/decode.
    if callable(obj):
        return True
    return hasattr(obj, "encode") and hasattr(obj, "decode")


def _load_tokenizer(model_id: str, *, hf_token: Optional[str] = None):
    _ensure_writable_hf_cache()
    from transformers import AutoTokenizer

    last_err: Optional[Exception] = None
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=use_fast,
                **_token_arg(hf_token),
            )
        except Exception as e:
            last_err = e
            continue

        if not _is_tokenizer_like(tok):
            # Extremely defensive: we've observed environments where a non-tokenizer
            # placeholder (e.g. bool) can surface here.
            continue

        try:
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
        except Exception:
            pass
        return tok

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to load a usable tokenizer for: {model_id}")


def ensure_transformers_layer_idx(model: nn.Module) -> None:
    """Ensure decoder self-attention modules have `layer_idx`.

    Transformers>=4.4x Cache implementations (DynamicCache/StaticCache/...) use
    `layer_idx` to route KV updates. Some pickled checkpoints (e.g. after module
    replacement) may miss this attribute.
    """
    try:
        # LLaMA / Mistral style
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    # Force-correct even if present: some replacement pipelines set all to 0.
                    setattr(attn, "layer_idx", int(i))
            return
        # OPT style
        dec_layers = getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)
        if dec_layers is not None:
            for i, layer in enumerate(dec_layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    setattr(attn, "layer_idx", int(i))
    except Exception:
        # Best-effort only.
        return


def ensure_transformers_config_compat(model: nn.Module) -> None:
    """Populate config internals expected by newer transformers releases."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return

    raw = getattr(cfg, "__dict__", {})
    defaults = {
        "_output_attentions": bool(raw.get("output_attentions", False)),
        "_output_hidden_states": bool(raw.get("output_hidden_states", False)),
        "_return_dict": bool(raw.get("return_dict", True)),
        "_torchscript": bool(raw.get("torchscript", False)),
    }
    for name, value in defaults.items():
        if not hasattr(cfg, name):
            setattr(cfg, name, value)

    if not hasattr(cfg, "_attn_implementation"):
        attn_impl = raw.get("attn_implementation", raw.get("_attn_implementation_internal", "eager"))
        setattr(cfg, "_attn_implementation", str(attn_impl))


def _maybe_enable_flashsvd_llama_runtime_patches(model: nn.Module) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    try:
        from models.llama import (
            SVD_LlamaAttention,
            SVD_LlamaMLP,
            enable_flashsvd_llama_layer_tail_cuda_graph,
        )
    except Exception:
        return

    for layer in layers:
        if isinstance(getattr(layer, "self_attn", None), SVD_LlamaAttention) or isinstance(
            getattr(layer, "mlp", None), SVD_LlamaMLP
        ):
            enable_flashsvd_llama_layer_tail_cuda_graph()
            return


def get_model_from_huggingface(
    model_id: str,
    *,
    hf_token: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load (model, tokenizer) from HuggingFace hub.

    Note: we keep the model on CPU; callers can move/cast as needed.
    """
    expanded_local = Path(str(model_id)).expanduser()
    if expanded_local.exists():
        return get_model_from_local(str(expanded_local))

    if _looks_like_explicit_local_path(str(model_id)):
        raise FileNotFoundError(
            f"Local checkpoint path does not exist: {expanded_local}. "
            "If you meant a Hugging Face repo subfolder, use "
            "'namespace/repo/path/to/export' or 'namespace/repo::path/to/export'."
        )

    repo_subfolder = _parse_hf_repo_subfolder_source(model_id)
    if repo_subfolder is not None:
        repo_id, subfolder = repo_subfolder
        local_dir = _download_hf_repo_subfolder(repo_id, subfolder, hf_token=hf_token)
        return get_model_from_local(str(local_dir))

    _ensure_writable_hf_cache()
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer(model_id, hf_token=hf_token)

    # Avoid HF warnings about dtype; let caller cast later if desired.
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, **_token_arg(hf_token))
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = _maybe_convert_uv_wrapped_llama_to_flashsvd(model)
    ensure_transformers_config_compat(model)
    ensure_transformers_layer_idx(model)
    _maybe_enable_flashsvd_llama_runtime_patches(model)

    # Common in quant/ptq pipelines.
    model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
    return model, tok


def get_model_from_source(
    source: str,
    *,
    hf_token: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    raw = str(source).strip()
    local_path = Path(raw).expanduser()
    if local_path.exists():
        return get_model_from_local(str(local_path))
    return get_model_from_huggingface(raw, hf_token=hf_token, torch_dtype=torch_dtype)


def get_model_from_local(path: str):
    """Load (model, tokenizer) from a local torch.save checkpoint.

    Expected format: torch.save({'model': model, 'tokenizer': tokenizer}, path)
    """
    p = Path(path)
    if p.is_dir():
        cfg = _read_local_config(p)
        load_path = _prepare_local_dynamic_module_dir(p, cfg)
        if _is_basis_sharing_config(cfg):
            return _load_basis_sharing_model_from_local_dir(load_path, cfg)
        _ensure_writable_hf_cache()
        from transformers import AutoModelForCausalLM

        tok = _load_tokenizer(str(load_path), hf_token=None)
        model = AutoModelForCausalLM.from_pretrained(
            str(load_path), trust_remote_code=True, low_cpu_mem_usage=True
        )
        model = _maybe_convert_uv_wrapped_llama_to_flashsvd(model)
        ensure_transformers_config_compat(model)
        ensure_transformers_layer_idx(model)
        _maybe_enable_flashsvd_llama_runtime_patches(model)
        model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
        return model, tok

    obj = _torch_load_local_checkpoint(p)
    if isinstance(obj, dict) and "model" in obj and "tokenizer" in obj:
        model = _maybe_convert_uv_wrapped_llama_to_flashsvd(obj["model"])
        ensure_transformers_config_compat(model)
        ensure_transformers_layer_idx(model)
        _maybe_enable_flashsvd_llama_runtime_patches(model)
        model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
        return model, obj["tokenizer"]
    if hasattr(obj, "forward"):
        # A pickled HF model (rare but possible): try to recover tokenizer.
        model = _maybe_convert_uv_wrapped_llama_to_flashsvd(obj)
        ensure_transformers_config_compat(model)
        ensure_transformers_layer_idx(model)
        _maybe_enable_flashsvd_llama_runtime_patches(model)
        model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
        model_id = getattr(model, "name_or_path", None)
        if model_id:
            try:
                tok = _load_tokenizer(str(model_id), hf_token=None)
                return model, tok
            except Exception:
                pass
        raise ValueError(
            f"Loaded a model object from {path} but could not recover a tokenizer. "
            "Please re-save as {'model': model, 'tokenizer': tok}."
        )
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


@torch.no_grad()
def measure_param_bytes(model: nn.Module) -> int:
    """Sum unique parameter storage bytes (avoids double-count on tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p is None:
            continue
        try:
            st = p.untyped_storage()
            key = (int(st.data_ptr()), int(st.nbytes()))
            nbytes = int(st.nbytes())
        except Exception:
            st = p.storage()
            nbytes = int(st.size()) * int(p.element_size())
            key = (int(st.data_ptr()), int(nbytes))
        if key in seen:
            continue
        seen.add(key)
        total += nbytes
    return total


def mib(nbytes: int) -> float:
    return float(nbytes) / (1024.0**2)


def set_env_flag(name: str, value: bool):
    if value:
        os.environ[name] = "1"
    else:
        # Don't leave stale "0" around; scripts typically check != "0".
        os.environ.pop(name, None)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_flashsvd_checkpoint(path: Path) -> bool:
    try:
        flashsvd_root = Path(__file__).resolve().parents[1]  # FlashSVD-v1.5/
        repo_root = Path(__file__).resolve().parents[2]  # FlashSVD/
        resolved = path.resolve()
        if not resolved.is_relative_to(flashsvd_root) and not resolved.is_relative_to(repo_root):
            return False
    except Exception:
        # Best-effort: if we can't resolve, fall back to name heuristics only.
        pass
    name = path.name.lower()
    # Common outputs produced by SVDLLM(.py) / SVDLLM_flashsvd(.py)
    return any(
        key in name
        for key in (
            "_whitening_only_",
            "_whitening_then_update_",
            "_update_only_",
            "_profiling_",
            "svdllm",
            "flashsvd",
            "merge.pt",
        )
    )


def _looks_like_dobi_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    return "dobi" in name or "dobisvd" in name


@contextmanager
def _pickle_compat_shims() -> Iterator[None]:
    added_modules: list[str] = []
    prev_attrs: list[tuple[object, str, object]] = []
    try:
        try:
            import transformers.models.llama.modeling_llama as llama_modeling

            if not hasattr(llama_modeling, "LlamaSdpaAttention"):
                setattr(llama_modeling, "LlamaSdpaAttention", llama_modeling.LlamaAttention)
                prev_attrs.append((llama_modeling, "LlamaSdpaAttention", None))
        except Exception:
            pass

        try:
            import transformers.activations as activations_mod

            if not hasattr(activations_mod, "SiLUActivation"):
                class SiLUActivation(nn.Module):
                    def forward(self, x):
                        return F.silu(x)

                setattr(activations_mod, "SiLUActivation", SiLUActivation)
                prev_attrs.append((activations_mod, "SiLUActivation", None))
        except Exception:
            pass

        pkg = sys.modules.get("modules")
        if pkg is None:
            pkg = types.ModuleType("modules")
            sys.modules["modules"] = pkg
            added_modules.append("modules")

        mod = sys.modules.get("modules.module")
        if mod is None:
            mod = types.ModuleType("modules.module")
            sys.modules["modules.module"] = mod
            added_modules.append("modules.module")

        if not hasattr(mod, "SVDTransformLayer"):
            class SVDTransformLayer(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            setattr(mod, "SVDTransformLayer", SVDTransformLayer)
            prev_attrs.append((mod, "SVDTransformLayer", None))

        if getattr(pkg, "module", None) is not mod:
            old = getattr(pkg, "module", None)
            setattr(pkg, "module", mod)
            prev_attrs.append((pkg, "module", old))

        yield
    finally:
        for owner, attr, old in reversed(prev_attrs):
            if old is None:
                try:
                    delattr(owner, attr)
                except Exception:
                    pass
            else:
                try:
                    setattr(owner, attr, old)
                except Exception:
                    pass
        for name in reversed(added_modules):
            sys.modules.pop(name, None)


def _torch_load_local_checkpoint(path: Path):
    """Load a local torch checkpoint, handling PyTorch>=2.6 weights_only default.

    PyTorch 2.6 changed `torch.load` default `weights_only=True`, which rejects
    pickled model/tokenizer objects (our SVDLLM scripts save them as a dict).
    We try a safe load first and fall back to `weights_only=False` only when we
    can reasonably assume the checkpoint is trusted.
    """

    try:
        from quant.awq_compat import ensure_autoawq_compatibility

        ensure_autoawq_compatibility()
    except Exception:
        pass

    def _load(*, weights_only: Optional[bool]):
        if weights_only is None:
            return torch.load(str(path), map_location="cpu")
        try:
            return torch.load(str(path), map_location="cpu", weights_only=weights_only)
        except TypeError:
            # Older PyTorch without weights_only.
            return torch.load(str(path), map_location="cpu")

    try:
        return _load(weights_only=True)
    except Exception as e:
        msg = str(e)
        is_weights_only = "Weights only load failed" in msg or "weights_only" in msg
        if not is_weights_only:
            raise

        trusted = (
            _env_truthy("FLASH_SVD_TRUST_PICKLE")
            or _looks_like_flashsvd_checkpoint(path)
            or _looks_like_dobi_checkpoint(path)
        )
        if not trusted:
            raise RuntimeError(
                f"Refusing to load pickled checkpoint with weights_only=False: {path}\n"
                "PyTorch>=2.6 defaults torch.load(weights_only=True), which can't load "
                "our pickled {'model': model, 'tokenizer': tok} format.\n"
                "If you trust this checkpoint, re-run with: FLASH_SVD_TRUST_PICKLE=1"
            ) from e

        # Trusted fallback: allow pickled model/tokenizer objects.
        with _pickle_compat_shims():
            return _load(weights_only=False)
