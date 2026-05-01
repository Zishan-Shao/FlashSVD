from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import demo_support
from utils import model_utils


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        del text, add_special_tokens
        return {
            "input_ids": torch.tensor([[1, 5, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids)


def test_public_api_exports():
    assert callable(demo_support.get_model_from_local)
    assert callable(demo_support.get_model_from_source)
    assert callable(demo_support.decode_kvcache_eval)
    assert callable(demo_support.configure_runtime)
    assert callable(demo_support.generate_text)


def test_configure_runtime_flashsvd_smoke():
    cfg = demo_support.configure_runtime(
        mode="flashsvd",
        ffn_backend="auto",
        enable_mlp_graph=True,
        mlp_graph_scope="layer_tail",
        enable_flash_dense_attn=True,
    )
    assert cfg["SVDLLM_FLASH_FALLBACK"] is None
    assert cfg["FLASH_SVD_MLP_CUDA_GRAPH"] == "1"
    assert cfg["FLASH_SVD_MLP_CUDA_GRAPH_SCOPE"] == "layer_tail"
    assert cfg["FLASH_SVD_ENABLE_DENSE_ATTN_DECODE"] == "1"
    assert cfg["FLASH_SVD_DENSE_DECODE_BACKEND"] == "auto"
    assert cfg["FLASH_SVD_DENSE_DECODE_GRAPH"] == "1"
    assert cfg["FLASH_SVD_DENSE_DECODE_AUTOTUNE_SCOPE"] == "attention"


def test_lowrankarena_review_alias_resolves_from_env(monkeypatch):
    monkeypatch.setenv(model_utils.LOWRANKARENA_REPO_ENV, "anonymous/review-artifact")

    assert model_utils._parse_hf_repo_subfolder_source("LowRankArena::llama_7b/example") == (
        "LowRankArena",
        "llama_7b/example",
    )
    assert model_utils._parse_hf_repo_subfolder_source("LowRankArena/llama_7b/example") == (
        "LowRankArena",
        "llama_7b/example",
    )
    assert model_utils._resolve_hf_repo_id("LowRankArena") == "anonymous/review-artifact"


def test_generate_text_smoke_cpu():
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    tokenizer = _TinyTokenizer()
    model = demo_support.cast_model_for_inference(model, dtype="fp32", device="cpu")
    result = demo_support.generate_text(
        model,
        tokenizer,
        prompt="smoke",
        max_new_tokens=4,
        device="cpu",
    )
    assert result["prompt_token_count"] == 3
    assert result["generated_token_count"] == 4
    assert len(result["output_ids"]) == 7
    assert len(result["completion_ids"]) == 4
