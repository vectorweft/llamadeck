"""Download target layout: <brand>/<series>/<base_model>.

The failure this exists for: a repo no classifier matches falls back to
(owner, repo-name) for brand/series, and derive_base_model() strips the quant
suffix down to that same repo name — so the model landed in
unsloth/LFM2.5-1.2B-Thinking/LFM2.5-1.2B-Thinking, one folder deeper than the
UI (and the model scanner) expects.
"""
from __future__ import annotations

import pytest

from lld.hf import classify, derive_base_model, target_segments


def _layout(repo_id: str, filename: str) -> str:
    brand, series = classify(repo_id)
    return "/".join(target_segments(brand, series, derive_base_model(repo_id, filename)))


@pytest.mark.parametrize(
    "repo_id,filename,expected",
    [
        # unclassified repo: series and base_model collapse to one level
        (
            "unsloth/LFM2.5-1.2B-Thinking-GGUF",
            "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
            "unsloth/LFM2.5-1.2B-Thinking",
        ),
        # ...including the mmproj of the same repo, so it stays next to the weights
        (
            "unsloth/LFM2.5-1.2B-Thinking-GGUF",
            "mmproj-F16.gguf",
            "unsloth/LFM2.5-1.2B-Thinking",
        ),
        # classified repo: all three levels are distinct and all three are kept
        (
            "unsloth/DeepSeek-V4-Flash-0731-GGUF",
            "UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00002.gguf",
            "DeepSeek/DeepSeek-V4/DeepSeek-V4-Flash-0731",
        ),
        (
            "bartowski/Qwen_Qwen3-32B-GGUF",
            "Qwen_Qwen3-32B-Q4_K_M.gguf",
            "Qwen/Qwen3/Qwen_Qwen3-32B",
        ),
    ],
)
def test_layout(repo_id: str, filename: str, expected: str) -> None:
    assert _layout(repo_id, filename) == expected


def test_repeat_is_case_insensitive_and_empty_segments_dropped() -> None:
    assert target_segments("unsloth", "Gemma-3", "gemma-3") == ["unsloth", "Gemma-3"]
    assert target_segments("Qwen", "", "Qwen3-32B") == ["Qwen", "Qwen3-32B"]
