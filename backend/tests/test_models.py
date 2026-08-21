from __future__ import annotations

from lld.models import _detect_family, _detect_quant, _excluded, _is_mmproj


def test_family_detection():
    assert _detect_family("Qwen3.6-35B-A3B-MXFP4_MOE.gguf") == "Qwen3.6"
    assert _detect_family("Qwen3.5-9B-UD-Q4_K_XL.gguf") == "Qwen3.5"
    assert _detect_family("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf") == "Qwen2.5-Coder"
    assert _detect_family("gemma-3-12b-it-Q4_K_M.gguf") == "Gemma-3"
    assert _detect_family("QwQ-32B-Q4_K_M.gguf") == "QwQ"
    assert _detect_family("Olmo-3-32B-Think-Q4_K_M.gguf") == "OLMo-3"
    assert _detect_family("DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf") == "DeepSeek-R1"
    assert _detect_family("gpt-oss-20b-MXFP4.gguf") == "gpt-oss"
    assert _detect_family("Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf") == "Ministral"


def test_quant_detection():
    assert _detect_quant("Qwen3.5-9B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert _detect_quant("Qwen3.6-35B-A3B-MXFP4_MOE.gguf") == "MXFP4_MOE"
    assert _detect_quant("DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf") == "Q8_0"
    assert _detect_quant("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert _detect_quant("gpt-oss-20b-MXFP4.gguf") == "MXFP4"


def test_mmproj_and_vocab_excluded():
    assert _is_mmproj("mmproj-F16.gguf")
    assert _is_mmproj("mmproj-BF16-Qwen3.6.gguf")
    assert _excluded("mmproj-F16.gguf")
    assert _excluded("ggml-vocab-qwen.gguf")
    assert not _excluded("Qwen3.6-35B.gguf")
