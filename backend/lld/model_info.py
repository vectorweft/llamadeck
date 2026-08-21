"""Curated, family-keyed narrative documentation for models in the registry.

This is the *human-facing* counterpart to model_defaults.py. Where
model_defaults answers "what numbers should the sampler use", this module
answers "what does the model expect from the prompt, how does it behave, what
do I need to know to deploy it well".

Each entry is hand-written from first-party model cards (Google, Unsloth,
HF blog) and small-model best-practices. Entries are matched by regex against
the same name hint (filename + general.name + base_model) used elsewhere.

The first regex that matches wins. Order specific → generic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RecommendedDrafter:
    """Hint about which model family / file pattern works as a vocab-matched
    speculative-decoding drafter for this target. The Models UI uses this to
    pre-select / suggest a drafter when wiring up a preset."""
    label: str                            # human-readable ("Gemma-4 E2B (~2 GB)")
    family_pattern: str                   # regex matched against ModelEntry.family (case-insensitive)
    name_pattern: str                     # regex matched against the GGUF filename (case-insensitive)
    rationale: str                        # one-sentence why
    draft_max: int | None = None          # suggested --draft-max
    draft_min: int | None = None          # suggested --draft-min

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family_pattern": self.family_pattern,
            "name_pattern": self.name_pattern,
            "rationale": self.rationale,
            "draft_max": self.draft_max,
            "draft_min": self.draft_min,
        }


@dataclass
class ModelInfo:
    family: str
    summary: str                          # one-sentence elevator pitch
    prompt_format: str                    # markdown — chat template / role contract
    behavior: list[str] = field(default_factory=list)  # bullet behavioral notes
    deployment: list[str] = field(default_factory=list)  # llama-server flags / VRAM hints
    caveats: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)  # [{title,url}]
    recommended_drafter: RecommendedDrafter | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "summary": self.summary,
            "prompt_format": self.prompt_format,
            "behavior": self.behavior,
            "deployment": self.deployment,
            "caveats": self.caveats,
            "references": self.references,
            "recommended_drafter": self.recommended_drafter.to_dict() if self.recommended_drafter else None,
        }


_GEMMA_4 = ModelInfo(
    family="Gemma-4",
    summary=(
        "Google's frontier multimodal open model (text+image, audio on E2B/E4B). "
        "31B variant: 30.7B params, 256K context, hybrid attention (1024-token "
        "sliding window + global). Strong reasoning, coding, multilingual (140+ "
        "languages pretrain, 35+ instruction-tuned)."
    ),
    prompt_format=(
        "Standard `system` / `user` / `assistant` roles — Gemma 4 introduced "
        "**native system-prompt support** (Gemma 3 did not). The chat template "
        "is shipped inside the GGUF (`tokenizer.chat_template`) and `--jinja` "
        "in llama-server picks it up automatically.\n\n"
        "**Thinking mode** is opt-in: prepend the `<|think|>` token to the "
        "system prompt to enable step-by-step reasoning. Output is structured "
        "as `<|channel>thought ... <channel|>` then the final answer. "
        "**Important:** when carrying multi-turn history, drop the thought "
        "block — only the final assistant text should remain.\n\n"
        "**Multimodal:** place image/audio content **before** text in the same "
        "user message for best results. Visual token budgets: 70/140/280/560/"
        "1120 — lower for classification/captioning, higher for OCR."
    ),
    behavior=[
        "Recommended sampling: temperature=1.0, top_p=0.95, top_k=64, min_p=0.0 (Google + Unsloth).",
        "Native function-calling / structured tool use — works with llama-server's `--jinja` tool-call parsing.",
        "Image input requires the matching `mmproj-*.gguf` projector file (Unsloth ships one alongside the GGUFs).",
        "Knowledge cutoff: January 2025.",
    ],
    deployment=[
        "31B Q4_K_XL ≈ 19 GB on disk; fits comfortably in 32 GB VRAM with 32K–128K context.",
        "Start with `--ctx-size 32768` for responsiveness, scale to 256K only when actually needed (KV cost grows linearly).",
        "`--cache-type-k q8_0 --cache-type-v q8_0` halves KV-cache RAM with negligible quality loss.",
        "`--flash-attn` recommended on RTX 5090 (CUDA fattn).",
        "**MTP / Multi-Token Prediction:** upstream llama.cpp supports Gemma 4 MTP since PR #23398 (`src/models/gemma4.cpp`) — the ik_llama.cpp fork is no longer needed. Use `--spec-type draft-mtp -md <mtp-head>.gguf`; the MTP head GGUF (~0.5 GB for 31B) pairs with the main model. Community conversion: `Radamanthys11/Gemma-4-31B-it-assistant-GGUF`.",
        "Upstream MTP invocation: `--spec-type draft-mtp -md mtp-gemma-4-31B-it.gguf` (MTP head reads the target's hidden state; `-ngld` has no effect for draft-mtp).",
    ],
    caveats=[
        "Carrying the `<|channel>thought` block forward in multi-turn breaks the model — strip it before the next turn.",
        "Hybrid attention with 1024-token sliding window means very long contexts behave differently from a pure-global model; long-context evals (MRCR-128K) drop to 66% — don't assume 256K is free quality-wise.",
        "Image-before-text ordering matters; reversed order degrades quality.",
    ],
    references=[
        {"title": "Unsloth: Gemma 4 31B GGUF model card", "url": "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF"},
        {"title": "Unsloth: Run Gemma 4 locally", "url": "https://unsloth.ai/docs/models/gemma-4"},
        {"title": "Google: Accelerating Gemma 4 with MTP drafters", "url": "https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/"},
        {"title": "Google AI: MTP overview", "url": "https://ai.google.dev/gemma/docs/mtp/overview"},
        {"title": "Community MTP drafter GGUF (Radamanthys11)", "url": "https://huggingface.co/Radamanthys11/Gemma-4-31B-it-assistant-GGUF"},
        {"title": "llama.cpp PR #23398 — Gemma4 MTP", "url": "https://github.com/ggml-org/llama.cpp/pull/23398"},
        {"title": "HF blog: Welcome Gemma 4", "url": "https://huggingface.co/blog/gemma4"},
    ],
    recommended_drafter=RecommendedDrafter(
        label="Gemma-4 MTP head (~0.5 GB) — spec_type=draft-mtp, upstream",
        family_pattern=r"Gemma-?4",
        name_pattern=r"mtp-gemma-?4",
        rationale="Native MTP head (PR #23398, in upstream): set spec_type to draft-mtp. More accurate drafts than a classic drafter, ~0.5 GB extra VRAM. Classic draft-simple with E2B remains an alternative.",
        draft_max=3,
        draft_min=None,
    ),
)

_GEMMA_3 = ModelInfo(
    family="Gemma-3",
    summary="Google's previous-generation multimodal open model (text+image). No native system-prompt role.",
    prompt_format=(
        "Uses `user` / `model` roles (no `system`). Inject system instructions "
        "as a prefix to the first user message. Chat template is in the GGUF; "
        "use `--jinja` in llama-server."
    ),
    behavior=[
        "Recommended sampling: temperature=1.0, top_p=0.95, top_k=64, min_p=0.0.",
        "Multimodal via mmproj projector (image input).",
    ],
    deployment=[
        "12B/27B variants run well on a single 32 GB GPU at 4-bit.",
    ],
    caveats=[
        "No native system role — using `system` content with naive templates is silently dropped or merged.",
    ],
    references=[
        {"title": "Google: Gemma docs", "url": "https://ai.google.dev/gemma/docs"},
    ],
)

_QWEN_36 = ModelInfo(
    family="Qwen3.6",
    summary="Alibaba's Qwen3.6 — dual-mode (thinking / non-thinking) reasoning model with strong tool use and coding.",
    prompt_format=(
        "Standard `system` / `user` / `assistant`. Mode is selected per-request "
        "via `enable_thinking` in the chat template (or by emitting `/no_think` "
        "in the user message). Thinking mode emits `<think>...</think>` blocks; "
        "**strip these before re-sending history** in multi-turn."
    ),
    behavior=[
        "Thinking ON: temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0.",
        "Non-thinking: temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5 (reduces repetition; >2.0 may cause language mixing per Unsloth).",
        "Native tool-calling via Hermes-style XML when `--jinja` is on.",
    ],
    deployment=[
        "MoE variants (e.g. 35B-A3B) need the full param count in VRAM but only the active experts in compute — large mem footprint, fast inference.",
    ],
    caveats=[
        "Carrying `<think>` blocks in history breaks subsequent turns.",
        "Mode-aware sampling matters — using thinking-mode params in non-thinking mode produces verbose, looping outputs.",
    ],
    references=[
        {"title": "Unsloth: Qwen3 best practices", "url": "https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507"},
    ],
)

_QWEN_25_CODER = ModelInfo(
    family="Qwen2.5-Coder",
    summary="Code-specialized Qwen2.5 — strong on code completion, repo-level edits, FIM (fill-in-middle).",
    prompt_format=(
        "Standard chat for instruct variant. FIM uses `<|fim_prefix|>` / "
        "`<|fim_suffix|>` / `<|fim_middle|>` special tokens — pass raw, not "
        "via chat template."
    ),
    behavior=[
        "Recommended sampling: temp=0.7, top_p=0.80, top_k=20, repeat_penalty=1.05.",
        "Lower temperature (0.2–0.3) recommended for deterministic code completion.",
    ],
    deployment=[],
    caveats=[],
    references=[
        {"title": "Qwen2.5-Coder model card", "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct"},
    ],
)

_QWQ = ModelInfo(
    family="QwQ",
    summary="Qwen QwQ — reasoning-only preview model, always emits chain-of-thought.",
    prompt_format=(
        "Standard chat roles. Output **always** contains `<think>...</think>` "
        "blocks — there is no non-thinking mode. Strip them before re-sending."
    ),
    behavior=[
        "Recommended sampling: temp=0.6, top_p=0.95, top_k=20, min_p=0.0.",
        "Long, exploratory thinking — expect 1k–10k thought tokens before the final answer.",
    ],
    deployment=[
        "Budget output tokens generously (e.g. `--predict 16384`); cutting off mid-thought yields garbage.",
    ],
    caveats=[
        "Not suitable for low-latency / agentic tool-calling workflows — first-token latency is high.",
    ],
    references=[
        {"title": "QwQ model card", "url": "https://huggingface.co/Qwen/QwQ-32B-Preview"},
    ],
)


_HY_V3 = ModelInfo(
    family="Hunyuan-V3 (Hy3)",
    summary=(
        "Tencent's Hunyuan V3 — very large sparse MoE (192 experts, 8 active per "
        "token + 1 shared, size label 192x10B, 81 layers, 256K context) with an "
        "embedded MTP (next-token-prediction) head. llama.cpp arch `hy_v3`, "
        "supported upstream including the MTP/nextn layer."
    ),
    prompt_format=(
        "Standard `system` / `user` / `assistant` roles; the chat template ships "
        "inside the GGUF — run llama-server with `--jinja` and it is picked up "
        "automatically."
    ),
    behavior=[
        "GGUF embeds producer sampling: temperature=0.9, top_p=1.0, top_k disabled — trust these as the base.",
        "Sparse MoE: only ~8/192 experts run per token, so CPU-offloaded experts remain usable (unlike a dense model of this size).",
    ],
    deployment=[
        "Even at IQ1_M the file is ~85 GB — it does NOT fit a 32 GB GPU. Use MoE offload: `--cpu-moe` (all experts in RAM) or better `--n-cpu-moe N` (first N expert layers in RAM, rest on GPU). The preset editor's fit panel computes N for you.",
        "Keep `-ngl 999` when using `--n-cpu-moe` — attention/embeddings/shared expert stay on GPU; only routed experts move to RAM.",
        "Embedded MTP head (`blk.*.nextn.*` tensors, filename `-mtp`): enable speculative decoding with `spec_type=draft-mtp`, no separate drafter file needed.",
        "KV cache at 256K context is huge (~175 KB/token) — start at 16K–32K ctx with q8_0 KV and grow only as needed.",
    ],
    caveats=[
        "IQ1_M is an extreme 1.8-bit quant — expect visibly lower quality than the model's benchmark numbers; prefer larger quants if the hardware ever allows.",
        "With most experts in RAM, token speed is bound by RAM bandwidth; don't run RAM-hungry apps alongside it.",
    ],
    references=[
        {"title": "Tencent Hunyuan on Hugging Face", "url": "https://huggingface.co/tencent"},
    ],
)


# Order: specific → generic. First match wins.
_INFO_PATTERNS: list[tuple[re.Pattern[str], ModelInfo]] = [
    (re.compile(r"Hy[-_]?3|Hy[-_]?V3|Hunyuan", re.I), _HY_V3),
    (re.compile(r"Gemma-?4", re.I), _GEMMA_4),
    (re.compile(r"Gemma-?3", re.I), _GEMMA_3),
    (re.compile(r"Qwen3\.6", re.I), _QWEN_36),
    (re.compile(r"Qwen2\.5[-_.]?Coder", re.I), _QWEN_25_CODER),
    (re.compile(r"QwQ", re.I), _QWQ),
]


def lookup_info(name_hint: str) -> ModelInfo | None:
    """Return the first ModelInfo whose pattern matches the hint, or None."""
    for pat, info in _INFO_PATTERNS:
        if pat.search(name_hint):
            return info
    return None


def info_for_path(model_path: str, gguf_name: str | None = None, base_model: str | None = None) -> ModelInfo | None:
    """Convenience: build a hint from the standard sources and look up."""
    hint = " ".join(filter(None, [gguf_name, base_model, Path(model_path).stem]))
    return lookup_info(hint)
