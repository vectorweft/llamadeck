"""Concrete services managed by the GPU broker.

Each service implements the `Service` protocol (lld.gpu_broker.Service):
  - LlmService    — wraps existing supervisor (preset → llama-server)
  - ComfyService  — ComfyUI subprocess (placeholder until faz-1.7)
  - XttsService   — XTTS subprocess (placeholder until faz-1.7)
"""
from __future__ import annotations

from .llm import LlmService
from .placeholder import PlaceholderService
from .tts import TtsService

__all__ = ["LlmService", "PlaceholderService", "TtsService"]
