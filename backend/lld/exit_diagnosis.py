"""Turns a llama-server crash into a sentence the user can act on.

`exited with code 1` is what the supervisor used to report for every failed
start, and it is worth nothing: the reason is always sitting a few lines up in
the server's own log, which the user has to know to go find. This module reads
the captured output back and answers "why", preferring an explanation with a
next step over a verbatim copy of the C++ error.

The patterns below are the failures a first-time user on a fresh box actually
hits — a model that doesn't fit, a split model with a part missing or renamed,
a llama.cpp build older than the model, a busy port. Anything unrecognised
falls back to the first error line verbatim, which still beats an exit code.
"""
from __future__ import annotations

import re

# llama.cpp log lines carry a severity field: "0.00.246 E gguf_init_from_file: …".
_ERROR_LINE = re.compile(r"^\S+\s+E\s+(?P<body>.*)$")
# Its own timestamp/severity prefix, stripped before quoting a line back.
_PREFIX = re.compile(r"^\S+\s+[EWID]\s+")


def _error_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        m = _ERROR_LINE.match(line)
        if m:
            out.append(m.group("body").strip())
    return out


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


def diagnose_exit(returncode: int | None, log_lines: list[str]) -> str:
    """One-line reason for a llama-server exit, with a fix where we know one."""
    tail = list(log_lines)[-400:]
    blob = _joined(tail)
    errors = _error_lines(tail)

    # --- the auto-fit probe hit a full GPU -----------------------------------
    # `--fit` defaults to ON: before loading anything, llama.cpp dry-runs a
    # llama_context on EVERY registered device to measure free memory. On a
    # CUDA device that means creating a context and a stream, which itself
    # needs VRAM — so a preset that touches the GPU not at all (`-ngl 0`, even
    # `--device none`) still aborts when another model has filled the card.
    # The failure looks like a plain out-of-memory, which sends the user off
    # shrinking a model that was never the problem. Checked before the generic
    # VRAM rules so the specific cause wins.
    if re.search(r"cudaStreamCreateWithFlags|hipStreamCreateWithFlags", blob) and re.search(
        r"out of memory", blob
    ):
        return (
            "the model never got as far as loading: llama.cpp's auto-fit probe (--fit is on by "
            "default) opened a context on a GPU that has no free VRAM left. Add `--fit off` to "
            "extra_flags for a preset that does not use that card, or free VRAM on it. "
            "-ngl 0 alone does not avoid this — the probe runs before the model is placed."
        )

    # --- out of VRAM ---------------------------------------------------------
    m = re.search(
        r"allocating ([\d.]+) MiB on device (\d+): cudaMalloc failed: out of memory", blob
    )
    if m:
        need_gb = float(m.group(1)) / 1024
        return (
            f"out of VRAM: llama.cpp asked device {m.group(2)} for {need_gb:.1f} GB and the "
            f"allocation failed. Move expert layers to RAM with --n-cpu-moe N (MoE models), "
            f"lower n_gpu_layers, or reduce ctx_size."
        )
    if re.search(r"failed to allocate (CUDA|ROCm|Vulkan|Metal)\S* buffer", blob):
        return (
            "out of VRAM while allocating a model buffer. Move expert layers to RAM with "
            "--n-cpu-moe N (MoE models), lower n_gpu_layers, or reduce ctx_size."
        )
    if re.search(r"unable to allocate .*(host|CPU) buffer|std::bad_alloc|Cannot allocate memory", blob):
        return (
            "out of system RAM while loading the model. Close other applications, move fewer "
            "layers to the CPU, or use a smaller quantization."
        )

    # --- split models --------------------------------------------------------
    m = re.search(r"failed to load GGUF split from (\S+)", blob) or re.search(
        r"failed to open GGUF file '([^']+)'", blob
    )
    if m:
        name = m.group(1).strip("'").rsplit("/", 1)[-1]
        return (
            f"missing part of a multi-part model: {name} was not found. llama.cpp derives every "
            f"part's name from part 1's filename, so all parts must sit in one folder sharing the "
            f"same prefix — a part 1 named differently from the rest causes exactly this."
        )

    # --- build too old for the model ----------------------------------------
    m = re.search(r"unknown model architecture: '([^']+)'", blob)
    if m:
        return (
            f"this llama.cpp build does not know the '{m.group(1)}' architecture. Update and "
            f"rebuild llama.cpp (Build page), then try again."
        )
    if re.search(r"unknown (pre-tokenizer|tokenizer) type", blob):
        return (
            "this llama.cpp build does not know the model's tokenizer. Update and rebuild "
            "llama.cpp (Build page), then try again."
        )

    # --- corrupt / truncated file -------------------------------------------
    if re.search(r"invalid magic|wrong number of tensors|tensor .* has wrong shape|invalid GGUF", blob):
        return (
            "the model file is not valid GGUF — most likely a truncated or corrupted download. "
            "Re-download it and check the file size against the source."
        )

    # --- context too large ---------------------------------------------------
    if re.search(r"n_ctx_per_seq .* > n_ctx_train|requested context .* exceeds", blob):
        return "the requested ctx_size is larger than the model supports. Lower ctx_size."

    # --- a flag llama-server would not take ----------------------------------
    # Argument errors are printed before the logger exists, so they carry no
    # "E" severity field and `_error_lines` never sees them. The usage block
    # that follows ends in "to show complete usage, run with -h" — which is
    # what the last-line fallback used to quote back, telling the user
    # precisely nothing about the flag that caused it.
    m = re.search(r'error while handling argument "(?P<flag>[^"]+)":\s*(?P<why>[^\n]+)', blob)
    if m:
        why = m.group("why").strip().rstrip(".")
        flag = m.group("flag")
        if "expected value" in why:
            return (
                f"llama-server needs a value for {flag} and refuses the whole "
                f"command line without one. Give it one in extra_flags "
                f"(see the flag list in the Command tab), or remove it."
            )
        return (
            f"llama-server rejected {flag}: {why}. Fix or remove it in "
            f"extra_flags, or in the Command tab."
        )

    # --- port ----------------------------------------------------------------
    if re.search(r"bind.*(in use|EADDRINUSE)|couldn't bind to server socket", blob):
        return "the port is already in use by another process. Pick a different port."

    # --- unrecognised: quote llama.cpp's own first error ---------------------
    if errors:
        first = errors[0]
        return f"llama-server failed: {first[:300]}"
    for line in reversed(tail):
        stripped = _PREFIX.sub("", line).strip()
        if stripped and not stripped.startswith("[LlamaDeck]"):
            return f"llama-server exited (code {returncode}). Last output: {stripped[:300]}"
    return f"exited with code {returncode}"
