"""The ways a failed start used to go unexplained.

Three real defects, each of which turned an actionable error into silence:
a log line past asyncio's 64 KiB limit killed the log reader for the rest of
the process's life; the crash-loop notice overwrote the diagnosis that said
*why*; and a preset pointing at a directory that does not exist was retried
three times before giving up with "manual restart needed" — for something no
restart can fix.
"""
from __future__ import annotations

import os
import stat

import pytest

from lld.settings import LlamaServerConfig
from lld.supervisor import ProcessHandle, SupervisorError


def _fake_server(tmp_path, script: str) -> str:
    p = tmp_path / "fake-llama-server"
    p.write_text("#!/bin/sh\n" + script)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _handle(binary: str, **cfg_kw) -> ProcessHandle:
    cfg = LlamaServerConfig(
        name="t", model_path=None, host="127.0.0.1", port=18771, **cfg_kw
    )
    h = ProcessHandle("t", cfg, binary)
    h._schedule_autorestart = lambda: None  # isolate from restart bookkeeping
    return h


async def _run_to_exit(h: ProcessHandle) -> None:
    import asyncio

    await h.start()
    for _ in range(400):
        await asyncio.sleep(0.01)
        if h._last_error is not None:
            break
    await h._cancel_readers()


async def test_line_over_64k_does_not_blind_the_log_reader(tmp_path):
    """Nothing bounds how long a line llama-server may print. One past
    StreamReader's 64 KiB limit made readline() raise, and the consumer died
    with it — losing every later line, including the one that said why the
    process exited."""
    binary = _fake_server(
        tmp_path,
        'echo "0.00.100.000 I srv    load: loading model"\n'
        "awk 'BEGIN{s=\"\";for(i=0;i<70000;i++)s=s \"x\";"
        'print "0.00.200.000 I srv  chat_template: " s}\'\n'
        'echo "0.00.300.000 I srv    load: listening"\n'
        'echo "0.00.900.000 E ggml_backend_cuda_buffer_type_alloc_buffer: '
        'allocating 8192.00 MiB on device 0: cudaMalloc failed: out of memory"\n'
        "exit 1\n",
    )
    h = _handle(binary)
    await _run_to_exit(h)

    ring = list(h._log_ring)
    assert any("listening" in ln for ln in ring), "lines after the long one were lost"
    assert any("cudaMalloc" in ln for ln in ring)
    # The oversized line is kept, truncated — not dropped, not fatal.
    assert any("chat_template" in ln for ln in ring)
    assert any("[line truncated]" in ln for ln in ring)
    # And the diagnosis reflects the real cause rather than the last line
    # before the reader died.
    assert "out of VRAM" in h._last_error


async def test_crash_loop_notice_keeps_the_diagnosis(tmp_path):
    """Suppressing auto-restart must add to the reason, not replace it."""
    binary = _fake_server(
        tmp_path,
        'echo "0.00.100.000 E llama_model_load: error loading model: '
        "unknown model architecture: 'deepseek4'\"\n"
        "exit 1\n",
    )
    h = _handle(binary)
    del h._schedule_autorestart  # exercise the real bookkeeping
    h._autorestart_attempts_window = [__import__("time").time()] * 3  # already looping

    await _run_to_exit(h)

    assert "deepseek4" in h._last_error, "actionable diagnosis was overwritten"
    assert "crash-looped" in h._last_error, "loop state should still be reported"
    if h._autorestart_task:
        h._autorestart_task.cancel()


async def test_missing_models_dir_is_refused_before_spawning(tmp_path):
    binary = _fake_server(tmp_path, "exit 1\n")
    h = _handle(binary, mode="router", models_dir=str(tmp_path / "nope"))
    with pytest.raises(SupervisorError) as e:
        await h.start()
    assert "models_dir does not exist" in str(e.value)
    assert h._proc is None, "nothing should have been spawned"


async def test_missing_model_file_is_refused_before_spawning(tmp_path):
    binary = _fake_server(tmp_path, "exit 1\n")
    h = _handle(binary)
    h.cfg.model_path = str(tmp_path / "gone.gguf")
    with pytest.raises(SupervisorError) as e:
        await h.start()
    assert "model file not found" in str(e.value)


async def test_existing_paths_do_not_block_a_start(tmp_path):
    """The guard must stay out of the way when the paths are real."""
    binary = _fake_server(tmp_path, 'echo "0.00.1 I srv up"\nexit 0\n')
    models = tmp_path / "models"
    models.mkdir()
    h = _handle(binary, mode="router", models_dir=str(models))
    await h.start()
    assert h._proc is not None
    await h.stop()


def test_router_seed_uses_the_configured_models_root(monkeypatch, tmp_path):
    """The seeded router preset pointed at ~/llama.cpp/models regardless of
    where the user's models actually live, so it could never start."""
    from lld import presets as presets_mod

    root = tmp_path / "gguf"
    root.mkdir()

    class _S:
        hf_models_root = str(root)

    monkeypatch.setattr(presets_mod, "load_settings", lambda: _S(), raising=False)
    monkeypatch.setattr("lld.settings.load_settings", lambda: _S())

    cfg = presets_mod.default_seeds()[0]
    assert cfg.models_dir == str(root)
    assert os.path.isdir(cfg.models_dir)


# --------------------------------------------------------------------------
# Device pinning: `-dev` ids are backend-specific, so a preset pinned to
# Vulkan1 is meaningless to a CUDA-only build. Catch it before spawning.
# --------------------------------------------------------------------------

# Scans every argument rather than $1: the probe puts --rpc first (a device
# list must be requested last or llama-server exits before reading it), so a
# fixture keyed on $1 breaks the moment an RPC server is configured.
_LIST_DEVICES_SH = """
for a in "$@"; do [ "$a" = "--list-devices" ] && LIST=1; done
if [ -n "$LIST" ]; then
  echo "Available devices:"
  echo "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)"
  exit 0
fi
sleep 5
"""


async def test_start_rejects_a_device_this_build_cannot_offload_to(tmp_path):
    from lld import devices as devices_mod

    devices_mod.invalidate_cache()
    h = _handle(_fake_server(tmp_path, _LIST_DEVICES_SH), devices=["Vulkan1"])
    with pytest.raises(SupervisorError) as e:
        await h._check_devices()
    # The message has to name what *is* available, or the user is left guessing.
    assert "Vulkan1" in str(e.value)
    assert "CUDA0" in str(e.value)


async def test_start_accepts_a_device_the_build_exposes(tmp_path):
    from lld import devices as devices_mod

    devices_mod.invalidate_cache()
    h = _handle(_fake_server(tmp_path, _LIST_DEVICES_SH), devices=["CUDA0"])
    await h._check_devices()  # must not raise


async def test_unqueryable_binary_does_not_block_a_pinned_preset(tmp_path):
    """No device list is not evidence the selection is wrong — llama-server
    still gets to render its own error."""
    from lld import devices as devices_mod

    devices_mod.invalidate_cache()
    h = _handle(_fake_server(tmp_path, "exit 1\n"), devices=["CUDA0"])
    await h._check_devices()  # must not raise
