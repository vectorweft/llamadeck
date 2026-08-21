"""RPC offload servers: the escape hatch for a backend the main binary can't host.

CUDA and HIP cannot live in one llama-server (ggml-hip compiles the ggml-cuda
sources with hipcc), so driving an NVIDIA and an AMD card natively at the same
time means running one of them behind `ggml-rpc-server`. These tests pin the
parts that silently misplace a model when they are wrong.
"""
from __future__ import annotations

import pytest

from lld.argv import from_argv, to_argv
from lld.devices import parse_list_devices, selectable_devices
from lld.rpc_server import RpcProcess, needs_rpc
from lld.settings import LlamaServerConfig, RpcServerConfig

WITH_RPC = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 32566 MiB free)
  RPC0: 127.0.0.1:50052 (32624 MiB, 32558 MiB free)
"""


def test_rpc_row_is_parsed_and_selectable():
    devs = {d.id: d for d in parse_list_devices(WITH_RPC, "AMD Ryzen 9 9950X")}
    rpc = devs["RPC0"]
    assert rpc.backend == "RPC"
    assert rpc.rpc_endpoint == "127.0.0.1:50052"
    assert rpc.total_mb == 32624
    assert rpc.selectable is True


def test_rpc_row_warns_when_it_may_be_a_local_card_reached_twice():
    """The R9700 shows up as Vulkan1 and again as RPC0 when the RPC server
    exports it. Nothing in the protocol proves they are the same card, so this
    warns instead of excluding — the endpoint could be another machine."""
    devs = {d.id: d for d in parse_list_devices(WITH_RPC, "AMD Ryzen 9 9950X")}
    assert devs["RPC0"].may_alias == "Vulkan1"
    # A warning, not an exclusion: both stay pickable.
    assert devs["RPC0"].duplicate_of is None
    assert [d.id for d in selectable_devices(list(devs.values()))] == [
        "CUDA0", "Vulkan1", "RPC0",
    ]


def test_rpc_row_without_a_size_match_is_not_flagged():
    text = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
  RPC0: 10.0.0.5:50052 (81920 MiB, 81000 MiB free)
"""
    devs = {d.id: d for d in parse_list_devices(text, "cpu")}
    assert devs["RPC0"].may_alias is None


def test_needs_rpc_only_fires_on_rpc_ids():
    assert needs_rpc(["RPC0"]) is True
    assert needs_rpc(["CUDA0", "RPC1"]) is True
    assert needs_rpc(["CUDA0", "Vulkan1"]) is False
    assert needs_rpc([]) is False
    assert needs_rpc(None) is False


def test_rpc_flag_precedes_device_flag(monkeypatch):
    """llama-server registers RPC devices as it parses --rpc, so -dev naming
    RPC0 before that point resolves to nothing."""
    import lld.argv as argv_mod

    monkeypatch.setattr(argv_mod, "rpc_flag_value", lambda: "127.0.0.1:50052")
    cfg = LlamaServerConfig(name="p", model_path="/m.gguf", devices=["RPC0"])
    argv = to_argv(cfg, "llama-server")
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50052"
    assert argv.index("--rpc") < argv.index("-dev")


def test_no_rpc_flag_for_a_local_only_selection(monkeypatch):
    import lld.argv as argv_mod

    monkeypatch.setattr(argv_mod, "rpc_flag_value", lambda: "127.0.0.1:50052")
    cfg = LlamaServerConfig(name="p", model_path="/m.gguf", devices=["CUDA0"])
    assert "--rpc" not in to_argv(cfg, "llama-server")


def test_rpc_flag_is_not_parsed_back_into_extra_flags():
    """It is regenerated from settings at every launch; keeping it would emit
    it twice on the next start."""
    cfg = from_argv([
        "llama-server", "-m", "/m.gguf",
        "--rpc", "127.0.0.1:50052", "-dev", "RPC0",
    ])
    assert cfg.devices == ["RPC0"]
    assert cfg.extra_flags == []


def test_server_argv_pins_the_exported_device():
    """Exporting everything would include the RPC host's CPU, which is never
    what the user meant by 'use the AMD card'."""
    proc = RpcProcess(cfg=RpcServerConfig(
        name="amd", binary="/x/ggml-rpc-server", host="127.0.0.1",
        port=50052, devices=["ROCm0"],
    ))
    assert proc.argv() == [
        "/x/ggml-rpc-server", "-H", "127.0.0.1", "-p", "50052", "-d", "ROCm0",
    ]


def test_server_without_a_binary_fails_with_a_usable_message():
    proc = RpcProcess(cfg=RpcServerConfig(name="amd", binary="/nope/missing"))
    assert proc.resolve_binary() == "/nope/missing"


@pytest.mark.asyncio
async def test_stop_never_kills_a_server_we_did_not_start():
    """An externally launched server on the same port is usable but not ours."""
    proc = RpcProcess(cfg=RpcServerConfig(name="amd"))
    assert proc.owned() is False
    await proc.stop()  # must be a no-op, not an error


@pytest.mark.asyncio
async def test_probe_puts_rpc_before_list_devices(tmp_path, monkeypatch):
    """llama-server prints the device list and exits the moment it parses
    --list-devices, so --rpc placed after it is never read and every remote
    device silently vanishes from the picker."""
    import stat

    from lld import devices as devices_mod

    recorder = tmp_path / "recorder"
    seen = tmp_path / "argv.txt"
    recorder.write_text(f'#!/bin/sh\necho "$@" > {seen}\necho "Available devices:"\n')
    recorder.chmod(recorder.stat().st_mode | stat.S_IEXEC)

    devices_mod.invalidate_cache()
    await devices_mod.probe_devices(str(recorder), rpc_endpoints=["127.0.0.1:50052"])
    args = seen.read_text().split()
    assert args.index("--rpc") < args.index("--list-devices")


def test_rpc_endpoint_is_emitted_for_a_raw_ot_placement(monkeypatch):
    """`-ot exps=RPC0` in extra_flags names a remote device just as much as a
    pin does. Without `--rpc`, RPC0 resolves to nothing and llama.cpp places
    the tensors on a local device without a word of complaint."""
    from lld import rpc_server
    from lld.argv import to_argv
    from lld.settings import LlamaServerConfig

    monkeypatch.setattr(rpc_server, "rpc_flag_value", lambda: "127.0.0.1:50052")
    monkeypatch.setattr("lld.argv.rpc_flag_value", lambda: "127.0.0.1:50052")

    cfg = LlamaServerConfig(name="t", model_path="/x.gguf",
                            extra_flags=["-ot", "exps=RPC0"])
    argv = to_argv(cfg, "llama-server")
    assert "--rpc" in argv
    # ...and before the flags that reference the device.
    assert argv.index("--rpc") < argv.index("-ot")


def test_no_rpc_flag_when_nothing_names_a_remote_device(monkeypatch):
    from lld.argv import to_argv
    from lld.settings import LlamaServerConfig

    monkeypatch.setattr("lld.argv.rpc_flag_value", lambda: "127.0.0.1:50052")
    cfg = LlamaServerConfig(name="t", model_path="/x.gguf",
                            devices=["CUDA0"], extra_flags=["-ot", "exps=CPU"])
    assert "--rpc" not in to_argv(cfg, "llama-server")
