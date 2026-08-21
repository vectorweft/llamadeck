"""llama.cpp device discovery — the list a preset's `-dev` selection is built from.

The shapes here are taken verbatim from a CUDA+Vulkan build on a box with an
RTX 5090, a Radeon AI PRO R9700 and a Ryzen 9950X's integrated graphics, which
is the configuration that motivated the feature: four rows for two usable GPUs.
"""
from __future__ import annotations

import asyncio

from lld.devices import (
    CPU_DEVICE_ID,
    LlamaDevice,
    cpu_device,
    device_budget_mb,
    is_cpu_pin,
    offload_targets,
    parse_list_devices,
    selectable_devices,
    unknown_device_ids,
)

MIXED_BOX = """WARNING: radv is not a conformant Vulkan implementation, testing use only.
Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
  Vulkan0: AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO) (47995 MiB, 45344 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 32566 MiB free)
  Vulkan2: NVIDIA GeForce RTX 5090 (32607 MiB, 31626 MiB free)
"""

CPU_NAME = "AMD Ryzen 9 9950X 16-Core Processor"


def _by_id(devices: list[LlamaDevice]) -> dict[str, LlamaDevice]:
    return {d.id: d for d in devices}


def test_parses_every_row_with_memory():
    devs = _by_id(parse_list_devices(MIXED_BOX, CPU_NAME))
    assert set(devs) == {"CUDA0", "Vulkan0", "Vulkan1", "Vulkan2"}
    assert (devs["CUDA0"].total_mb, devs["CUDA0"].free_mb) == (32149, 31626)
    assert devs["CUDA0"].backend == "CUDA"
    # The name carries its own parentheses; the memory group must still be the
    # one at the end of the line.
    assert devs["Vulkan1"].name == "AMD Radeon AI PRO R9700 (RADV GFX1201)"
    assert devs["Vulkan1"].total_mb == 32624


def test_one_card_seen_by_two_backends_collapses_onto_the_native_one():
    """A CUDA+Vulkan build lists the 5090 twice. Selecting both would be a
    mistake, and Vulkan is the slower path for an NVIDIA card."""
    devs = _by_id(parse_list_devices(MIXED_BOX, CPU_NAME))
    assert devs["CUDA0"].duplicate_of is None
    assert devs["Vulkan2"].duplicate_of == "CUDA0"
    assert devs["Vulkan2"].selectable is False


def test_integrated_gpu_is_flagged_by_its_cpu_name():
    """RADV names a desktop Ryzen's iGPU after the CPU, which is the only
    thing separating it from a discrete card in this output."""
    devs = _by_id(parse_list_devices(MIXED_BOX, CPU_NAME))
    assert devs["Vulkan0"].integrated is True
    assert devs["Vulkan0"].selectable is False
    assert devs["Vulkan1"].integrated is False


def test_only_the_two_real_gpus_are_selectable():
    devs = parse_list_devices(MIXED_BOX, CPU_NAME)
    assert [d.id for d in selectable_devices(devs)] == ["CUDA0", "Vulkan1"]


def test_software_rasterizer_is_never_a_target():
    text = """Available devices:
  Vulkan0: NVIDIA GeForce RTX 5090 (32607 MiB, 31626 MiB free)
  Vulkan1: llvmpipe (LLVM 21.1.8, 256 bits) (48000 MiB, 47000 MiB free)
"""
    devs = _by_id(parse_list_devices(text, CPU_NAME))
    assert devs["Vulkan1"].software is True
    assert devs["Vulkan1"].selectable is False
    assert [d.id for d in selectable_devices(list(devs.values()))] == ["Vulkan0"]


def test_cuda_only_build_reports_just_its_own_device():
    """The point of asking the binary rather than the OS: a CUDA-only build
    cannot offload to the Radeon the kernel can see."""
    text = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)
"""
    devs = parse_list_devices(text, CPU_NAME)
    assert [d.id for d in selectable_devices(devs)] == ["CUDA0"]


def test_unparseable_output_yields_no_devices():
    assert parse_list_devices("ggml_cuda_init: failed\n", CPU_NAME) == []


def test_missing_cpu_name_leaves_integrated_undetected_rather_than_guessing():
    """Without the CPU name there is no signal, and inventing one would be
    worse than reporting the row as usable."""
    devs = _by_id(parse_list_devices(MIXED_BOX, ""))
    assert devs["Vulkan0"].integrated is False


# ── the CPU row ───────────────────────────────────────────────────────
# `--list-devices` prints GPUs only, so "run this in RAM while both cards are
# busy" — the reason a 1.5 GB model exists on this box at all — had no row to
# tick in the editor.

def test_cpu_row_uses_llama_cpps_own_spelling_for_do_not_offload():
    """The id has to be what `-dev` accepts, or a translation layer creeps in
    and the command box stops round-tripping."""
    cpu = cpu_device()
    assert cpu.id == CPU_DEVICE_ID == "none"
    assert cpu.backend == "CPU"
    # "none" is not a name anyone recognises as their processor.
    assert cpu.label == "CPU"
    assert cpu.to_dict()["label"] == "CPU"


def test_cpu_row_is_selectable_and_appended_after_the_gpus():
    targets = offload_targets(parse_list_devices(MIXED_BOX, CPU_NAME))
    assert [d.id for d in targets][-1] == CPU_DEVICE_ID
    assert targets[-1].selectable is True
    # …and it does not disturb what the probe said about the real cards.
    assert [d.id for d in targets[:-1]] == ["CUDA0", "Vulkan0", "Vulkan1", "Vulkan2"]


def test_real_devices_keep_their_id_as_the_label():
    for dev in parse_list_devices(MIXED_BOX, CPU_NAME):
        assert dev.to_dict()["label"] == dev.id


def test_a_cpu_pin_is_only_a_cpu_pin_on_its_own():
    assert is_cpu_pin(["none"]) is True
    assert is_cpu_pin([]) is False
    assert is_cpu_pin(None) is False
    # `-dev none,CUDA0` is not "half of each" — it is invalid, and treating it
    # as a CPU pin would silently drop the GPU the user asked for.
    assert is_cpu_pin(["none", "CUDA0"]) is False


def test_cpu_pin_is_never_reported_as_an_unknown_device(monkeypatch):
    """`none` comes from cpu_device(), never from the binary, so the existence
    check has to know about it — otherwise saving the preset is refused."""
    async def fake_probe(binary, *a, **kw):
        return parse_list_devices(MIXED_BOX, CPU_NAME)

    monkeypatch.setattr("lld.devices.probe_devices", fake_probe)
    assert asyncio.run(unknown_device_ids("llama-server", ["none"])) == []
    assert asyncio.run(unknown_device_ids("llama-server", ["Vulkan9"])) == ["Vulkan9"]


def test_cpu_pin_reports_no_vram_budget_rather_than_an_empty_one(monkeypatch):
    """(0, 0) would read as "this card is full" to the broker's pre-spawn check
    and reject a model that needs no VRAM at all."""
    async def fake_probe(binary, *a, **kw):
        return parse_list_devices(MIXED_BOX, CPU_NAME)

    monkeypatch.setattr("lld.devices.probe_devices", fake_probe)
    assert asyncio.run(device_budget_mb("llama-server", ["none"])) is None
    assert asyncio.run(device_budget_mb("llama-server", ["Vulkan1"])) == (32624, 32566)


# ── a full card reporting no card ─────────────────────────────────────

FULL_5090 = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (0 MiB, 0 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 5953 MiB free)
  Vulkan2: NVIDIA GeForce RTX 5090 (32607 MiB, 398 MiB free)
"""


def test_total_memory_is_recovered_from_the_alias_when_cuda_cannot_read_it():
    """Under memory pressure cudaMemGetInfo fails and llama.cpp prints 0/0.
    A 32 GB card shown as "0 GB" is wrong in the picker and, worse, budgets as
    empty — while the Vulkan view of the same card could still read it."""
    devs = _by_id(parse_list_devices(FULL_5090, CPU_NAME))
    assert devs["CUDA0"].total_mb == 32607
    # Free is live telemetry read at a different moment on a card that really
    # is full. Healing it would invent headroom that does not exist.
    assert devs["CUDA0"].free_mb == 0


def test_a_card_with_no_alias_is_left_exactly_as_reported():
    devs = _by_id(parse_list_devices(
        "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5090 (0 MiB, 0 MiB free)\n",
        CPU_NAME,
    ))
    assert (devs["CUDA0"].total_mb, devs["CUDA0"].free_mb) == (0, 0)


# --------------------------------------------------------------------------
# Machines whose only accelerator is integrated: every Mac, every APU-only box
# --------------------------------------------------------------------------

MAC = """Available devices:
  Metal0: Apple M3 Max (98304 MiB, 96000 MiB free)
"""

STRIX_HALO = """Available devices:
  Vulkan0: AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics (RADV GFX1151) (98304 MiB, 96000 MiB free)
"""


def test_a_macs_metal_device_is_pinnable():
    """llama.cpp names the Metal device after the chip, and sysctl reports the
    same string as the CPU — so the integrated heuristic fires on the only GPU
    a Mac has. Greying it out leaves the device picker unusable there."""
    devs = parse_list_devices(MAC, "Apple M3 Max")
    assert len(devs) == 1
    metal = devs[0]
    assert metal.integrated is True      # it is: Metal shares the memory pool
    assert metal.sole_accelerator is True
    assert metal.selectable is True
    assert [d.id for d in selectable_devices(devs)] == ["Metal0"]


def test_an_apu_only_box_can_pin_its_igpu():
    devs = parse_list_devices(STRIX_HALO, "AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics")
    assert devs[0].integrated is True
    assert devs[0].selectable is True


def test_an_igpu_beside_a_real_card_stays_unpickable():
    """The exclusion still has to work where it was aimed: a desktop iGPU
    driving the display next to a card that can actually hold a model."""
    devs = _by_id(parse_list_devices(MIXED_BOX, CPU_NAME))
    igpu = next(d for d in devs.values() if d.integrated)
    assert igpu.sole_accelerator is False
    assert igpu.selectable is False


def test_a_software_only_list_does_not_promote_llvmpipe():
    """No accelerator at all is not the same as one integrated accelerator."""
    devs = parse_list_devices(
        "Available devices:\n  Vulkan0: llvmpipe (LLVM 19, 256 bits) (16000 MiB, 15000 MiB free)\n",
        CPU_NAME,
    )
    assert devs[0].software is True
    assert devs[0].selectable is False
