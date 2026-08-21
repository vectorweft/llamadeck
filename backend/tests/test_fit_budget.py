"""The VRAM a plan is budgeted against, per pinned card.

The box behind these numbers is the mixed workstation: an RTX 5090 on CUDA0
(also listed as Vulkan2), a Radeon AI PRO R9700 on Vulkan1, and a 9950X's
integrated graphics on Vulkan0. What matters here is the asymmetry between the
two cards — nvidia-smi reports VRAM per process, amdgpu reports none — because
the first attempt at handling it counted nothing reclaimable outside NVIDIA
and thereby declared a busy Radeon permanently full.
"""
from __future__ import annotations

from lld import fit_budget
from lld.devices import parse_list_devices
from lld.settings import LlamaServerConfig
from lld.vram import GpuInfo, GpuProcess

MIXED_BOX = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32607 MiB, 371 MiB free)
  Vulkan0: AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO) (48515 MiB, 45933 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 583 MiB free)
  Vulkan2: NVIDIA GeForce RTX 5090 (32607 MiB, 371 MiB free)
"""
CPU_NAME = "AMD Ryzen 9 9950X 16-Core Processor"


def _patch_devices(monkeypatch, text: str = MIXED_BOX):
    devs = parse_list_devices(text, CPU_NAME)

    async def _probe(binary, *a, **kw):
        return devs

    monkeypatch.setattr(fit_budget, "probe_devices", _probe)
    monkeypatch.setattr(fit_budget, "load_settings",
                        lambda: type("S", (), {"llama_bin": "/bin/llama-server"})())
    return devs


def _gpus() -> list[GpuInfo]:
    """What the OS probe sees: the discrete cards only (the iGPU is dropped
    by offload_gpus, exactly as on the real box)."""
    return [
        GpuInfo(index=0, name="NVIDIA GeForce RTX 5090", total_mb=32607,
                used_mb=32236, free_mb=371, vendor="nvidia"),
        GpuInfo(index=1, name="AMD Radeon AI PRO R9700", total_mb=32624,
                used_mb=32041, free_mb=583, vendor="amd"),
    ]


def _cfg(devices: list[str] | None = None) -> LlamaServerConfig:
    return LlamaServerConfig(name="plan", model_path="/m.gguf", devices=devices or [])


def _running(name: str, devices: list[str], gpu_mb: int, pid: int = 100) -> dict:
    return {
        "name": name, "running": True, "pid": pid,
        "config": {"name": name, "model_path": f"/{name}.gguf", "devices": devices},
        "vram_estimate": {"gpu_mb": gpu_mb, "ram_mb": 0},
    }


# ---- the pinned Radeon ------------------------------------------------------

async def test_busy_radeon_counts_our_own_model_as_reclaimable(monkeypatch):
    """The reported bug: a 27 GB preset pinned to the R9700 while our own
    27 GB model sits on it came back "too big — download a smaller version",
    because no per-PID reading exists on amdgpu and nothing was counted as
    stoppable. The card is one stop away from empty, and must say so."""
    _patch_devices(monkeypatch)
    statuses = {"qwen-r9700": _running("qwen-r9700", ["Vulkan1"], 32804)}
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=[], statuses=statuses
    )
    assert (b.total_mb, b.free_mb) == (32624, 583)   # the Radeon, not the 5090
    assert b.budget_mb == 32624                      # everything on it is ours
    assert b.unified is False


async def test_reclaim_never_exceeds_what_the_card_holds(monkeypatch):
    """The estimate is the *claim*; the card's own used bytes are the cap.
    A model that estimates 32 GB but has only 10 GB allocated must not have
    22 GB of someone else's memory promised on its behalf."""
    _patch_devices(monkeypatch)
    gpus = _gpus()
    gpus[1] = GpuInfo(index=1, name="AMD Radeon AI PRO R9700", total_mb=32624,
                      used_mb=10000, free_mb=22624, vendor="amd")
    devs = parse_list_devices(MIXED_BOX.replace("32624 MiB, 583 MiB free",
                                                "32624 MiB, 22624 MiB free"), CPU_NAME)

    async def _probe(binary, *a, **kw):
        return devs

    monkeypatch.setattr(fit_budget, "probe_devices", _probe)
    statuses = {"big": _running("big", ["Vulkan1"], 32804)}
    b = await fit_budget.plan_budget(_cfg(["Vulkan1"]), gpus=gpus, procs=[], statuses=statuses)
    assert b.free_mb == 22624
    assert b.budget_mb == 32624  # 22624 free + 10000 used, not 22624 + 32804


async def test_idle_radeon_invents_nothing(monkeypatch):
    """Nothing of ours running there: the budget is what is free, full stop."""
    _patch_devices(monkeypatch)
    b = await fit_budget.plan_budget(_cfg(["Vulkan1"]), gpus=_gpus(), procs=[], statuses={})
    assert (b.free_mb, b.budget_mb) == (583, 583)


async def test_model_on_the_other_card_is_not_reclaimable(monkeypatch):
    """Stopping the 5090's model frees nothing on the Radeon — the whole
    reason per-card attribution exists."""
    _patch_devices(monkeypatch)
    statuses = {"vision-5090": _running("vision-5090", ["CUDA0"], 30000)}
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=[], statuses=statuses
    )
    assert b.budget_mb == 583


async def test_stopped_preset_does_not_count(monkeypatch):
    _patch_devices(monkeypatch)
    row = _running("idle", ["Vulkan1"], 20000)
    row["running"] = False
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=[], statuses={"idle": row}
    )
    assert b.budget_mb == 583


async def test_switch_frees_the_outgoing_preset_on_the_radeon(monkeypatch):
    """A switch stops the model on the card on purpose, so its VRAM is free
    rather than merely reclaimable — otherwise the swap is blocked as busy."""
    _patch_devices(monkeypatch)
    statuses = {"qwen-r9700": _running("qwen-r9700", ["Vulkan1"], 32804)}
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=[], statuses=statuses,
        freeing=["qwen-r9700"],
    )
    assert b.free_mb == 32624
    assert b.budget_mb == 32624


async def test_backend_alias_is_the_same_card(monkeypatch):
    """A preset pinned to Vulkan2 and a plan pinned to CUDA0 are competing for
    one 5090. Compared as raw ids they look unrelated."""
    _patch_devices(monkeypatch)
    statuses = {"twin": _running("twin", ["Vulkan2"], 30000)}
    # No procs at all → no per-PID reading to fall back on, so the pin-based
    # attribution has to see through the alias.
    b = await fit_budget.plan_budget(_cfg(["CUDA0"]), gpus=_gpus(), procs=[], statuses=statuses)
    assert b.budget_mb == 371 + 30000


# ---- NVIDIA keeps its exact, per-process reading ---------------------------

async def test_nvidia_pin_uses_per_process_vram(monkeypatch):
    """nvidia-smi knows exactly which process holds what; the estimate-based
    path is a fallback for cards that have no such reading, not a replacement."""
    _patch_devices(monkeypatch)
    procs = [
        GpuProcess(pid=1, process_name="llama-server", used_mb=20000),
        GpuProcess(pid=2, process_name="Xorg", used_mb=12236),
    ]
    b = await fit_budget.plan_budget(_cfg(["CUDA0"]), gpus=_gpus(), procs=procs, statuses={})
    # The compositor's 12 GB never comes back; only the llama-server's does.
    assert b.budget_mb == 371 + 20000


async def test_unresolvable_pin_falls_back_to_the_whole_machine(monkeypatch):
    """Ids this binary does not expose mean "no opinion" — planning against
    one card's memory on the strength of a stale pin would be worse."""
    _patch_devices(monkeypatch)
    b = await fit_budget.plan_budget(_cfg(["ROCm7"]), gpus=_gpus(), procs=[], statuses={})
    assert b.total_mb == 32607 + 32624


# ---- a card llama.cpp could not read --------------------------------------

ZERO_CUDA = """Available devices:
  CUDA0: NVIDIA GeForce RTX 5090 (0 MiB, 0 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32624 MiB, 583 MiB free)
"""


async def test_zero_row_is_repaired_from_the_os_probe(monkeypatch):
    """A full card makes cudaMemGetInfo fail, and llama.cpp then prints
    "(0 MiB, 0 MiB free)". Believing it means gpu_total_mb == 0, which
    check_fit reads as "this machine has no GPU" — and a 30 GB preset pinned
    to the 5090 comes back "fits in system RAM"."""
    _patch_devices(monkeypatch, ZERO_CUDA)
    b = await fit_budget.plan_budget(_cfg(["CUDA0"]), gpus=_gpus(), procs=[], statuses={})
    assert (b.total_mb, b.free_mb) == (32607, 371)


async def test_zero_row_stays_zero_when_the_card_cannot_be_identified(monkeypatch):
    """No matching row in the OS probe: report what was read rather than
    borrow another card's numbers."""
    _patch_devices(monkeypatch, ZERO_CUDA)
    others = [GpuInfo(index=0, name="NVIDIA GeForce RTX 4090", total_mb=24564,
                      used_mb=0, free_mb=24564, vendor="nvidia")]
    b = await fit_budget.plan_budget(_cfg(["CUDA0"]), gpus=others, procs=[], statuses={})
    assert (b.total_mb, b.free_mb) == (0, 0)


# ---- no pin ----------------------------------------------------------------

async def test_amd_only_box_without_a_pin_still_reclaims(monkeypatch):
    """Same blindness, no pin involved: on a box with no nvidia-smi at all the
    per-PID list is empty, and every running model used to look unstoppable."""
    _patch_devices(monkeypatch)
    gpus = [GpuInfo(index=0, name="AMD Radeon AI PRO R9700", total_mb=32624,
                    used_mb=32041, free_mb=583, vendor="amd")]
    statuses = {"solo": _running("solo", [], 30000)}
    b = await fit_budget.plan_budget(_cfg(), gpus=gpus, procs=[], statuses=statuses)
    assert b.budget_mb == 583 + 30000


# ---- per-card process attribution ------------------------------------------

def _proc(pid: int, used_mb: int, vendor: str, index: int, name: str = "llama-server"):
    return GpuProcess(pid=pid, process_name=name, used_mb=used_mb,
                      gpu_index=index, vendor=vendor)


async def test_radeon_pin_uses_the_amdgpu_process_reading(monkeypatch):
    """DRM fdinfo answers "who holds what" on the Radeon, so the budget comes
    from the same kind of exact per-process number NVIDIA has always had."""
    _patch_devices(monkeypatch)
    statuses = {"qwen-r9700": _running("qwen-r9700", ["Vulkan1"], 32804, pid=131078)}
    procs = [
        _proc(131078, 26675, "amd", 1),
        # The same server also opens a CUDA context on the 5090. Counting it
        # toward the Radeon's budget would be free memory that isn't there.
        _proc(131078, 498, "nvidia", 0),
    ]
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=procs, statuses=statuses
    )
    assert b.budget_mb == 583 + 26675


async def test_a_model_on_the_5090_is_not_reclaimable_for_the_radeon(monkeypatch):
    """The per-process path has to respect the card boundary too."""
    _patch_devices(monkeypatch)
    statuses = {"vision-5090": _running("vision-5090", ["CUDA0"], 30000, pid=169797)}
    procs = [_proc(169797, 31264, "nvidia", 0)]
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=procs, statuses=statuses
    )
    assert b.budget_mb == 583


async def test_estimates_take_over_when_the_reading_misses_our_server(monkeypatch):
    """An old kernel has no fdinfo and nvidia-smi never sees a Radeon: our
    server is running on the card yet appears in no process row. That is when
    the estimate-based fallback has to step in."""
    _patch_devices(monkeypatch)
    statuses = {"qwen-r9700": _running("qwen-r9700", ["Vulkan1"], 32804, pid=131078)}
    b = await fit_budget.plan_budget(
        _cfg(["Vulkan1"]), gpus=_gpus(), procs=[_proc(4242, 900, "nvidia", 0, "Xorg")],
        statuses=statuses,
    )
    assert b.budget_mb == 32624
