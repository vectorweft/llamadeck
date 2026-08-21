"""The offload targets llama.cpp itself reports (`llama-server --list-devices`).

This is deliberately a *second* source of GPU truth next to `vram.py`. The two
answer different questions and must not be conflated:

  * `vram.py` asks the operating system what hardware exists. It is what the
    dashboard's telemetry is built on, and it works with no llama.cpp at all.
  * this module asks one specific llama-server binary what it can actually
    offload to. That depends on which backends the binary was compiled with —
    a CUDA-only build cannot see a Radeon no matter what the kernel reports.

Only the ids from here (`CUDA0`, `Vulkan1`) may be passed to `-dev`/`-ot`, so
a preset's device selection is stored and validated against this list.

One row is not llama.cpp's: `--list-devices` prints GPUs only, so "run this on
the CPU" — the whole point of a small model while both cards are busy — had no
row to tick. `cpu_device()` synthesises it, with llama.cpp's own spelling for
"do not offload" (`-dev none`) as its id so it round-trips through the command
box unchanged.
"""
from __future__ import annotations

import re
import time

from .procutil import run_capture
from dataclasses import dataclass, field
from pathlib import Path

# "  CUDA0: NVIDIA GeForce RTX 5090 (32149 MiB, 31626 MiB free)"
# The name itself may contain parentheses ("… (RADV GFX1201)"), so the memory
# group is anchored to the end of the line rather than to the first "(".
_DEVICE_RE = re.compile(
    r"^\s*(?P<id>\S+?):\s+(?P<name>.+?)\s+"
    r"\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\)\s*$"
)

# Which backend to prefer when one physical GPU is exposed by several. A build
# with both CUDA and Vulkan lists an RTX 5090 twice; the vendor-native backend
# is the faster one, so it wins and the other alias is marked a duplicate.
_BACKEND_RANK = {
    "CUDA": 0, "HIP": 1, "ROCm": 1, "Metal": 1, "Vulkan": 2,
    "SYCL": 3, "OpenCL": 4, "RPC": 5, "BLAS": 8, "CPU": 9,
}

# llama-server's own spelling for "offload nothing": `-dev none`. Used as the
# CPU row's device id so a preset needs no translation layer — what is stored
# is what is passed, and parsing the command line back yields the same id.
CPU_DEVICE_ID = "none"

# Names CPU/software renderers go by. Offloading to these is never what the
# user meant — llvmpipe in particular is a software rasterizer that would
# "work" and be catastrophically slow.
_SOFTWARE_MARKERS = ("llvmpipe", "swiftshader", "softpipe", "lavapipe")


@dataclass
class LlamaDevice:
    id: str
    name: str
    total_mb: int
    free_mb: int
    backend: str
    # What the picker shows in the id column. Defaults to the id, which is
    # right for every real device; the CPU row overrides it because its id is
    # llama.cpp's "none" and nobody would recognise that as their processor.
    label: str = ""
    # An APU's iGPU. Present in the list, never a sensible offload target.
    integrated: bool = False
    # A software rasterizer, not real hardware.
    software: bool = False
    # Set when this row is the same physical GPU as another, lower-ranked
    # backend's row. Holds the id we prefer.
    duplicate_of: str | None = None
    # RPC rows only: the host:port behind them. llama-server prints the
    # endpoint where other backends print a GPU name, so this is all the
    # identity an RPC device has.
    rpc_endpoint: str | None = None
    # RPC rows only, and a *suspicion* rather than a fact: a local device with
    # exactly this much memory exists, so the two are probably the same card
    # reached two ways. It cannot be proven — the RPC protocol reports no
    # hardware identity — so this warns instead of blocking, unlike
    # `duplicate_of`.
    may_alias: str | None = None

    # Set by parse_list_devices when this integrated GPU is the only
    # accelerator the binary can see. See `selectable`.
    sole_accelerator: bool = False

    @property
    def selectable(self) -> bool:
        """Whether a preset may be pinned to this device.

        Integrated GPUs are normally excluded: on a desktop the iGPU is
        driving the display out of a token carve-out of system RAM, and
        offering it next to a real card only invites an OOM.

        That reasoning inverts when it is the ONLY accelerator, which is the
        normal case on two whole classes of machine this app is meant to run
        on. llama.cpp names a Mac's Metal device after the chip ("Apple M3
        Max") and RADV names an APU's iGPU after the CPU, so both match the
        integrated heuristic — and excluding them left a Mac and a Strix Halo
        box with the device picker greyed out on the single GPU they have.
        `vram.offload_gpus` already draws the same distinction for the OS-level
        probe; this is the device-list half of it.
        """
        if self.software or self.duplicate_of:
            return False
        return self.sole_accelerator or not self.integrated

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "name": self.name,
            "total_mb": self.total_mb,
            "free_mb": self.free_mb,
            "backend": self.backend,
            "integrated": self.integrated,
            "software": self.software,
            "duplicate_of": self.duplicate_of,
            "rpc_endpoint": self.rpc_endpoint,
            "may_alias": self.may_alias,
            "sole_accelerator": self.sole_accelerator,
            "selectable": self.selectable,
        }


def _backend_of(device_id: str) -> str:
    m = re.match(r"^([A-Za-z]+)", device_id)
    return m.group(1) if m else device_id


def _normalized_name(name: str) -> str:
    """Key for deciding two rows are the same card.

    Vulkan appends its driver in parentheses ("… (RADV GFX1201)") where CUDA
    does not, so that suffix is dropped before comparing.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip().lower()


def parse_list_devices(text: str, cpu_name: str = "") -> list[LlamaDevice]:
    """Parse `llama-server --list-devices` output.

    `cpu_name` is how an integrated GPU is spotted: RADV names a desktop
    Ryzen's iGPU after the CPU itself ("AMD Ryzen 9 9950X 16-Core Processor
    (RADV RAPHAEL_MENDOCINO)"), which is otherwise indistinguishable from a
    discrete card in this output.
    """
    devices: list[LlamaDevice] = []
    cpu_key = cpu_name.strip().lower()
    for line in text.splitlines():
        if line.strip().lower().startswith("available devices"):
            continue
        m = _DEVICE_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        lowered = name.lower()
        dev = LlamaDevice(
            id=m.group("id"),
            name=name,
            total_mb=int(m.group("total")),
            free_mb=int(m.group("free")),
            backend=_backend_of(m.group("id")),
            software=any(mark in lowered for mark in _SOFTWARE_MARKERS),
            integrated=bool(cpu_key) and cpu_key in lowered,
        )
        devices.append(dev)

    # Collapse aliases of one physical GPU onto the best-ranked backend.
    best: dict[str, LlamaDevice] = {}
    for dev in devices:
        if dev.software:
            continue
        key = _normalized_name(dev.name)
        incumbent = best.get(key)
        rank = _BACKEND_RANK.get(dev.backend, 7)
        if incumbent is None or rank < _BACKEND_RANK.get(incumbent.backend, 7):
            best[key] = dev
    for dev in devices:
        if dev.software:
            continue
        winner = best.get(_normalized_name(dev.name))
        if winner is not None and winner.id != dev.id:
            dev.duplicate_of = winner.id

    # A card that is full can report no card at all. Under memory pressure
    # cudaMemGetInfo fails and llama.cpp prints "(0 MiB, 0 MiB free)" — seen
    # here with the 5090 loaded up:
    #
    #   CUDA0:   NVIDIA GeForce RTX 5090 (0 MiB, 0 MiB free)
    #   Vulkan2: NVIDIA GeForce RTX 5090 (32607 MiB, 398 MiB free)
    #
    # Total memory is a property of the hardware and cannot honestly be zero,
    # so take it from the alias that could still read it: otherwise the picker
    # offers a "0 GB" card and `device_budget_mb` plans a 32 GB card as empty.
    # `free_mb` is deliberately NOT healed — 0 free on a full card is the
    # truth, and the two rows are reading the same memory at different moments.
    for dev in devices:
        if dev.total_mb > 0 or dev.software:
            continue
        alias = next(
            (
                o for o in devices
                if o is not dev and o.total_mb > 0
                and (o.duplicate_of == dev.id or dev.duplicate_of == o.id)
            ),
            None,
        )
        if alias is not None:
            dev.total_mb = alias.total_mb

    # RPC rows carry an endpoint where the others carry a GPU name, so the
    # name-based collapse above cannot see that `RPC0` and `Vulkan1` may be the
    # same physical card. Matching total memory is the only hint available;
    # flag it so the editor can warn, but never auto-exclude — the endpoint
    # might genuinely be another machine holding an identical card.
    local_by_size = {
        d.total_mb: d.id
        for d in devices
        if d.backend != "RPC" and not d.software and not d.integrated
    }
    for dev in devices:
        if dev.backend != "RPC":
            continue
        dev.rpc_endpoint = dev.name
        twin = local_by_size.get(dev.total_mb)
        if twin:
            dev.may_alias = twin

    # A machine whose only accelerator is integrated (any Mac, any APU-only
    # box such as Strix Halo) must still be able to pin to it.
    real = [d for d in devices if not d.software and not d.duplicate_of]
    if real and all(d.integrated for d in real):
        for dev in real:
            dev.sole_accelerator = True
    return devices


def cpu_device() -> LlamaDevice:
    """The synthetic "run it on the CPU" row.

    llama-server's `--list-devices` prints GPUs only (it filters on
    GGML_BACKEND_DEVICE_TYPE_GPU), so the picker built from that list could
    offer every card on the box and no way to say "neither". On a machine
    where both GPUs are already full, a 1.5 GB model belongs in RAM — and the
    editor had no row for it.

    Memory is system RAM rather than VRAM, which is the honest budget for this
    target: `free_mb` is `available`, not `free`, because reclaimable page
    cache is memory a model may still have.
    """
    from .accel import platform_info

    total_mb = free_mb = 0
    try:
        import psutil

        mem = psutil.virtual_memory()
        mb = 1024 * 1024
        total_mb = int(mem.total // mb)
        free_mb = int(mem.available // mb)
    except Exception:  # noqa: BLE001 - a missing reading must not hide the row
        pass
    name = ""
    try:
        name = platform_info().cpu_name or ""
    except Exception:  # noqa: BLE001
        pass
    return LlamaDevice(
        id=CPU_DEVICE_ID,
        label="CPU",
        name=name or "CPU (system RAM)",
        total_mb=total_mb,
        free_mb=free_mb,
        backend="CPU",
    )


def offload_targets(devices: list[LlamaDevice]) -> list[LlamaDevice]:
    """`devices` plus the CPU row — everything a preset may be pinned to.

    Kept separate from `probe_devices` so that function stays exactly what its
    docstring promises: what one llama-server binary reports. Callers that ask
    "what can the user pick?" want this; callers that ask "does this build see
    a Radeon?" want the raw probe.
    """
    return [*devices, cpu_device()]


def is_cpu_pin(device_ids: list[str] | None) -> bool:
    """Whether a selection means "do not offload at all"."""
    return [d for d in (device_ids or []) if d] == [CPU_DEVICE_ID]


@dataclass
class _CacheEntry:
    key: tuple[str, float, tuple[str, ...]]
    at: float
    devices: list[LlamaDevice] = field(default_factory=list)


# Identity (ids, names, totals) only changes when the binary does, but
# `free_mb` is live telemetry and the planner budgets against it. A short TTL
# keeps it honest without paying ~0.25 s per call while the editor polls.
_CACHE_TTL_S = 5.0
_cache: _CacheEntry | None = None


def invalidate_cache() -> None:
    """Drop the memoized list — call after a rebuild swaps the binary."""
    global _cache
    _cache = None


async def probe_devices(
    binary: str,
    timeout: float = 20.0,
    rpc_endpoints: list[str] | None = None,
) -> list[LlamaDevice]:
    """Devices this binary can offload to, or [] when it cannot be queried.

    `rpc_endpoints` defaults to whatever RPC servers are running, because a
    remote device only exists in the list when `--rpc` names its endpoint — and
    callers should not each have to remember that. Pass an explicit list (or
    `[]`) to override.

    Memoized on (path, mtime, endpoints) with a short TTL: identity changes
    only when the binary or the endpoint set does, but `free_mb` is live
    telemetry the planner budgets against.
    """
    global _cache
    if not binary:
        return []
    if rpc_endpoints is None:
        from .rpc_server import get_rpc_manager  # local: rpc_server imports settings

        try:
            rpc_endpoints = get_rpc_manager().running_endpoints()
        except Exception:  # noqa: BLE001 - telemetry must never break a probe
            rpc_endpoints = []
    endpoints = tuple(sorted(e for e in (rpc_endpoints or []) if e))
    try:
        mtime = Path(binary).stat().st_mtime
    except OSError:
        return []
    key = (binary, mtime, endpoints)
    now = time.monotonic()
    if _cache is not None and _cache.key == key and now - _cache.at < _CACHE_TTL_S:
        return _cache.devices
    # Order matters: llama-server acts on --list-devices as it parses it and
    # exits, so anything after it — including --rpc — is never read and the
    # remote devices silently go missing.
    argv = [binary]
    if endpoints:
        argv += ["--rpc", ",".join(endpoints)]
    argv += ["--list-devices"]
    res = await run_capture(argv, timeout=timeout)
    if not res.ok:
        return []
    from .accel import platform_info

    devices = parse_list_devices(res.text, platform_info().cpu_name)
    _cache = _CacheEntry(key=key, at=now, devices=devices)
    return devices


def cached_devices() -> list[LlamaDevice]:
    """The last probed device list, without probing. Empty when never probed.

    For sync callers that need vendor information but must not spawn a
    subprocess (`estimate_vram` runs on every keystroke in the editor). Every
    path that matters probes shortly beforehand, so this is normally warm; an
    empty answer means "unknown", never "no devices".
    """
    return list(_cache.devices) if _cache is not None else []


def pin_is_nvidia(device_ids: list[str] | None) -> bool | None:
    """Whether a device pin resolves to NVIDIA. None when it cannot be told.

    Ids alone are not enough — a CUDA+Vulkan build lists the 5090 as both
    CUDA0 and Vulkan2 — so the cached device list decides. Falling back to the
    id prefix covers the case where nothing has probed yet: `CUDA*` is
    certainly NVIDIA and `ROCm*`/`HIP*` certainly is not, while a bare
    `Vulkan*` stays unknown.
    """
    ids = [d for d in (device_ids or []) if d]
    if not ids:
        return None
    by_id = {d.id: d for d in cached_devices()}
    picked = [by_id[i] for i in ids if i in by_id]
    if picked:
        return pin_holds_nvidia(picked)
    if any(i.startswith("CUDA") for i in ids):
        return True
    if all(i.startswith(("ROCm", "HIP")) for i in ids):
        return False
    return None


def device_vendor(dev: LlamaDevice) -> str:
    """"nvidia" | "amd" | "apple" | "" — the silicon behind one device row.

    Deliberately the same vocabulary `vram.GpuInfo.vendor` uses, because the
    two have to be compared: a measurement is taken per card by one of those
    probes and looked up again from a device pin.
    """
    name = dev.name.lower()
    if dev.backend == "CUDA" or "nvidia" in name or "geforce" in name:
        return "nvidia"
    if dev.backend in ("HIP", "ROCm") or any(
        m in name for m in ("amd", "radeon", "gfx", "ryzen")
    ):
        return "amd"
    if dev.backend == "Metal" or "apple" in name:
        return "apple"
    return ""


def pin_vendor(device_ids: list[str] | None) -> str:
    """The vendor a device pin lands on, or "" when it cannot be told.

    "" covers three different situations that all deserve the same caution: no
    pin at all, ids nothing has probed yet, and a pin spanning two vendors.
    Callers must not assume a vendor from it — an offset measured on CUDA is
    meaningless for a Vulkan allocator, and vice versa.
    """
    ids = [d for d in (device_ids or []) if d]
    if not ids:
        return ""
    by_id = {d.id: d for d in cached_devices()}
    picked = [by_id[i] for i in ids if i in by_id]
    if picked:
        vendors = {device_vendor(d) for d in picked}
        return vendors.pop() if len(vendors) == 1 else ""
    if all(i.startswith("CUDA") for i in ids):
        return "nvidia"
    if all(i.startswith(("ROCm", "HIP")) for i in ids):
        return "amd"
    return ""


def pin_holds_nvidia(devices: list[LlamaDevice]) -> bool:
    """Whether a device pin lands on an NVIDIA card.

    Per-process VRAM is an NVIDIA-only reading (`nvidia-smi`; amdgpu exposes no
    per-PID breakdown without root). So "stopping that model frees N GB" only
    holds when the plan and those processes live on the same silicon —
    otherwise an R9700-pinned preset's budget is inflated by bytes held on the
    5090, and the fit panel promises a fit that stopping anything cannot buy.

    The backend name alone is not enough: a CUDA+Vulkan build lists the 5090
    as both CUDA0 and Vulkan2, so the card's own name has the final say.
    """
    return any(d.backend == "CUDA" or "nvidia" in d.name.lower() for d in devices)


def selectable_devices(devices: list[LlamaDevice]) -> list[LlamaDevice]:
    """The rows a preset may actually be pinned to."""
    return [d for d in devices if d.selectable]


async def unknown_device_ids(binary: str, wanted: list[str]) -> list[str]:
    """Selected ids this binary does not expose, or [] when there is no problem.

    An empty result also covers "cannot tell": a binary that fails to answer
    yields no device list, and that is not evidence the selection is wrong.
    Shared by the supervisor's start check and the preset save validation so
    the two can never disagree about what is valid.
    """
    wanted = [d for d in (wanted or []) if d]
    if not wanted:
        return []
    devices = await probe_devices(binary)
    if not devices:
        return []
    # `none` comes from cpu_device(), not from the binary, so it will never be
    # in the probe — and it is valid for every build ever made.
    known = {d.id for d in devices} | {CPU_DEVICE_ID}
    return [d for d in wanted if d not in known]


async def device_budget_mb(binary: str, wanted: list[str]) -> tuple[int, int] | None:
    """(total_mb, free_mb) summed over the selected devices.

    None means "no opinion" — no selection, no queryable binary, or ids that
    do not resolve. Callers fall back to the whole-machine GPU aggregate, so a
    preset without a pin plans exactly as it did before this existed.

    Using llama.cpp's own numbers avoids having to map its device ids onto the
    OS-level probe in vram.py: the two disagree slightly on totals (the driver
    reserves some), and matching them by name is brittle across backends.
    """
    wanted = [d for d in (wanted or []) if d]
    if not wanted:
        return None
    # A CPU pin holds no VRAM, and answering (0, 0) would read as "this card
    # is full" to the broker's pre-spawn check and reject a model that needs
    # no VRAM at all. "No opinion" is the truthful answer.
    if is_cpu_pin(wanted):
        return None
    devices = await probe_devices(binary)
    if not devices:
        return None
    by_id = {d.id: d for d in devices}
    picked = [by_id[d] for d in wanted if d in by_id]
    if len(picked) != len(wanted):
        return None
    return sum(d.total_mb for d in picked), sum(d.free_mb for d in picked)
