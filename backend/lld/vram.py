"""GPU memory probe — NVIDIA, AMD and Apple Silicon.

Three sources, tried in order, all optional and all dependency-free (no
pynvml, no pyamdsmi — an empty list simply means "no GPU telemetry here"):

  1. `nvidia-smi`                       — NVIDIA, discrete VRAM
  2. `/sys/class/drm/card*/device/…`    — AMD (amdgpu), incl. Strix Halo APUs
  3. `sysctl` + system RAM              — Apple Silicon unified memory

Callers treat an empty result as "plan against system RAM instead" (see
fit_check). Everything below is best-effort: a probe that errors returns
nothing rather than raising.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .procutil import run_capture

log = logging.getLogger(__name__)

_CARD_RE = re.compile(r"^card\d+$")

#: The last reason nvidia-smi gave for not working, so a broken driver is
#: reported once rather than on every poll of the VRAM panel.
_nv_failure: str | None = None

#: How long any single `nvidia-smi` call gets. It is a status query against a
#: healthy driver — but on a box whose kernel module and userspace libraries
#: have drifted apart (a driver package upgraded without a reboot) it blocks
#: for tens of seconds instead of failing, which is what made startup here take
#: 41s. Cap it, kill it, and report "no NVIDIA GPU" rather than wait.
_NV_TIMEOUT_S = 2.0


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    # "nvidia" | "amd" | "apple"
    vendor: str = "nvidia"
    # True when this memory is carved out of / shared with system RAM.
    unified: bool = False
    # True for an APU's iGPU. Such a part reports a token VRAM carve-out plus
    # the whole GTT aperture, so budgeting a model against it would promise
    # tens of GB that are really system RAM. See `offload_gpus`.
    integrated: bool = False
    # AMD only: GTT is the pool the GPU can borrow from system RAM on top of
    # its VRAM carve-out. 0 when unknown/not applicable.
    gtt_total_mb: int = 0
    gtt_used_mb: int = 0
    # Live sensors. Every one is optional and independently None: an iGPU has
    # no fan, Metal exposes no sensors at all, and a driver that answers for
    # temperature may still refuse the clock. Callers render "—", never 0 —
    # a missing reading and a card sitting at 0 % are not the same thing.
    util_percent: float | None = None
    temp_c: float | None = None
    # The hotspot ("junction") is what actually throttles the card; the edge
    # sensor reads 20 C cooler and is the one people quote by mistake.
    hotspot_c: float | None = None
    mem_temp_c: float | None = None
    fan_percent: float | None = None
    fan_rpm: int | None = None
    power_w: float | None = None
    clock_mhz: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "vendor": self.vendor,
            "unified": self.unified,
            "integrated": self.integrated,
            "gtt_total_mb": self.gtt_total_mb,
            "gtt_used_mb": self.gtt_used_mb,
            "util_percent": self.util_percent,
            "temp_c": self.temp_c,
            "hotspot_c": self.hotspot_c,
            "mem_temp_c": self.mem_temp_c,
            "fan_percent": self.fan_percent,
            "fan_rpm": self.fan_rpm,
            "power_w": self.power_w,
            "clock_mhz": self.clock_mhz,
        }


@dataclass
class GpuProcess:
    pid: int
    process_name: str
    used_mb: int
    gpu_index: int | None = None
    # Which probe produced this row. Indices are per-vendor (nvidia-smi counts
    # from 0, amdgpu's come from the DRM card number), so on a mixed box the
    # index alone does not identify a card — the pair does.
    vendor: str = "nvidia"

    @property
    def card_key(self) -> tuple[str, int | None]:
        return (self.vendor, self.gpu_index)


@dataclass
class AmdCard:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    gtt_total_mb: int = 0
    gtt_used_mb: int = 0
    # Maker of the dedicated memory chips ("samsung", "hynix"). Empty on an
    # APU, which has no dedicated memory to name — that absence is what
    # `amd_card_is_integrated` keys on.
    vram_vendor: str = ""
    pci_id: str = ""
    # "0000:03:00.0" — the PCI address, not the vendor:device id. DRM fdinfo
    # names the card this way and nothing else in sysfs matches it.
    pci_addr: str = ""
    # Live hwmon readings; see `amd_sensors`. Keys match GpuInfo's sensor
    # fields, and an absent sensor is simply not in the dict.
    sensors: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


async def probe_processes() -> list[GpuProcess]:
    """Per-process VRAM usage across every vendor that can report it.

    NVIDIA answers through nvidia-smi, AMD through DRM fdinfo; Metal has no
    per-process reading at all and contributes nothing. Rows carry `vendor`
    alongside `gpu_index` because the two vendors number their cards
    independently — on a mixed box the index alone names two different cards.

    Discrete cards only. See `probe_amd_processes` for why an APU's render
    node would otherwise fill this list with the desktop.

    `gpu_index` is resolved from the process's PCI bus id. It used to be left
    at None even though the query already asked for the bus id, so every
    caller that wanted "which card is this on" had no answer — and on a box
    with two cards, "VRAM that stopping this frees" is meaningless without it.
    """
    nvidia, amd = await asyncio.gather(
        _probe_nvidia_processes(), asyncio.to_thread(probe_amd_processes)
    )
    return nvidia + amd


async def _probe_nvidia_processes() -> list[GpuProcess]:
    if not shutil.which("nvidia-smi"):
        return []
    bus_to_index = await _nvidia_bus_index_map()
    res = await run_capture([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_bus_id",
        "--format=csv,noheader,nounits",
    ], timeout=_NV_TIMEOUT_S)
    if not res.ok:
        return []
    out: list[GpuProcess] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            out.append(GpuProcess(
                pid=int(parts[0]),
                process_name=parts[1],
                used_mb=int(parts[2]),
                gpu_index=bus_to_index.get(parts[3].lower()) if len(parts) > 3 else None,
            ))
        except ValueError:
            continue
    return out


async def _nvidia_bus_index_map() -> dict[str, int]:
    """PCI bus id (lowercased) -> nvidia-smi GPU index. Empty when unavailable.

    A separate query because the compute-apps listing reports the bus id and
    only the GPU listing knows which index it belongs to.
    """
    res = await run_capture([
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits",
    ], timeout=_NV_TIMEOUT_S)
    if not res.ok:
        return {}
    out: dict[str, int] = {}
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            out[parts[1].lower()] = int(parts[0])
        except ValueError:
            continue
    return out


async def probe_gpus() -> list[GpuInfo]:
    """Every GPU LlamaDeck can see. Empty on a CPU-only box.

    NVIDIA and AMD are probed *together*. Returning whichever vendor answered
    first was wrong on a mixed workstation — an RTX 5090 alongside a Radeon
    R9700 is a real llama.cpp target (one CUDA+Vulkan binary drives both), and
    the first-vendor-wins shortcut hid the second card completely.

    Apple stays a fallback: Metal and a discrete AMD card never coexist as
    offload targets, so there is nothing to merge there.
    """
    gpus = await probe_nvidia()
    gpus.extend(probe_amd())
    if gpus:
        return gpus
    return probe_apple()


def offload_gpus(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """The GPUs worth planning a model against.

    Integrated parts are dropped. A desktop Ryzen's iGPU reports a 2 GB
    carve-out plus the full ~46 GB GTT aperture, so budgeting against it would
    promise system RAM as if it were VRAM — and on such a box it is driving the
    display, not running models.

    On an APU-only machine (Strix Halo) the iGPU is the only accelerator there
    is, so an all-integrated list is returned unchanged rather than emptied.
    """
    discrete = [g for g in gpus if not g.integrated]
    return discrete or gpus


# Memory is the only part of this query that must succeed. The sensors ride
# along on the same call — one subprocess, not five — and each is allowed to
# come back "[N/A]".
_NV_FIELDS = (
    "index", "name", "memory.total", "memory.used", "memory.free",
    "utilization.gpu", "temperature.gpu", "fan.speed", "power.draw", "clocks.sm",
)
_NV_FIELDS_BASE = _NV_FIELDS[:5]


def _nv_num(raw: str) -> float | None:
    """One nvidia-smi cell as a number. "[N/A]", "[Not Supported]", "" -> None.

    A blower-less card answers "[N/A]" for fan.speed forever, which is a real
    answer ("this card has no fan"), not an error — so it must not poison the
    row it shares with the memory numbers.
    """
    v = raw.strip()
    if not v or v.startswith("["):
        return None
    try:
        return float(v)
    except ValueError:
        return None


async def _nvidia_query(fields: tuple[str, ...]) -> list[GpuInfo] | None:
    """Run one --query-gpu. None means the call itself failed (vs. no GPUs)."""
    res = await run_capture([
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ], timeout=_NV_TIMEOUT_S)
    if not res.ok:
        if res.timed_out:
            _note_nvidia_failure(f"it did not answer within {_NV_TIMEOUT_S:.0f}s")
        elif res.error:
            _note_nvidia_failure(res.error)
        elif res.text:
            # nvidia-smi prints its own failures on STDOUT, not stderr —
            # "Failed to initialize NVML: Driver/library version mismatch"
            # arrives there with exit code 18.
            _note_nvidia_failure(res.text.splitlines()[0])
        else:
            _note_nvidia_failure(f"it exited {res.rc} with no output")
        return None
    _clear_nvidia_failure()
    gpus: list[GpuInfo] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        cell = dict(zip(fields, parts))
        try:
            gpu = GpuInfo(
                index=int(cell["index"]),
                name=cell["name"],
                total_mb=int(cell["memory.total"]),
                used_mb=int(cell["memory.used"]),
                free_mb=int(cell["memory.free"]),
                vendor="nvidia",
            )
        except ValueError:
            continue
        gpu.util_percent = _nv_num(cell.get("utilization.gpu", ""))
        gpu.temp_c = _nv_num(cell.get("temperature.gpu", ""))
        gpu.fan_percent = _nv_num(cell.get("fan.speed", ""))
        gpu.power_w = _nv_num(cell.get("power.draw", ""))
        clock = _nv_num(cell.get("clocks.sm", ""))
        gpu.clock_mhz = int(clock) if clock is not None else None
        gpus.append(gpu)
    return gpus


def _note_nvidia_failure(reason: str) -> None:
    """Record and report a failing `nvidia-smi`, once per distinct reason.

    Silence here is expensive. The usual cause is a driver package upgraded
    without a reboot: the loaded kernel module and the userspace libraries have
    drifted apart, nvidia-smi refuses to initialise NVML, and the card simply
    disappears from the dashboard — no error, no empty row, nothing in the log.
    The VRAM panel polls every few seconds, so this cannot warn every time
    either; a repeat of the same reason is dropped.
    """
    global _nv_failure
    if reason == _nv_failure:
        return
    _nv_failure = reason
    log.warning(
        "nvidia-smi is installed but not answering (%s) — NVIDIA GPUs will not be "
        "listed. A driver package upgraded without a reboot is the usual cause: "
        "compare `cat /proc/driver/nvidia/version` with `modinfo nvidia | grep ^version`.",
        reason,
    )


def _clear_nvidia_failure() -> None:
    global _nv_failure
    if _nv_failure is not None:
        log.info("nvidia-smi is answering again")
        _nv_failure = None


def nvidia_probe_failure() -> str | None:
    """Why the NVIDIA probe is coming back empty, or None when it is fine.

    Surfaced by /api/server/vram so a missing card is explained in the UI
    instead of leaving the user to wonder where their GPU went.
    """
    return _nv_failure


async def probe_nvidia() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    gpus = await _nvidia_query(_NV_FIELDS)
    if gpus is None:
        # nvidia-smi rejects the WHOLE query when it does not recognise one
        # field name, so on an older driver the sensors would take the memory
        # numbers down with them. Retry with the fields it has always had.
        gpus = await _nvidia_query(_NV_FIELDS_BASE)
    return gpus or []


# --------------------------------------------------------------------------
# AMD (amdgpu on Linux)
# --------------------------------------------------------------------------

def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_str(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


# Shipped by pciutils/hwdata. Present on essentially every desktop Linux, but
# only ever a nicety here: a card newer than the file (the R9700, 1002:7551)
# simply falls back to its PCI id.
_PCI_IDS_PATHS = ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids")


def _pci_device_name(vendor_id: str, device_id: str) -> str:
    """Human name for a PCI vendor:device pair, or "" when unlisted.

    pci.ids is a flat text file: vendor lines start at column 0, their device
    lines are indented one tab, subsystem lines two.
    """
    if not vendor_id or not device_id:
        return ""
    for path in _PCI_IDS_PATHS:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                in_vendor = False
                for line in fh:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    if not line.startswith("\t"):
                        if in_vendor:
                            return ""  # left our vendor block without a hit
                        in_vendor = line[:4].lower() == vendor_id
                    elif in_vendor and not line.startswith("\t\t"):
                        if line[1:5].lower() == device_id:
                            return line[5:].strip()
        except OSError:
            continue
    return ""


# amdgpu labels its thermal probes; which number carries which probe differs
# between cards, so the label is read rather than the index assumed.
_AMD_TEMP_KEYS = {"edge": "temp_c", "junction": "hotspot_c", "mem": "mem_temp_c"}


def amd_sensors(dev: Path) -> dict:
    """Live sensors for one amdgpu card, from sysfs + its hwmon node.

    Every reading is optional: `gpu_busy_percent` is missing on older kernels,
    a passively cooled card has no fan node, and an iGPU exposes neither. Keys
    absent from the returned dict mean "this card cannot answer", which the UI
    renders as "—" instead of a zero it would otherwise have to invent.
    """
    out: dict = {}
    busy = _read_int(dev / "gpu_busy_percent")
    if busy is not None:
        out["util_percent"] = float(busy)
    try:
        hwmons = sorted((dev / "hwmon").iterdir())
    except OSError:
        return out
    for hwmon in hwmons:
        for i in range(1, 6):
            micro_c = _read_int(hwmon / f"temp{i}_input")
            if micro_c is None:
                continue
            label = _read_str(hwmon / f"temp{i}_label").lower()
            # An unlabelled single probe is the edge sensor by convention.
            key = _AMD_TEMP_KEYS.get(label, "temp_c" if i == 1 else "")
            if key:
                out.setdefault(key, micro_c / 1000.0)
        rpm = _read_int(hwmon / "fan1_input")
        if rpm is not None:
            out.setdefault("fan_rpm", rpm)
        pwm = _read_int(hwmon / "pwm1")
        pwm_max = _read_int(hwmon / "pwm1_max") or 255
        if pwm is not None and pwm_max > 0:
            out.setdefault("fan_percent", round(pwm * 100.0 / pwm_max, 1))
        for fname in ("power1_average", "power1_input"):
            micro_w = _read_int(hwmon / fname)
            if micro_w:
                out.setdefault("power_w", micro_w / 1_000_000.0)
                break
        # freq1 is the shader clock in Hz. RDNA parks it at 0 when the card is
        # idle, which reads as "no answer" — better a dash than a fake 0 MHz.
        hz = _read_int(hwmon / "freq1_input")
        if hz:
            out.setdefault("clock_mhz", hz // 1_000_000)
    return out


def amd_card_is_integrated(card: AmdCard) -> bool:
    """True when this card's VRAM is carved out of system RAM (an APU iGPU).

    A discrete card has its own memory chips and amdgpu names their maker in
    `mem_info_vram_vendor` ("samsung", "hynix", …); an APU has no dedicated
    memory to name, so the file is absent.

    The GTT comparison is only a fallback for when that signal is missing. It
    cannot be the primary test: *every* amdgpu device gets a GTT aperture sized
    against system RAM, so `GTT >= VRAM` on its own labels a 32 GB discrete
    Radeon integrated — on a 96 GB box the R9700 reports 32 GB VRAM against a
    46 GB aperture.
    """
    if card.vram_vendor:
        return False
    return card.gtt_total_mb > 0 and card.gtt_total_mb >= card.total_mb


def amd_sysfs_cards(root: Path | None = None) -> list[AmdCard]:
    """Parse amdgpu's DRM sysfs nodes. Byte counters, no root needed.

    `root` is injectable so tests can point at a fixture tree instead of
    /sys/class/drm.
    """
    base = root or Path("/sys/class/drm")
    cards: list[AmdCard] = []
    try:
        entries = sorted(p for p in base.iterdir() if _CARD_RE.match(p.name))
    except OSError:
        return []
    for entry in entries:
        dev = entry / "device"
        total = _read_int(dev / "mem_info_vram_total")
        if total is None:
            continue  # not an amdgpu card (or an older kernel)
        used = _read_int(dev / "mem_info_vram_used") or 0
        gtt_total = _read_int(dev / "mem_info_gtt_total") or 0
        gtt_used = _read_int(dev / "mem_info_gtt_used") or 0
        vendor_id = _read_str(dev / "vendor").removeprefix("0x").lower()
        device_id = _read_str(dev / "device").removeprefix("0x").lower()
        pci_id = f"{vendor_id}:{device_id}" if vendor_id and device_id else ""
        name = ""
        for candidate in ("product_name", "device_name"):
            text = _read_str(dev / candidate)
            if text:
                name = text
                break
        # amdgpu leaves product_name empty on most consumer and workstation
        # cards, so fall back to the PCI id database and finally to the raw id
        # — "AMD GPU (1002:7551)" still tells the user which card this is,
        # where a bare "AMD GPU" for two different cards does not.
        if not name:
            name = _pci_device_name(vendor_id, device_id)
        if not name:
            name = f"AMD GPU ({pci_id})" if pci_id else "AMD GPU"
        try:
            index = int(entry.name.removeprefix("card"))
        except ValueError:
            index = len(cards)
        mb = 1024 * 1024
        cards.append(AmdCard(
            index=index,
            name=name,
            total_mb=total // mb,
            used_mb=used // mb,
            free_mb=max(0, (total - used)) // mb,
            gtt_total_mb=gtt_total // mb,
            gtt_used_mb=gtt_used // mb,
            vram_vendor=_read_str(dev / "mem_info_vram_vendor"),
            pci_id=pci_id,
            pci_addr=_pci_addr_of(dev),
            sensors=amd_sensors(dev),
        ))
    return cards


# A PCI address as DRM fdinfo prints it ("0000:03:00.0"). Resolving the sysfs
# `device` symlink lands on exactly this, but only on a real /sys — a fixture
# tree has a plain directory there, so the shape is checked rather than assumed.
_PCI_ADDR_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d$")


def _pci_addr_of(dev: Path) -> str:
    try:
        name = dev.resolve().name
    except OSError:
        return ""
    return name if _PCI_ADDR_RE.match(name) else ""


def _fdinfo_kib(value: str) -> int:
    """"27315284 KiB" -> 27315284. Unknown units are refused, not guessed."""
    parts = value.split()
    if not parts:
        return 0
    try:
        n = int(parts[0])
    except ValueError:
        return 0
    unit = parts[1].lower() if len(parts) > 1 else "kib"
    if unit in ("kib", "kb"):
        return n
    if unit in ("mib", "mb"):
        return n * 1024
    if unit in ("gib", "gb"):
        return n * 1024 * 1024
    return 0


def _drm_clients_of(pid: int, proc_root: Path) -> dict[tuple[str, str], int]:
    """{(pci_addr, drm-client-id): vram_kib} for one process.

    llama.cpp holds a dozen descriptors on the render node open and every one
    of them reports the *same* allocation — summing descriptors turned a 26 GB
    model into 53 GB. Descriptors belonging to one client share
    `drm-client-id`, which is what makes the dedupe possible.
    """
    out: dict[tuple[str, str], int] = {}
    fd_dir = proc_root / str(pid) / "fd"
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return out  # gone, or another user's process
    for fd in fds:
        try:
            if not os.readlink(fd).startswith("/dev/dri/"):
                continue
        except OSError:
            continue
        try:
            text = (proc_root / str(pid) / "fdinfo" / fd.name).read_text()
        except OSError:
            continue
        addr = client = ""
        vram_kib = 0
        for line in text.splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "drm-pdev":
                addr = value
            elif key == "drm-client-id":
                client = value
            elif key == "drm-memory-vram":
                vram_kib = _fdinfo_kib(value)
            elif key == "drm-total-vram" and not vram_kib:
                vram_kib = _fdinfo_kib(value)
        if addr and client:
            out[(addr, client)] = max(out.get((addr, client), 0), vram_kib)
    return out


def probe_amd_processes(
    cards: list[AmdCard] | None = None,
    proc_root: Path | None = None,
    include_integrated: bool = False,
) -> list[GpuProcess]:
    """Per-process VRAM on amdgpu, read from DRM fdinfo (kernel 5.19+).

    amdgpu has no nvidia-smi: `mem_info_vram_used` is a card-wide total that
    cannot say who holds what. Without this, "stopping that model frees N GB"
    was unanswerable on a Radeon — which is what made a busy card look
    permanently full — and no measurement could correct its VRAM estimate
    either, so the planner had to keep a 2 GB guess-margin forever.

    Integrated graphics are skipped. Every windowed program on the machine
    holds a DRM context on whichever card drives the display, so an APU's
    render node lists the entire desktop — gnome-shell, the browser, Xwayland,
    the IME. On this box that buried the two llama-servers under ten rows of
    compositor clients holding tens of megabytes on a 3 GB carve-out that is
    not an offload target and never will be. NVIDIA never had the problem:
    `--query-compute-apps` lists compute contexts only. Pass
    `include_integrated=True` to see them anyway.

    Only processes of this user are visible; anything unreadable is skipped
    rather than guessed at.
    """
    cards = amd_sysfs_cards() if cards is None else cards
    if not include_integrated:
        cards = [c for c in cards if not amd_card_is_integrated(c)]
    by_addr = {c.pci_addr: c.index for c in cards if c.pci_addr}
    if not by_addr:
        return []
    root = proc_root or Path("/proc")
    out: list[GpuProcess] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        per_card: dict[int, int] = {}
        for (addr, _client), kib in _drm_clients_of(pid, root).items():
            index = by_addr.get(addr)
            if index is None:
                continue
            per_card[index] = per_card.get(index, 0) + kib
        if not per_card:
            continue
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            comm = ""
        for index, kib in per_card.items():
            used_mb = kib // 1024
            if used_mb > 0:
                out.append(GpuProcess(
                    pid=pid, process_name=comm, used_mb=used_mb,
                    gpu_index=index, vendor="amd",
                ))
    return out


def probe_amd(root: Path | None = None) -> list[GpuInfo]:
    from .accel import platform_info

    out: list[GpuInfo] = []
    unified = platform_info().unified_memory
    for c in amd_sysfs_cards(root):
        # A UMA-mode APU reports a token VRAM carve-out (often 512 MB) and
        # borrows the rest from system RAM via GTT. Budgeting against the
        # carve-out alone would declare every model "doesn't fit", so on
        # integrated parts the usable pool is VRAM + GTT.
        integrated = amd_card_is_integrated(c)
        card_unified = unified or integrated
        total = c.total_mb + c.gtt_total_mb if card_unified else c.total_mb
        used = c.used_mb + c.gtt_used_mb if card_unified else c.used_mb
        out.append(GpuInfo(
            index=c.index,
            name=c.name,
            total_mb=total,
            used_mb=used,
            free_mb=max(0, total - used),
            vendor="amd",
            unified=card_unified,
            integrated=integrated,
            gtt_total_mb=c.gtt_total_mb,
            gtt_used_mb=c.gtt_used_mb,
            **c.sensors,
        ))
    return out


# --------------------------------------------------------------------------
# Apple Silicon (unified memory)
# --------------------------------------------------------------------------

def apple_gpu_budget_mb(total_ram_mb: int, wired_limit_mb: int = 0) -> int:
    """How much of the unified pool Metal may actually allocate.

    macOS caps GPU-wired memory; the cap is tunable via
    `sysctl iogpu.wired_limit_mb` and defaults to roughly 75 % of RAM on
    machines with >= 32 GB, ~2/3 below that. A 0 value means "default".
    """
    if wired_limit_mb > 0:
        return min(wired_limit_mb, total_ram_mb)
    ratio = 0.75 if total_ram_mb >= 32 * 1024 else 0.67
    return int(total_ram_mb * ratio)


def probe_apple() -> list[GpuInfo]:
    from .accel import platform_info, sysctl

    p = platform_info()
    if not p.is_apple_silicon:
        return []
    import psutil

    mem = psutil.virtual_memory()
    mb = 1024 * 1024
    total_ram_mb = mem.total // mb
    available_mb = mem.available // mb
    try:
        wired = int(sysctl("iogpu.wired_limit_mb") or 0)
    except ValueError:
        wired = 0
    budget = apple_gpu_budget_mb(total_ram_mb, wired)
    # There is no per-process Metal accounting without private APIs, so
    # "free" is what the system could still hand the GPU right now.
    free = min(budget, available_mb)
    return [GpuInfo(
        index=0,
        name=f"{p.cpu_name} (unified memory)",
        total_mb=budget,
        used_mb=max(0, budget - free),
        free_mb=free,
        vendor="apple",
        unified=True,
    )]
