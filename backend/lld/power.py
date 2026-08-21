"""GPU + CPU power probe in Watts.

GPU:
  * NVIDIA — `nvidia-smi --query-gpu=power.draw` (instantaneous, per-card).
  * AMD    — amdgpu's hwmon `power1_average` (µW) under the DRM node; no root
             needed. APUs often expose only the whole-SoC rail, which already
             includes the CPU — see probe_amd_gpu_power_w.
  * Apple  — nothing: `powermetrics` requires root, so power stays None and
             the UI hides the panel.

CPU: Linux RAPL energy counters under /sys/class/powercap. Every `package-*`
     domain is summed (multi-socket, and AMD Zen exposes its sockets through
     the same intel-rapl tree). Watts = ΔenergyJ / Δt across two reads kept in
     module-level state.

The RAPL files are typically root-only (mode 0400). To expose them once:
    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
or persist with a udev/systemd-tmpfiles rule. probe_cpu_power_w returns None
on permission errors so the UI degrades cleanly.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import psutil

from .procutil import run_capture

_POWERCAP = Path("/sys/class/powercap")
# domain path -> (energy_uj, monotonic timestamp) of the previous sample
_prev_cpu: dict[Path, tuple[int, float]] = {}

# Latest CPU % sampled by power_loop. None until the first delta is available.
_cpu_percent: float | None = None

# Why cpu_w is None, so the UI can say so instead of hiding the panel:
#   "ok" | "warming" | "denied" (RAPL present but root-only) | "unsupported"
_cpu_power_status: str = "warming"


# Latest CPU package temperature, same 2 Hz cadence as _cpu_percent.
_cpu_temp_c: float | None = None

# Which sensor is "the CPU temperature", best first. AMD's k10temp calls its
# control temperature Tctl, Intel's coretemp calls it the package. Everything
# else on a desktop board (chipset, VRM, drive, network chip) also lands in
# psutil's dict, so the chip is matched by name rather than by taking the
# first reading that happens to be there.
_CPU_TEMP_CHIPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("k10temp", ("tctl", "tdie")),
    ("coretemp", ("package id 0",)),
    ("zenpower", ("tdie",)),
    ("cpu_thermal", ("",)),
)


def probe_cpu_temp_c() -> float | None:
    """CPU package temperature in Celsius, or None where nothing reports one.

    Board fan tachometers are deliberately NOT read alongside this: on this
    box the super-I/O chip exposes seven unlabelled fans, and guessing which
    one is the CPU cooler would put a made-up number on the dashboard.
    """
    try:
        chips = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None  # macOS/Windows have no such reading without extra tooling
    for chip, labels in _CPU_TEMP_CHIPS:
        for entry in chips.get(chip, ()):
            if entry.label.lower() in labels and entry.current:
                return float(entry.current)
    return None


def get_cpu_temp_c() -> float | None:
    return _cpu_temp_c


def get_cpu_percent() -> float | None:
    return _cpu_percent


def get_cpu_power_status() -> str:
    return _cpu_power_status


def get_ram_snapshot() -> dict:
    m = psutil.virtual_memory()
    return {
        "total_mb": m.total // (1024 * 1024),
        "used_mb": (m.total - m.available) // (1024 * 1024),
        "free_mb": m.available // (1024 * 1024),
        "percent": float(m.percent),
    }


class EnergyTracker:
    """Integrates total system power into cumulative energy and busy time.

    Only accumulates while the LLM is actively working (busy_slots > 0). Uses
    trapezoidal integration between consecutive power samples — caller is
    expected to tick at a steady cadence (~2 Hz).
    """

    def __init__(self) -> None:
        self.total_j: float = 0.0
        self.busy_s: float = 0.0
        self._last_ts: float | None = None
        self._last_w: float | None = None

    def tick(self, total_w: float | None, is_busy: bool) -> None:
        now = time.monotonic()
        if (
            is_busy
            and self._last_ts is not None
            and self._last_w is not None
            and total_w is not None
        ):
            dt = now - self._last_ts
            # Cap dt to ignore long gaps (sleep, restart, paused loop).
            if 0 < dt < 5.0:
                self.total_j += ((self._last_w + total_w) / 2.0) * dt
                self.busy_s += dt
        self._last_ts = now
        # Reset baseline when not busy so the next busy window starts fresh
        # (otherwise a long idle gap would be linearly interpolated through).
        self._last_w = total_w if is_busy else None

    def snapshot(self) -> dict:
        return {
            "energy_wh": self.total_j / 3600.0,
            "energy_j": self.total_j,
            "busy_seconds": self.busy_s,
        }


_tracker = EnergyTracker()


def get_energy_tracker() -> EnergyTracker:
    return _tracker


async def power_loop(get_busy_fn) -> None:
    """Background task: poll power + CPU usage at 2 Hz and feed the tracker.

    `get_busy_fn` is called each tick and must return True if any LLM preset
    has at least one active slot. Errors are swallowed so the loop never
    crashes the backend.
    """
    global _cpu_percent, _cpu_temp_c
    # Seed psutil's cpu_percent so subsequent non-blocking calls return deltas.
    psutil.cpu_percent(interval=None)
    while True:
        try:
            gpu_list = await probe_gpu_power_w()
            cpu = await probe_cpu_power_w()
            gpu = sum(gpu_list) if gpu_list else None
            if gpu is not None or cpu is not None:
                total = (gpu or 0.0) + (cpu or 0.0)
            else:
                total = None
            try:
                busy = bool(get_busy_fn())
            except Exception:
                busy = False
            _tracker.tick(total, busy)
            # Sample CPU % over the same 0.5s tick window — non-blocking
            # because psutil computed it against its previous internal call.
            try:
                _cpu_percent = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                _cpu_temp_c = probe_cpu_temp_c()
            except Exception:
                pass
        except Exception:
            pass
        await asyncio.sleep(0.5)


async def probe_gpu_power_w() -> list[float]:
    """Per-GPU power draw in Watts, from EVERY vendor present. Empty when none.

    NVIDIA and AMD are not alternatives. This box runs a 5090 and an R9700 at
    the same time, and taking the NVIDIA branch merely because `nvidia-smi`
    exists left the second card's draw out of the dashboard entirely — the
    panel under-reported system power by the whole AMD card.
    """
    nvidia = await _probe_nvidia_power_w()
    return nvidia + probe_amd_gpu_power_w()


async def _probe_nvidia_power_w() -> list[float]:
    if not shutil.which("nvidia-smi"):
        return []
    # run_capture, not a bare wait_for: this runs twice a second forever, so a
    # driver that makes nvidia-smi hang would otherwise leave a stuck process
    # behind on every single tick.
    res = await run_capture([
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ], timeout=2.0)
    if not res.ok:
        return []
    out: list[float] = []
    for line in res.stdout.strip().splitlines():
        s = line.strip()
        if not s or s.lower().startswith("[n/a"):
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def probe_amd_gpu_power_w(root: Path | None = None) -> list[float]:
    """Per-card discrete AMD GPU power from amdgpu's hwmon node (µW -> W).

    `root` is injectable for tests.

    An APU's iGPU is skipped: its rail is the whole SoC, the same silicon RAPL
    already reports as CPU package power. Counting both would inflate system
    power by the entire CPU on any desktop with integrated graphics — which is
    every Ryzen box this runs on. (On an APU-ONLY machine there is no discrete
    card to report and the CPU reading covers the GPU anyway.)
    """
    from .vram import amd_card_is_integrated, amd_sysfs_cards

    base = root or Path("/sys/class/drm")
    # Skip only what is POSITIVELY identified as integrated. A node we cannot
    # classify (no VRAM info at all — an unfamiliar kernel layout) still gets
    # reported: losing the power reading silently is worse than the risk of
    # double-counting one exotic part.
    igpu = {c.index for c in amd_sysfs_cards(root) if amd_card_is_integrated(c)}
    out: list[float] = []
    try:
        cards = sorted(p for p in base.iterdir() if p.name.startswith("card") and p.name[4:].isdigit())
    except OSError:
        return []
    for card in cards:
        if int(card.name[4:]) in igpu:
            continue
        hwmon_root = card / "device" / "hwmon"
        try:
            hwmons = sorted(hwmon_root.iterdir())
        except OSError:
            continue
        for hwmon in hwmons:
            for fname in ("power1_average", "power1_input"):
                try:
                    micro_w = int((hwmon / fname).read_text().strip())
                except (OSError, ValueError):
                    continue
                if micro_w > 0:
                    out.append(micro_w / 1_000_000.0)
                    break
            else:
                continue
            break
    return out


def _rapl_package_domains(root: Path | None = None) -> list[Path]:
    """Top-level RAPL package domains (`intel-rapl:0`, `intel-rapl:1`, …).

    Sub-domains (`intel-rapl:0:0` = cores) are skipped so we don't count the
    same energy twice. AMD Zen shows up here as well — the kernel registers
    its sockets under the same intel-rapl tree.
    """
    base = root or _POWERCAP
    out: list[Path] = []
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return []
    for entry in entries:
        # "intel-rapl:0" has one colon; "intel-rapl:0:0" (a sub-domain) has two.
        if entry.name.count(":") != 1:
            continue
        try:
            if not (entry / "name").read_text().strip().startswith("package"):
                continue
        except OSError:
            continue
        if (entry / "energy_uj").exists():
            out.append(entry)
    return out


async def probe_cpu_power_w(root: Path | None = None) -> float | None:
    """Average CPU package power in W between this call and the previous one.

    Sums every package domain (multi-socket boxes report one each). Returns
    None on the first call (no delta yet), on unreadable/permission-denied
    counters, and on platforms without RAPL (macOS).
    """
    global _cpu_power_status
    domains = _rapl_package_domains(root)
    if not domains:
        _cpu_power_status = "unsupported"
        return None
    now_ts = time.monotonic()
    total_w = 0.0
    have_sample = False
    denied = False
    for domain in domains:
        try:
            now_uj = int((domain / "energy_uj").read_text().strip())
        except PermissionError:
            denied = True
            continue
        except (OSError, ValueError):
            continue
        prev = _prev_cpu.get(domain)
        _prev_cpu[domain] = (now_uj, now_ts)
        if prev is None:
            continue
        dt = now_ts - prev[1]
        de = now_uj - prev[0]
        if dt <= 0:
            continue
        if de < 0:
            # RAPL counter wraparound — add max_energy_range_uj.
            try:
                de += int((domain / "max_energy_range_uj").read_text().strip())
            except (OSError, ValueError):
                continue
        total_w += (de / 1_000_000.0) / dt
        have_sample = True
    if have_sample:
        _cpu_power_status = "ok"
    elif denied:
        # The counters exist but are mode 0400. This is the common case on a
        # stock distro, and it is worth naming: the panel is otherwise just
        # silently empty on a machine that does support RAPL.
        _cpu_power_status = "denied"
    else:
        # Readable, but no delta yet — the first tick after start.
        _cpu_power_status = "warming"
    return total_w if have_sample else None
