"""Live GPU sensor readings (temperature, fan, utilisation, clock).

The dashboard shows these as bare numbers, so "no reading" must survive the
whole way as None — a card without a fan must never render as a fan sitting
at 0 %.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lld import vram


# --------------------------------------------------------------------------
# AMD — sysfs + hwmon
# --------------------------------------------------------------------------

def _write_card(root: Path, index: int, *, hwmon: dict[str, str] | None = None,
                busy: str | None = None, vram_vendor: str = "samsung") -> Path:
    dev = root / f"card{index}" / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_vram_total").write_text(str(32 * 1024**3))
    (dev / "mem_info_vram_used").write_text(str(8 * 1024**3))
    (dev / "mem_info_vram_vendor").write_text(vram_vendor)
    if busy is not None:
        (dev / "gpu_busy_percent").write_text(busy)
    if hwmon is not None:
        node = dev / "hwmon" / "hwmon9"
        node.mkdir(parents=True)
        for k, v in hwmon.items():
            (node / k).write_text(v)
    return dev


def test_amd_sensors_read_labelled_probes_and_fan(tmp_path):
    """The R9700's shape: three labelled temps, RPM plus a PWM duty cycle."""
    _write_card(tmp_path, 2, busy="63\n", hwmon={
        "temp1_label": "edge\n", "temp1_input": "54000\n",
        "temp2_label": "junction\n", "temp2_input": "71000\n",
        "temp3_label": "mem\n", "temp3_input": "66000\n",
        "fan1_input": "1840\n", "pwm1": "153\n", "pwm1_max": "255\n",
        "power1_average": "212000000\n", "freq1_input": "2400000000\n",
    })
    card = vram.amd_sysfs_cards(tmp_path)[0]
    assert card.sensors == {
        "util_percent": 63.0,
        "temp_c": 54.0,
        "hotspot_c": 71.0,
        "mem_temp_c": 66.0,
        "fan_rpm": 1840,
        "fan_percent": 60.0,
        "power_w": 212.0,
        "clock_mhz": 2400,
    }


def test_amd_sensors_survive_a_card_that_reports_nothing(tmp_path):
    """No hwmon node at all (an older kernel) is not an error."""
    _write_card(tmp_path, 0)
    assert vram.amd_sysfs_cards(tmp_path)[0].sensors == {}


def test_amd_igpu_has_no_fan_keys(tmp_path):
    """An iGPU reports a temperature but has no fan of its own.

    The keys must be *absent*, not zero: probe_amd splats this dict onto
    GpuInfo, where a missing key leaves the field None and the UI prints "—".
    """
    _write_card(tmp_path, 3, busy="4\n", vram_vendor="", hwmon={
        "temp1_label": "edge\n", "temp1_input": "45000\n",
        "power1_input": "30000\n",
    })
    sensors = vram.amd_sysfs_cards(tmp_path)[0].sensors
    assert sensors["temp_c"] == 45.0
    assert "fan_rpm" not in sensors and "fan_percent" not in sensors
    assert "hotspot_c" not in sensors


def test_amd_idle_shader_clock_is_not_reported_as_zero(tmp_path):
    """RDNA parks freq1 at 0 when idle; that is "no answer", not 0 MHz."""
    _write_card(tmp_path, 0, hwmon={"freq1_input": "0\n", "temp1_input": "40000\n"})
    assert "clock_mhz" not in vram.amd_sysfs_cards(tmp_path)[0].sensors


def test_amd_unlabelled_probe_one_is_the_edge_sensor(tmp_path):
    _write_card(tmp_path, 0, hwmon={"temp1_input": "48000\n"})
    assert vram.amd_sysfs_cards(tmp_path)[0].sensors["temp_c"] == 48.0


def test_probe_amd_puts_the_sensors_on_the_gpu(tmp_path, monkeypatch):
    _write_card(tmp_path, 2, busy="63\n", hwmon={
        "temp1_label": "junction\n", "temp1_input": "71000\n",
        "fan1_input": "1840\n",
    })
    gpu = vram.probe_amd(tmp_path)[0]
    assert (gpu.hotspot_c, gpu.fan_rpm, gpu.util_percent) == (71.0, 1840, 63.0)
    assert gpu.temp_c is None  # this card labelled probe 1 as the junction


# --------------------------------------------------------------------------
# NVIDIA — nvidia-smi CSV
# --------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int = 0) -> None:
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, b""


def _fake_smi(monkeypatch, by_query: dict[str, tuple[bytes, int]]):
    """Answer per query string, so the fallback path can be exercised."""
    seen: list[str] = []

    async def _exec(*args, **kwargs):
        query = next(a for a in args if a.startswith("--query-gpu="))
        seen.append(query)
        for needle, (out, rc) in by_query.items():
            if needle in query:
                return _FakeProc(out, rc)
        return _FakeProc(b"", 2)

    monkeypatch.setattr(vram.asyncio, "create_subprocess_exec", _exec)
    monkeypatch.setattr(vram.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    return seen


async def test_nvidia_reads_the_sensors(monkeypatch):
    _fake_smi(monkeypatch, {"utilization.gpu": (
        b"0, NVIDIA GeForce RTX 5090, 32607, 31752, 399, 97, 78, 67, 499.84, 2385\n", 0)})
    gpu = (await vram.probe_nvidia())[0]
    assert (gpu.util_percent, gpu.temp_c, gpu.fan_percent) == (97.0, 78.0, 67.0)
    assert (gpu.power_w, gpu.clock_mhz) == (499.84, 2385)
    assert gpu.used_mb == 31752


async def test_nvidia_fanless_card_reports_no_fan(monkeypatch):
    """A blower-less A100 answers [N/A] — one dead cell, not a dead row."""
    _fake_smi(monkeypatch, {"utilization.gpu": (
        b"0, NVIDIA A100, 81920, 1024, 80896, 0, 41, [N/A], 62.50, 1410\n", 0)})
    gpu = (await vram.probe_nvidia())[0]
    assert gpu.fan_percent is None
    assert (gpu.temp_c, gpu.total_mb) == (41.0, 81920)


async def test_nvidia_falls_back_when_the_driver_rejects_the_query(monkeypatch):
    """An old driver fails the WHOLE query over one unknown field name.

    The memory numbers are the point of this probe; losing them to a sensor
    the dashboard merely decorates with would be the worse trade.
    """
    seen = _fake_smi(monkeypatch, {
        "utilization.gpu": (b"", 6),
        "memory.free": (b"0, NVIDIA GeForce GTX 1080, 8192, 512, 7680\n", 0),
    })
    gpus = await vram.probe_nvidia()
    assert len(seen) == 2  # full query, then the fallback
    assert gpus[0].used_mb == 512
    assert gpus[0].temp_c is None


async def test_nvidia_absent_is_not_an_error(monkeypatch):
    monkeypatch.setattr(vram.shutil, "which", lambda _: None)
    assert await vram.probe_nvidia() == []


@pytest.mark.parametrize("cell,expected", [
    ("97", 97.0), (" 78 ", 78.0), ("[N/A]", None),
    ("[Not Supported]", None), ("", None), ("n/a", None),
])
def test_nvidia_cell_parsing(cell, expected):
    assert vram._nv_num(cell) == expected
