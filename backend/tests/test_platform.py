"""Platform-portability tests: Apple Silicon, AMD (incl. Ryzen AI Max APUs)
and Windows paths that the maintainer's NVIDIA/Linux box never exercises.

Everything the code learns about the host comes through a handful of probes
(sysfs trees, sysctl, shutil.which), so the probes take an injectable root or
are monkeypatched, and each test pins one hardware profile.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lld import accel, fit_check, power, vram
from lld.settings import LlamaServerConfig
from lld.vram_estimate import VramEstimate


@pytest.fixture(autouse=True)
def _clear_platform_cache(monkeypatch):
    accel.refresh()
    # Detection probes real install dirs (/usr/local/cuda/bin, /opt/rocm/…);
    # blank them so a test's fake PATH is the only source of truth. Tests that
    # exercise off-PATH discovery re-populate this themselves.
    monkeypatch.setattr(accel, "_TOOL_DIRS", {})
    yield
    accel.refresh()


# --------------------------------------------------------------------------
# AMD sysfs (amdgpu) — the Ryzen AI Max +395 / Strix Halo shape
# --------------------------------------------------------------------------

def _write_amd_card(root: Path, index: int, vram_bytes: int, used_bytes: int,
                    gtt_bytes: int = 0, gtt_used: int = 0, name: str = "",
                    vram_vendor: str = "") -> None:
    dev = root / f"card{index}" / "device"
    dev.mkdir(parents=True)
    if vram_vendor:
        (dev / "mem_info_vram_vendor").write_text(f"{vram_vendor}\n")
    (dev / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
    (dev / "mem_info_vram_used").write_text(f"{used_bytes}\n")
    if gtt_bytes:
        (dev / "mem_info_gtt_total").write_text(f"{gtt_bytes}\n")
        (dev / "mem_info_gtt_used").write_text(f"{gtt_used}\n")
    if name:
        (dev / "product_name").write_text(f"{name}\n")


GB = 1024 * 1024 * 1024


def test_amd_discrete_card_reports_vram_only(tmp_path, monkeypatch):
    """A discrete Radeon: 24 GB VRAM, GTT is a side pool — the budget is VRAM."""
    _write_amd_card(tmp_path, 0, 24 * GB, 3 * GB, gtt_bytes=8 * GB, name="Radeon RX 7900 XTX")
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    gpus = vram.probe_amd(tmp_path)
    assert len(gpus) == 1
    g = gpus[0]
    assert (g.vendor, g.unified) == ("amd", False)
    assert g.name == "Radeon RX 7900 XTX"
    assert g.total_mb == 24 * 1024
    assert g.free_mb == 21 * 1024


def test_amd_apu_uma_mode_counts_gtt_into_the_budget(tmp_path, monkeypatch):
    """Strix Halo in UMA/auto mode: a token 512 MB carve-out plus a huge GTT
    pool. Budgeting against the carve-out alone would call every model too
    big, so VRAM + GTT is the usable pool and the card is marked unified."""
    _write_amd_card(tmp_path, 0, 512 * 1024 * 1024, 128 * 1024 * 1024,
                    gtt_bytes=96 * GB, gtt_used=2 * GB)
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    g = vram.probe_amd(tmp_path)[0]
    assert g.unified is True
    assert g.total_mb == 512 + 96 * 1024
    assert g.gtt_total_mb == 96 * 1024


def test_amd_discrete_card_with_gtt_larger_than_vram_is_not_unified(tmp_path, monkeypatch):
    """The R9700 shape: 32 GB of dedicated VRAM against a ~46 GB GTT aperture.

    GTT is sized against system RAM and every amdgpu device gets one, so a
    `GTT >= VRAM` test alone declared this discrete card an APU and handed the
    planner a fictional 78 GB pool. The dedicated memory vendor is what
    separates it from a real APU.
    """
    _write_amd_card(tmp_path, 2, 32624 * 1024 * 1024, 57 * 1024 * 1024,
                    gtt_bytes=45947 * 1024 * 1024, gtt_used=22 * 1024 * 1024,
                    vram_vendor="samsung")
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    g = vram.probe_amd(tmp_path)[0]
    assert (g.unified, g.integrated) == (False, False)
    assert g.total_mb == 32624          # not 32624 + 45947
    assert g.free_mb == 32624 - 57


def test_discrete_radeon_beside_an_igpu_is_not_an_apu_box(tmp_path, monkeypatch):
    """Desktop Ryzen: the 9950X iGPU is integrated, but the discrete R9700 next
    to it is what runs models — so the machine is not unified-memory."""
    _write_amd_card(tmp_path, 2, 32624 * 1024 * 1024, 0,
                    gtt_bytes=45947 * 1024 * 1024, vram_vendor="samsung")
    _write_amd_card(tmp_path, 3, 2048 * 1024 * 1024, 0,
                    gtt_bytes=45947 * 1024 * 1024)  # iGPU: no memory vendor
    cards = vram.amd_sysfs_cards(tmp_path)
    monkeypatch.setattr(vram, "amd_sysfs_cards", lambda root=None: cards)
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "")
    assert accel._detect_amd_apu() == (False, "")


def test_igpu_alone_still_reads_as_an_apu_box(tmp_path, monkeypatch):
    """Guard the inverse: with only the iGPU present it is still unified."""
    _write_amd_card(tmp_path, 3, 2048 * 1024 * 1024, 0, gtt_bytes=45947 * 1024 * 1024)
    cards = vram.amd_sysfs_cards(tmp_path)
    monkeypatch.setattr(vram, "amd_sysfs_cards", lambda root=None: cards)
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "")
    unified, _ = accel._detect_amd_apu()
    assert unified is True


def test_probe_gpus_returns_both_vendors_on_a_mixed_box(monkeypatch):
    """An RTX 5090 plus a Radeon R9700 is one llama.cpp target (a single
    CUDA+Vulkan binary drives both). Returning only the vendor that answered
    first hid the second card entirely."""
    import asyncio

    nv = vram.GpuInfo(index=0, name="RTX 5090", total_mb=32607, used_mb=0,
                      free_mb=32607, vendor="nvidia")
    amd = vram.GpuInfo(index=2, name="R9700", total_mb=32624, used_mb=0,
                       free_mb=32624, vendor="amd")

    async def _nv():
        return [nv]

    monkeypatch.setattr(vram, "probe_nvidia", _nv)
    monkeypatch.setattr(vram, "probe_amd", lambda root=None: [amd])
    gpus = asyncio.run(vram.probe_gpus())
    assert [g.vendor for g in gpus] == ["nvidia", "amd"]


def test_offload_gpus_excludes_the_integrated_card():
    """The iGPU reports its carve-out plus the whole GTT aperture; budgeting
    against it would promise system RAM as VRAM."""
    dgpu = vram.GpuInfo(index=0, name="RTX 5090", total_mb=32607, used_mb=0,
                        free_mb=32607, vendor="nvidia")
    igpu = vram.GpuInfo(index=3, name="Radeon Graphics", total_mb=47995,
                        used_mb=0, free_mb=47995, vendor="amd",
                        unified=True, integrated=True)
    assert vram.offload_gpus([dgpu, igpu]) == [dgpu]
    # max() over the unfiltered list would have picked the iGPU's fictional pool
    assert max(g.total_mb for g in vram.offload_gpus([dgpu, igpu])) == 32607


def test_offload_gpus_keeps_an_apu_only_machine():
    """On Strix Halo the iGPU is the only accelerator — filtering it would
    leave the planner with nothing."""
    igpu = vram.GpuInfo(index=0, name="Radeon 8060S", total_mb=96 * 1024,
                        used_mb=0, free_mb=96 * 1024, vendor="amd",
                        unified=True, integrated=True)
    assert vram.offload_gpus([igpu]) == [igpu]


def test_amd_probe_ignores_non_amdgpu_nodes(tmp_path):
    (tmp_path / "card0").mkdir()  # no device/mem_info_* → not an amdgpu card
    (tmp_path / "renderD128").mkdir()
    assert vram.amd_sysfs_cards(tmp_path) == []


def test_amd_probe_survives_missing_sysfs():
    assert vram.amd_sysfs_cards(Path("/nonexistent/sys/class/drm")) == []


def test_apu_detection_from_gtt_layout(tmp_path, monkeypatch):
    """No ROCm installed (no rocminfo): the GTT >= VRAM layout is what tells
    us the GPU eats system RAM."""
    _write_amd_card(tmp_path, 0, 512 * 1024 * 1024, 0, gtt_bytes=64 * GB)
    cards = vram.amd_sysfs_cards(tmp_path)
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "")
    monkeypatch.setattr(vram, "amd_sysfs_cards", lambda root=None: cards)
    unified, detail = accel._detect_amd_apu()
    assert unified is True
    assert "system RAM" in detail


def test_apu_detection_from_gfx_target(monkeypatch):
    """gfx1151 is Strix Halo — the Ryzen AI Max the user asked about."""
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "gfx1151")
    unified, detail = accel._detect_amd_apu()
    assert unified is True
    assert "gfx1151" in detail


def test_discrete_gfx_target_is_not_an_apu(monkeypatch):
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "gfx1100")  # Navi 31
    monkeypatch.setattr(vram, "amd_sysfs_cards", lambda root=None: [])
    assert accel._detect_amd_apu() == (False, "")


# --------------------------------------------------------------------------
# Apple Silicon
# --------------------------------------------------------------------------

def test_apple_budget_default_split():
    # 64 GB Mac, no explicit wired limit -> 75 % of the pool
    assert vram.apple_gpu_budget_mb(64 * 1024) == 48 * 1024
    # 16 GB Mac -> the smaller default ratio
    assert vram.apple_gpu_budget_mb(16 * 1024) == int(16 * 1024 * 0.67)
    # explicit sysctl wins, but can never exceed physical RAM
    assert vram.apple_gpu_budget_mb(64 * 1024, 60 * 1024) == 60 * 1024
    assert vram.apple_gpu_budget_mb(16 * 1024, 99 * 1024) == 16 * 1024


def test_apple_probe_reports_unified_gpu(monkeypatch):
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="darwin", arch="arm64", is_apple_silicon=True, unified_memory=True,
        cpu_name="Apple M4 Max",
    ))
    monkeypatch.setattr(accel, "sysctl", lambda key: "")
    monkeypatch.setattr(vram, "sysctl", lambda key: "", raising=False)

    class _Mem:
        total = 64 * GB
        available = 40 * GB

    monkeypatch.setattr("psutil.virtual_memory", lambda: _Mem)
    gpus = vram.probe_apple()
    assert len(gpus) == 1
    g = gpus[0]
    assert (g.vendor, g.unified) == ("apple", True)
    assert "Apple M4 Max" in g.name
    assert g.total_mb == 48 * 1024          # 75 % wired limit
    assert g.free_mb == 40 * 1024           # capped by what RAM actually has
    assert g.used_mb == 8 * 1024


def test_probe_apple_is_empty_on_other_platforms(monkeypatch):
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    assert vram.probe_apple() == []


# --------------------------------------------------------------------------
# Build backends
# --------------------------------------------------------------------------

def _fake_which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def test_backends_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="darwin", arch="arm64", is_apple_silicon=True, unified_memory=True, cpu_name="Apple M3",
    ))
    monkeypatch.setattr(accel.shutil, "which", _fake_which({"xcrun"}))
    ids = {b.id: b for b in accel.detect_backends()}
    assert ids[accel.METAL].available and ids[accel.METAL].supported
    assert not ids[accel.CUDA].supported  # no NVIDIA on macOS
    assert accel.preferred_backend() == accel.METAL
    # CPU-only on a Mac must explicitly disable Metal, which is on by default.
    assert accel.resolve(accel.CPU) == (accel.CPU, ["-DGGML_METAL=OFF"], [])
    assert accel.resolve(None)[0] == accel.METAL


def test_backends_on_amd_rocm_box(monkeypatch):
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=True,
        cpu_name="AMD Ryzen AI Max+ 395",
    ))
    monkeypatch.setattr(accel.shutil, "which", _fake_which({"hipcc", "rocminfo"}))
    monkeypatch.setattr(accel, "amd_gfx_target", lambda: "gfx1151")
    monkeypatch.setattr(accel, "has_nvidia_gpu", lambda: False)
    ids = {b.id: b for b in accel.detect_backends()}
    assert ids[accel.HIP].available
    assert ids[accel.HIP].cmake_flags == ["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1151"]
    assert not ids[accel.CUDA].available
    assert accel.preferred_backend() == accel.HIP


def test_backends_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    monkeypatch.setattr(accel.shutil, "which", _fake_which(set()))
    assert accel.preferred_backend() == accel.CPU
    assert accel.resolve("auto") == (accel.CPU, [], [])
    # An unknown id must not blow up a build request.
    assert accel.resolve("quantum-gpu")[0] == accel.CPU


def test_cached_backend_reads_cmake_cache(tmp_path):
    from lld.build import BuildManager

    assert BuildManager.cached_backend(tmp_path) is None
    (tmp_path / "CMakeCache.txt").write_text("GGML_HIP:BOOL=ON\nCMAKE_BUILD_TYPE:STRING=Release\n")
    assert BuildManager.cached_backend(tmp_path) == accel.HIP
    (tmp_path / "CMakeCache.txt").write_text("GGML_METAL:BOOL=OFF\n")
    assert BuildManager.cached_backend(tmp_path) == accel.CPU


# --------------------------------------------------------------------------
# Power probes
# --------------------------------------------------------------------------

def test_amd_gpu_power_from_hwmon(tmp_path):
    hwmon = tmp_path / "card0" / "device" / "hwmon" / "hwmon3"
    hwmon.mkdir(parents=True)
    (hwmon / "power1_average").write_text("42500000\n")  # µW
    assert power.probe_amd_gpu_power_w(tmp_path) == [42.5]


def test_amd_igpu_power_is_not_counted(tmp_path):
    """An APU's rail is the whole SoC, which RAPL already reports as CPU
    package power. Counting both inflated system power by an entire CPU on
    every desktop with integrated graphics — and this box has one next to two
    discrete cards."""
    for idx, (vram_vendor, micro_w) in enumerate(
        [("samsung", "197000000"), ("", "60000000")]   # R9700, then the iGPU
    ):
        dev = tmp_path / f"card{idx}" / "device"
        (dev / "hwmon" / "hwmon0").mkdir(parents=True)
        (dev / "hwmon" / "hwmon0" / "power1_average").write_text(micro_w + "\n")
        (dev / "mem_info_vram_total").write_text(str(32 * 1024**3))
        (dev / "mem_info_gtt_total").write_text(str(46 * 1024**3))
        if vram_vendor:
            (dev / "mem_info_vram_vendor").write_text(vram_vendor)

    assert power.probe_amd_gpu_power_w(tmp_path) == [197.0]


@pytest.mark.asyncio
async def test_gpu_power_counts_every_vendor(monkeypatch):
    """A 5090 and an R9700 draw power at the same time. Reporting only NVIDIA
    because `nvidia-smi` happens to exist hid ~200 W of the second card."""
    async def fake_nvidia():
        return [17.0]

    monkeypatch.setattr(power, "_probe_nvidia_power_w", fake_nvidia)
    monkeypatch.setattr(power, "probe_amd_gpu_power_w", lambda: [197.0])
    assert await power.probe_gpu_power_w() == [17.0, 197.0]


def test_amd_gpu_power_absent_is_empty(tmp_path):
    (tmp_path / "card0" / "device").mkdir(parents=True)
    assert power.probe_amd_gpu_power_w(tmp_path) == []


@pytest.mark.asyncio
async def test_rapl_sums_every_package_domain(tmp_path):
    """Two sockets, and a sub-domain that must not be double-counted."""
    for name, energy in (("intel-rapl:0", 1_000_000), ("intel-rapl:1", 2_000_000)):
        d = tmp_path / name
        d.mkdir()
        (d / "name").write_text("package-0\n")
        (d / "energy_uj").write_text(f"{energy}\n")
    sub = tmp_path / "intel-rapl:0:0"
    sub.mkdir()
    (sub / "name").write_text("core\n")
    (sub / "energy_uj").write_text("500000\n")

    power._prev_cpu.clear()
    assert await power.probe_cpu_power_w(tmp_path) is None  # first read: no delta
    for name, energy in (("intel-rapl:0", 2_000_000), ("intel-rapl:1", 4_000_000)):
        (tmp_path / name / "energy_uj").write_text(f"{energy}\n")
    w = await power.probe_cpu_power_w(tmp_path)
    assert w is not None and w > 0  # 3 J over a very short dt
    power._prev_cpu.clear()


@pytest.mark.asyncio
async def test_rapl_absent_returns_none(tmp_path):
    assert await power.probe_cpu_power_w(tmp_path / "nope") is None


# --------------------------------------------------------------------------
# Fit-check on unified memory
# --------------------------------------------------------------------------

def _est(model_mb: int, kv_mb: int = 1024, compute_mb: int = 1024) -> VramEstimate:
    return VramEstimate(
        total_mb=model_mb + kv_mb + compute_mb, model_mb=model_mb, kv_cache_mb=kv_mb,
        compute_mb=compute_mb, source="computed", details={"n_layers": 32},
    )


def _patch(monkeypatch, est: VramEstimate, profile: dict | None = None):
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: est)
    monkeypatch.setattr(fit_check, "read_model_profile", lambda path: profile or {"n_layers": 32})


def _cfg(**kw) -> LlamaServerConfig:
    return LlamaServerConfig(name="t", model_path="/dev/null/fake.gguf", **kw)


def test_unified_does_not_double_spend_the_pool(monkeypatch):
    """36 GB Mac, 30 GB free: a 24 GB hybrid plan must not be judged as
    "24 GB of VRAM plus 24 GB of RAM" — it is one pool."""
    _patch(monkeypatch, _est(20480))
    r = fit_check.check_fit(
        _cfg(n_gpu_layers=16), gpu_total_mb=27 * 1024, gpu_free_mb=27 * 1024,
        ram_total_mb=36 * 1024, ram_available_mb=30 * 1024, unified=True,
    )
    assert r["hardware"]["unified_memory"] is True
    # GPU share + CPU share are both counted against the same 30 GB.
    assert r["plan"]["gpu_need_mb"] + r["plan"]["ram_need_mb"] <= 30 * 1024
    assert r["level"] in ("fits", "fits_if_alone")
    assert any("one pool" in m["text"] for m in r["messages"])


def test_unified_flags_the_wired_limit(monkeypatch):
    """The model fits in RAM but exceeds what the OS lets the GPU map:
    warn with the actual fix instead of silently OOM-ing at load."""
    _patch(monkeypatch, _est(40 * 1024))
    r = fit_check.check_fit(
        # 64 GB machine, but the OS only lets the GPU wire down 32 GB.
        _cfg(), gpu_total_mb=32 * 1024, gpu_free_mb=32 * 1024,
        ram_total_mb=64 * 1024, ram_available_mb=60 * 1024, unified=True,
    )
    warn = [m["text"] for m in r["messages"] if m["severity"] == "warn"]
    assert any("iogpu.wired_limit_mb" in w for w in warn)


def test_unified_too_big_for_the_pool(monkeypatch):
    """A 90 GB model on a 32 GB Mac: no offload flag can save it."""
    _patch(monkeypatch, _est(90 * 1024))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=24 * 1024, gpu_free_mb=24 * 1024,
        ram_total_mb=32 * 1024, ram_available_mb=28 * 1024, unified=True,
    )
    assert r["level"] == "too_big"


def test_discrete_gpu_math_unchanged(monkeypatch):
    """Regression guard: the unified flag must not leak into normal boxes."""
    _patch(monkeypatch, _est(20480))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=24576, gpu_free_mb=24576,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["hardware"]["unified_memory"] is False
    assert r["level"] == "fits"


# --------------------------------------------------------------------------
# Mixed hardware
# --------------------------------------------------------------------------

def test_amd_igpu_next_to_an_nvidia_card_is_not_unified(monkeypatch):
    """A desktop Ryzen (9950X etc.) exposes an integrated Raphael GPU even
    when an RTX card does the work — the box must not be called unified."""
    monkeypatch.setattr(accel, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(accel.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(accel, "_detect_amd_apu", lambda: (True, "should not be consulted"))
    assert accel.platform_info().unified_memory is False


def test_amd_apu_without_nvidia_is_unified(monkeypatch):
    monkeypatch.setattr(accel, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(accel.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(accel, "_detect_amd_apu", lambda: (True, "AMD integrated GPU"))
    p = accel.platform_info()
    assert p.unified_memory is True and "integrated" in p.detail


def test_toolchain_outside_path_is_still_found(monkeypatch, tmp_path):
    """A systemd unit's PATH has no /usr/local/cuda/bin. Finding nvcc there
    must both enable CUDA and tell cmake (and the build PATH) where it is."""
    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(accel, "_TOOL_DIRS", {"nvcc": ("$CUDA_HOME/bin",)})
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=False, cpu_name="cpu",
    ))
    monkeypatch.setattr(accel.shutil, "which", _fake_which({"nvidia-smi"}))
    assert accel.find_tool("nvcc") == str(nvcc)
    ids = {b.id: b for b in accel.detect_backends()}
    assert ids[accel.CUDA].available
    assert f"-DCMAKE_CUDA_COMPILER={nvcc}" in ids[accel.CUDA].cmake_flags
    assert ids[accel.CUDA].path_prepend == [str(cuda_bin)]
    assert accel.preferred_backend() == accel.CUDA


# --------------------------------------------------------------------------
# Safe-by-default networking
# --------------------------------------------------------------------------

def test_new_presets_bind_localhost_by_default():
    """A llama-server has no auth of its own — a fresh install must not put
    an inference endpoint on the LAN unless the user asks for it."""
    from lld.presets import default_seeds

    assert LlamaServerConfig(name="x").host == "127.0.0.1"
    assert [s.name for s in default_seeds() if s.host == "0.0.0.0"] == []


# --------------------------------------------------------------------------
# amdgpu per-process VRAM (DRM fdinfo) — the reading nvidia-smi has no
# counterpart for on AMD, and without which no Radeon model can be measured.
# --------------------------------------------------------------------------

def _fdinfo(pdev: str, client: str, vram: str, gtt: str = "1024 KiB") -> str:
    return (
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-total-vram:\t{vram}\n"
        f"drm-memory-vram:\t{vram}\n"
        f"drm-memory-gtt: \t{gtt}\n"
    )


def _write_proc(root: Path, pid: int, comm: str, fds: dict[str, str]) -> None:
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "fdinfo").mkdir(parents=True)
    (d / "comm").write_text(f"{comm}\n")
    for name, text in fds.items():
        (d / "fd" / name).symlink_to("/dev/dri/renderD128")
        (d / "fdinfo" / name).write_text(text)


R9700 = vram.AmdCard(index=2, name="Radeon AI PRO R9700", total_mb=32624,
                     used_mb=26732, free_mb=5892, pci_addr="0000:03:00.0")
# No vram_vendor (an APU names no memory maker) and a GTT aperture larger
# than the carve-out — which is what amd_card_is_integrated() looks for.
IGPU = vram.AmdCard(index=3, name="Radeon Graphics", total_mb=3072,
                    used_mb=1922, free_mb=1150, pci_addr="0000:7c:00.0",
                    gtt_total_mb=46000)


def test_amd_process_vram_dedupes_the_descriptors_of_one_client(tmp_path):
    """llama.cpp keeps a dozen descriptors on the render node open and every
    one reports the same allocation — summing them turned a 26 GB model into
    53 GB. Descriptors of one client share drm-client-id."""
    _write_proc(tmp_path, 31304, "llama-server", {
        "42": _fdinfo("0000:03:00.0", "585", "27315284 KiB"),
        "44": _fdinfo("0000:03:00.0", "585", "27315284 KiB"),
        "46": _fdinfo("0000:03:00.0", "585", "27315284 KiB"),
    })
    procs = vram.probe_amd_processes([R9700], proc_root=tmp_path)
    assert len(procs) == 1
    assert procs[0].used_mb == 27315284 // 1024
    assert (procs[0].pid, procs[0].process_name, procs[0].vendor) == (31304, "llama-server", "amd")
    assert procs[0].gpu_index == 2


def test_amd_process_vram_is_split_per_card(tmp_path):
    """A Vulkan-pinned server also opens a context on the iGPU. The two are
    separate rows: what it holds on the Radeon is what stopping it frees
    there, and the iGPU's few MB have nothing to do with that."""
    _write_proc(tmp_path, 31304, "llama-server", {
        "38": _fdinfo("0000:7c:00.0", "584", "28 KiB"),
        "42": _fdinfo("0000:03:00.0", "585", "27315284 KiB"),
    })
    rows = {p.gpu_index: p for p in vram.probe_amd_processes([R9700, IGPU], proc_root=tmp_path)}
    assert rows[2].used_mb == 26675
    assert 3 not in rows  # 28 KiB rounds to 0 MiB and is not worth a row


def test_the_desktop_on_an_igpu_is_not_a_vram_consumer(tmp_path):
    """Every windowed program holds a DRM context on the card that drives the
    display. On an APU that means the whole desktop shows up in a panel whose
    question is "what is holding VRAM on the cards I can offload to" — ten rows
    of gnome-shell, browser and Xwayland burying the one llama-server that
    matters, on a 3 GB carve-out that is not an offload target."""
    _write_proc(tmp_path, 3395, "gnome-shell", {
        "9": _fdinfo("0000:7c:00.0", "12", "986 MiB")})
    _write_proc(tmp_path, 7452, "brave", {
        "9": _fdinfo("0000:7c:00.0", "13", "564 MiB")})
    _write_proc(tmp_path, 300748, "llama-server", {
        "9": _fdinfo("0000:03:00.0", "14", "26661 MiB")})

    rows = vram.probe_amd_processes([R9700, IGPU], proc_root=tmp_path)
    assert [(p.process_name, p.gpu_index) for p in rows] == [("llama-server", 2)]

    # Still reachable on purpose — the reading is real, it just does not
    # belong in that panel.
    everything = vram.probe_amd_processes(
        [R9700, IGPU], proc_root=tmp_path, include_integrated=True
    )
    assert sorted(p.process_name for p in everything) == [
        "brave", "gnome-shell", "llama-server",
    ]


def test_amd_process_vram_ignores_cards_it_cannot_place(tmp_path):
    """A pdev with no matching card (another GPU, or a fixture without a PCI
    address) is skipped rather than attributed to whichever card came first."""
    _write_proc(tmp_path, 100, "brave", {"3": _fdinfo("0000:aa:00.0", "1", "500 MiB")})
    assert vram.probe_amd_processes([R9700], proc_root=tmp_path) == []


def test_amd_process_vram_survives_an_unreadable_process(tmp_path):
    """Another user's process denies /proc/<pid>/fd. That is not an error."""
    _write_proc(tmp_path, 100, "llama-server", {"3": _fdinfo("0000:03:00.0", "7", "1024 MiB")})
    (tmp_path / "999").mkdir()  # no fd/ at all
    procs = vram.probe_amd_processes([R9700], proc_root=tmp_path)
    assert [p.pid for p in procs] == [100]
    assert procs[0].used_mb == 1024


def test_amd_process_probe_is_quiet_without_pci_addresses(tmp_path):
    """A card whose PCI address could not be read cannot be matched to any
    fdinfo row, so the probe says nothing instead of guessing."""
    card = vram.AmdCard(index=0, name="x", total_mb=1, used_mb=0, free_mb=1)
    assert vram.probe_amd_processes([card], proc_root=tmp_path) == []


# --------------------------------------------------------------------------
# End-to-end shape of a machine that is not this one
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_gpus_on_a_mac_falls_through_to_metal(monkeypatch, tmp_path):
    """A Mac has no nvidia-smi and no /sys/class/drm. Both probes must come
    back empty *without* raising, so the Apple fallback is reached — this is
    the whole GPU story on the platform, so a stray exception here leaves the
    dashboard reporting a CPU-only machine."""
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="darwin", arch="arm64", is_apple_silicon=True, unified_memory=True,
        cpu_name="Apple M4 Max",
    ))
    monkeypatch.setattr(vram.shutil, "which", lambda _n: None)   # no nvidia-smi
    monkeypatch.setattr(vram, "amd_sysfs_cards", lambda root=None: [])
    monkeypatch.setattr(accel, "sysctl", lambda key: "")

    class _Mem:
        total = 128 * GB
        available = 100 * GB

    monkeypatch.setattr("psutil.virtual_memory", lambda: _Mem)

    gpus = await vram.probe_gpus()
    assert [g.vendor for g in gpus] == ["apple"]
    # An all-unified machine keeps its only accelerator as an offload target.
    assert vram.offload_gpus(gpus) == gpus


@pytest.mark.asyncio
async def test_power_probes_are_silent_on_a_mac(monkeypatch, tmp_path):
    """macOS has neither RAPL nor amdgpu hwmon, and `powermetrics` needs root.
    The panel is meant to hide itself, which needs None/[] — not a traceback
    out of a 2 Hz background loop."""
    empty = tmp_path / "no-drm"
    empty.mkdir()
    monkeypatch.setattr(power.shutil, "which", lambda _n: None)
    monkeypatch.setattr(power, "_POWERCAP", tmp_path / "no-powercap")
    # The real function, pointed at a tree with no amdgpu nodes in it.
    real_amd = power.probe_amd_gpu_power_w
    monkeypatch.setattr(power, "probe_amd_gpu_power_w", lambda root=None: real_amd(empty))

    assert await power.probe_gpu_power_w() == []
    assert await power.probe_cpu_power_w(tmp_path / "no-powercap") is None
    assert power.get_cpu_power_status() == "unsupported"


@pytest.mark.asyncio
async def test_a_hanging_vendor_tool_does_not_stall_the_gpu_probe(monkeypatch):
    """A half-upgraded NVIDIA driver makes nvidia-smi block in the kernel
    rather than fail. The probe must give up and report no NVIDIA GPU."""
    import sys

    from lld.procutil import run_capture

    monkeypatch.setattr(vram.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(vram, "_NV_TIMEOUT_S", 0.3)

    async def slow(argv, **kw):
        return await run_capture(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=kw["timeout"],
        )

    monkeypatch.setattr(vram, "run_capture", slow)
    monkeypatch.setattr(vram, "probe_amd", lambda root=None: [])
    monkeypatch.setattr(vram, "probe_apple", lambda: [])

    assert await vram.probe_gpus() == []


def test_strix_halo_plans_against_the_whole_unified_pool(tmp_path, monkeypatch):
    """Ryzen AI Max: one iGPU, a token VRAM carve-out and ~96 GB of GTT. The
    budget is the pool, and the card stays an offload target because it is the
    only accelerator the machine has."""
    _write_amd_card(tmp_path, 0, 512 * 1024 * 1024, 0, gtt_bytes=96 * GB, gtt_used=4 * GB)
    monkeypatch.setattr(accel, "platform_info", lambda: accel.PlatformInfo(
        os="linux", arch="x86_64", is_apple_silicon=False, unified_memory=True,
        cpu_name="AMD Ryzen AI Max+ 395",
        detail="AMD integrated GPU (gfx1151) — VRAM is carved out of system RAM",
    ))
    gpus = vram.probe_amd(tmp_path)
    assert gpus[0].integrated is True
    assert gpus[0].total_mb == 512 + 96 * 1024
    assert vram.offload_gpus(gpus) == gpus   # not emptied: it is all there is


@pytest.mark.asyncio
async def test_a_broken_nvidia_driver_is_reported_once_with_its_reason(monkeypatch, caplog):
    """The card vanishing from the dashboard with nothing in the log is the
    failure this replaces. nvidia-smi prints its own errors on STDOUT and exits
    18, so the reason has to be read from there — and the VRAM panel polls
    every few seconds, so it must be said once, not on every tick."""
    from lld.procutil import CommandResult

    monkeypatch.setattr(vram, "_nv_failure", None)
    monkeypatch.setattr(vram.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    async def broken(argv, **kw):
        return CommandResult(
            18, "Failed to initialize NVML: Driver/library version mismatch\n", "",
        )

    monkeypatch.setattr(vram, "run_capture", broken)

    with caplog.at_level("WARNING", logger="lld.vram"):
        assert await vram.probe_nvidia() == []
        assert await vram.probe_nvidia() == []

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"
    assert "Driver/library version mismatch" in warnings[0].getMessage()
    assert vram.nvidia_probe_failure() == (
        "Failed to initialize NVML: Driver/library version mismatch"
    )


@pytest.mark.asyncio
async def test_a_recovered_driver_clears_the_warning(monkeypatch):
    from lld.procutil import CommandResult

    monkeypatch.setattr(vram, "_nv_failure", "something was wrong")
    monkeypatch.setattr(vram.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    async def working(argv, **kw):
        return CommandResult(0, "0, RTX 5090, 32607, 1000, 31607, 0, 40, 30, 25.0, 210\n", "")

    monkeypatch.setattr(vram, "run_capture", working)

    gpus = await vram.probe_nvidia()
    assert [g.name for g in gpus] == ["RTX 5090"]
    assert vram.nvidia_probe_failure() is None


@pytest.mark.asyncio
async def test_no_nvidia_hardware_at_all_is_not_a_failure(monkeypatch):
    """A Mac or an AMD-only box has no nvidia-smi. That is not something to
    warn about — there is simply nothing to probe."""
    monkeypatch.setattr(vram, "_nv_failure", None)
    monkeypatch.setattr(vram.shutil, "which", lambda _n: None)
    assert await vram.probe_nvidia() == []
    assert vram.nvidia_probe_failure() is None
