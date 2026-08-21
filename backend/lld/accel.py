"""Platform and accelerator detection.

One place that answers "what machine is this, and how should llama.cpp be
built on it?", so the rest of the codebase stops assuming Linux + NVIDIA:

  * `platform_info()` — OS, CPU arch, Apple Silicon, unified memory. Used by
    the VRAM probe, the fit-check and the UI.
  * `detect_backends()` — the llama.cpp compute backends that can plausibly be
    built here (CUDA / Metal / HIP / Vulkan / CPU), each with its cmake flags
    and whether the toolchain is actually installed. Drives the Build page.

Everything is probed lazily and cached: detection shells out to `sysctl` /
`rocminfo`, which we don't want on the hot path. `refresh()` clears the cache
(the Build page calls it when the user hits "re-detect").
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Build-backend ids. "auto" means: let detect pick, and if nothing GPU-ish is
# found, hand cmake no accelerator flags at all (llama.cpp then does the right
# thing per platform — e.g. Metal is ON by default on Apple).
AUTO = "auto"
CUDA = "cuda"
METAL = "metal"
HIP = "hip"
VULKAN = "vulkan"
CPU = "cpu"


@dataclass
class PlatformInfo:
    os: str  # "linux" | "darwin" | "windows"
    arch: str  # "x86_64" | "arm64" | ...
    is_apple_silicon: bool
    # True when the GPU and the CPU share one pool of physical memory
    # (Apple Silicon, AMD APUs like Strix Halo / Ryzen AI Max). VRAM numbers
    # then come out of system RAM instead of being an independent budget.
    unified_memory: bool
    cpu_name: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "os": self.os,
            "arch": self.arch,
            "is_apple_silicon": self.is_apple_silicon,
            "unified_memory": self.unified_memory,
            "cpu_name": self.cpu_name,
            "detail": self.detail,
        }


@dataclass
class Backend:
    id: str
    label: str
    # cmake flags appended to the configure step
    cmake_flags: list[str] = field(default_factory=list)
    # directories to prepend to PATH for the build subprocesses (a toolchain
    # found outside PATH, e.g. /usr/local/cuda/bin under a systemd unit)
    path_prepend: list[str] = field(default_factory=list)
    # toolchain present on this machine?
    available: bool = False
    # short human explanation (why unavailable / what was detected)
    detail: str = ""
    # can this backend even exist on this OS/arch?
    supported: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "cmake_flags": self.cmake_flags,
            "path_prepend": self.path_prepend,
            "available": self.available,
            "detail": self.detail,
            "supported": self.supported,
        }


_platform_cache: PlatformInfo | None = None
_backends_cache: list[Backend] | None = None


def refresh() -> None:
    """Drop cached probes (toolchains can be installed while LlamaDeck runs)."""
    global _platform_cache, _backends_cache
    _platform_cache = None
    _backends_cache = None


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    """Best-effort command capture — empty string on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key]).strip()


def platform_info() -> PlatformInfo:
    global _platform_cache
    if _platform_cache is not None:
        return _platform_cache

    system = platform.system().lower()
    os_name = {"darwin": "darwin", "windows": "windows"}.get(system, "linux" if system == "linux" else system)
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    is_apple_silicon = os_name == "darwin" and arch == "arm64"

    cpu_name = platform.processor() or arch
    detail = ""
    unified = False

    if os_name == "darwin":
        brand = sysctl("machdep.cpu.brand_string")
        if brand:
            cpu_name = brand
        # Every Apple Silicon Mac is unified memory. Intel Macs with a
        # discrete GPU are not, and llama.cpp barely targets them any more.
        unified = is_apple_silicon
        if unified:
            detail = "Apple Silicon — GPU and CPU share one memory pool"
    elif os_name == "linux":
        model = _read_cpuinfo_model()
        if model:
            cpu_name = model
        # Desktop AMD CPUs ship an iGPU (Raphael/Granite Ridge) that would
        # otherwise make every Ryzen box look "unified" — even a 9950X with an
        # RTX 5090 in it. A discrete NVIDIA card is what actually runs the
        # model there, so it wins.
        if not has_nvidia_gpu():
            unified, apu_detail = _detect_amd_apu()
            if unified:
                detail = apu_detail

    _platform_cache = PlatformInfo(
        os=os_name,
        arch=arch,
        is_apple_silicon=is_apple_silicon,
        unified_memory=unified,
        cpu_name=cpu_name,
        detail=detail,
    )
    return _platform_cache


def has_nvidia_gpu() -> bool:
    """True when an NVIDIA GPU is present (driver node, or nvidia-smi on PATH)."""
    if Path("/proc/driver/nvidia/gpus").is_dir():
        try:
            return any(Path("/proc/driver/nvidia/gpus").iterdir())
        except OSError:
            return False
    return bool(shutil.which("nvidia-smi"))


def _read_cpuinfo_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


# AMD APUs whose iGPU carves its VRAM out of system RAM. gfx1151 is Strix
# Halo (Ryzen AI Max 385/390/395), gfx1150 Strix Point, gfx1103 Phoenix/Hawk
# Point, gfx1036/gfx1037 Raphael/Granite Ridge desktop graphics.
_APU_GFX_PREFIXES = (
    "gfx90c",    # Cezanne / Renoir
    "gfx1036",   # Raphael
    "gfx1037",   # Granite Ridge
    "gfx1103",   # Phoenix / Hawk Point
    "gfx1150",   # Strix Point
    "gfx1151",   # Strix Halo — Ryzen AI Max 385/390/395
    "gfx1152",   # Krackan Point
)


def _detect_amd_apu() -> tuple[bool, str]:
    """(is_unified, detail) for AMD integrated graphics on Linux.

    Signal used: the amdgpu driver marks integrated parts as `local_memory`
    carved from system RAM. We can't read that flag directly from sysfs, so we
    go by the GPU's gfx target when ROCm is installed, and otherwise fall back
    to `amd_card_is_integrated` — which keys on the card having no dedicated
    memory chips to name.

    `platform_info` only calls this when no NVIDIA GPU is present, but the
    check must stand on its own regardless: an AMD-only box with a discrete
    Radeon has no such gate to hide behind.
    """
    target = amd_gfx_target()
    if target and target.startswith(_APU_GFX_PREFIXES):
        return True, f"AMD integrated GPU ({target}) — VRAM is carved out of system RAM"
    # local import: vram imports accel
    from .vram import amd_card_is_integrated, amd_sysfs_cards

    cards = amd_sysfs_cards()
    # A discrete Radeon alongside the iGPU wins, exactly as a discrete NVIDIA
    # card does in `platform_info`: the machine has real VRAM to plan against,
    # so it is not a unified-memory box even though an iGPU is present.
    if any(not amd_card_is_integrated(c) for c in cards):
        return False, ""
    if cards:
        return True, "AMD integrated GPU — VRAM is carved out of system RAM"
    return False, ""


def amd_gfx_target() -> str:
    """The first ROCm gfx target on this machine ("gfx1151"), or "" if ROCm
    isn't installed. Used both for APU detection and to pin -DAMDGPU_TARGETS
    so a HIP build doesn't compile every architecture."""
    if not shutil.which("rocminfo"):
        return ""
    for line in _run(["rocminfo"], timeout=15.0).splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            return line.split(":", 1)[1].strip()
    return ""


def _has(*names: str) -> str:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return ""


# Toolchains routinely live outside a service's PATH: a systemd unit inherits
# a bare PATH, so `nvcc` in /usr/local/cuda/bin is invisible even though the
# same machine builds fine from an interactive shell. Look in the standard
# install locations too, and hand cmake the absolute compiler when we had to.
_TOOL_DIRS: dict[str, tuple[str, ...]] = {
    "nvcc": ("$CUDA_HOME/bin", "$CUDA_PATH/bin", "/usr/local/cuda/bin",
             "/opt/cuda/bin", "/usr/lib/nvidia-cuda-toolkit/bin"),
    "hipcc": ("$ROCM_PATH/bin", "$HIP_PATH/bin", "/opt/rocm/bin"),
    "rocminfo": ("$ROCM_PATH/bin", "/opt/rocm/bin"),
    "glslc": ("$VULKAN_SDK/bin",),
}


def find_tool(name: str) -> str:
    """Absolute path to a build tool: PATH first, then the usual install dirs."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    import os

    for raw in _TOOL_DIRS.get(name, ()):
        if raw.startswith("$"):
            var, _, rest = raw[1:].partition("/")
            root = os.environ.get(var)
            if not root:
                continue
            candidate = Path(root) / rest / name
        else:
            candidate = Path(raw) / name
        if candidate.is_file():
            return str(candidate)
    return ""


def _off_path(tool_path: str) -> bool:
    return bool(tool_path) and shutil.which(Path(tool_path).name) is None


def detect_backends() -> list[Backend]:
    """Every llama.cpp backend LlamaDeck can drive, ordered best-first for
    this machine. Unsupported-on-this-OS backends are still returned (with
    supported=False) so the UI can explain rather than silently hide them."""
    global _backends_cache
    if _backends_cache is not None:
        return _backends_cache

    p = platform_info()
    out: list[Backend] = []

    # --- CUDA (NVIDIA) ----------------------------------------------------
    nvcc = find_tool("nvcc")
    nvsmi = _has("nvidia-smi") or has_nvidia_gpu()
    cuda_supported = p.os in ("linux", "windows")
    cuda_flags = ["-DGGML_CUDA=ON"]
    cuda_dirs: list[str] = []
    if _off_path(nvcc):
        cuda_flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        cuda_dirs.append(str(Path(nvcc).parent))
    out.append(Backend(
        id=CUDA,
        label="CUDA (NVIDIA)",
        cmake_flags=cuda_flags,
        path_prepend=cuda_dirs,
        supported=cuda_supported,
        available=bool(cuda_supported and nvcc),
        detail=(
            (f"nvcc: {nvcc}" if _off_path(nvcc) else "") if nvcc else
            "NVIDIA driver found but the CUDA toolkit (nvcc) is not installed"
            if nvsmi else "no NVIDIA toolchain found"
        ) if cuda_supported else "NVIDIA GPUs are not available on macOS",
    ))

    # --- Metal (Apple) ----------------------------------------------------
    metal_supported = p.os == "darwin"
    out.append(Backend(
        id=METAL,
        label="Metal (Apple)",
        cmake_flags=["-DGGML_METAL=ON"],
        supported=metal_supported,
        available=metal_supported and bool(_has("xcrun")),
        detail=(
            "" if _has("xcrun") else "Xcode command line tools missing — run: xcode-select --install"
        ) if metal_supported else "Metal only exists on macOS",
    ))

    # --- HIP / ROCm (AMD) -------------------------------------------------
    hip_supported = p.os in ("linux", "windows")
    hipcc = find_tool("hipcc")
    rocm_dir = Path("/opt/rocm").exists()
    gfx = amd_gfx_target() if hip_supported else ""
    hip_flags = ["-DGGML_HIP=ON"]
    hip_dirs: list[str] = []
    if gfx:
        # Without this, cmake compiles for every AMD arch it knows — minutes
        # of extra build time, and on some ROCm versions an outright failure.
        hip_flags.append(f"-DAMDGPU_TARGETS={gfx}")
    if _off_path(hipcc):
        hip_flags.append(f"-DCMAKE_HIP_COMPILER={hipcc}")
        hip_dirs.append(str(Path(hipcc).parent))
    out.append(Backend(
        id=HIP,
        label="HIP / ROCm (AMD)",
        cmake_flags=hip_flags,
        path_prepend=hip_dirs,
        supported=hip_supported,
        available=bool(hip_supported and hipcc),
        detail=(
            (f"ROCm target {gfx}" if gfx else "ROCm found") if hipcc else
            "ROCm is installed but hipcc was not found under /opt/rocm/bin"
            if rocm_dir else "ROCm is not installed"
        ) if hip_supported else "ROCm does not support macOS",
    ))

    # --- Vulkan (vendor-neutral GPU) --------------------------------------
    glslc = find_tool("glslc")
    vk_supported = p.os in ("linux", "windows")
    vk_flags = ["-DGGML_VULKAN=ON"]
    vk_dirs: list[str] = []
    if _off_path(glslc):
        vk_flags.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")
        vk_dirs.append(str(Path(glslc).parent))
    out.append(Backend(
        id=VULKAN,
        label="Vulkan (any GPU)",
        cmake_flags=vk_flags,
        path_prepend=vk_dirs,
        supported=vk_supported,
        available=bool(vk_supported and glslc),
        detail=(
            "" if glslc else "needs the Vulkan SDK + glslc (Debian/Ubuntu: libvulkan-dev glslc)"
        ) if vk_supported else "use Metal on macOS",
    ))

    # --- CPU only ---------------------------------------------------------
    out.append(Backend(
        id=CPU,
        label="CPU only",
        # On Apple, Metal is ON by default — an explicit OFF is the only way
        # to actually get a CPU-only binary.
        cmake_flags=["-DGGML_METAL=OFF"] if p.os == "darwin" else [],
        available=True,
        detail="always works; no GPU offload",
    ))

    _backends_cache = out
    return out


def preferred_backend() -> str:
    """What `auto` resolves to on this machine."""
    for b in detect_backends():
        if b.id != CPU and b.available:
            return b.id
    return CPU


def resolve(backend: str | None) -> tuple[str, list[str], list[str]]:
    """(resolved_id, cmake_flags, path_prepend) for a requested backend id.

    Unknown ids fall back to auto rather than failing the build — the value
    can come from an old client or a hand-written API call.
    """
    wanted = (backend or AUTO).lower()
    if wanted == AUTO:
        wanted = preferred_backend()
    for b in detect_backends():
        if b.id == wanted:
            return b.id, list(b.cmake_flags), list(b.path_prepend)
    log.warning("unknown build backend %r — falling back to auto", backend)
    resolved = preferred_backend()
    for b in detect_backends():
        if b.id == resolved:
            return b.id, list(b.cmake_flags), list(b.path_prepend)
    return CPU, [], []
