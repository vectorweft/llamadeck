"""Fit-check level decisions across hardware profiles.

The interesting matrix is GPU presence × offload plan × RAM headroom.
The no-GPU rows model a CPU-only box or an Apple-Silicon/AMD host where
nvidia-smi is absent (gpu_total_mb == 0): there the whole model must be
planned against system RAM, and GPU-side headroom math must not apply.
estimate_vram / read_model_profile are patched — no real GGUF needed.
"""
from __future__ import annotations

from lld import fit_check
from lld.settings import LlamaServerConfig
from lld.vram_estimate import VramEstimate


def _est(model_mb: int, kv_mb: int = 1024, compute_mb: int = 1024) -> VramEstimate:
    return VramEstimate(
        total_mb=model_mb + kv_mb + compute_mb,
        model_mb=model_mb,
        kv_cache_mb=kv_mb,
        compute_mb=compute_mb,
        source="computed",
        details={"n_layers": 32},
    )


def _patch(monkeypatch, est: VramEstimate, profile: dict | None = None):
    monkeypatch.setattr(fit_check, "estimate_vram", lambda cfg: est)
    monkeypatch.setattr(fit_check, "read_model_profile", lambda path: profile or {"n_layers": 32})


def _cfg(**kw) -> LlamaServerConfig:
    return LlamaServerConfig(name="t", model_path="/dev/null/fake.gguf", **kw)


MOE_PROFILE = {
    "n_layers": 48,
    "exps_mb": 20000,
    "n_exp_layers": 48,
    "exps_per_layer_mb": 20000 / 48,
    "expert_count": 128,
    "expert_used_count": 8,
}


# ---- no-GPU machines (gpu_total_mb == 0) -----------------------------------

def test_no_gpu_dense_fits_in_ram(monkeypatch):
    """8 GB dense model, 64 GB RAM box without a GPU: must be 'fits', even
    with the default n_gpu_layers=999 (CPU-only builds ignore -ngl)."""
    _patch(monkeypatch, _est(8192))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=0, gpu_free_mb=0,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["level"] == "fits"
    assert r["plan"]["gpu_need_mb"] == 0
    assert r["plan"]["ram_need_mb"] == 8192 + 1024 + 1024


def test_no_gpu_moe_fits_in_ram(monkeypatch):
    """MoE model on a no-GPU box: must not fall into the 'core part exceeds
    the GPU' (too_big) branch — it runs fine entirely in RAM."""
    _patch(monkeypatch, _est(24576), profile=MOE_PROFILE)
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=0, gpu_free_mb=0,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["level"] == "fits"


def test_no_gpu_model_exceeds_total_ram(monkeypatch):
    """80 GB model on a 32 GB no-GPU box: honest 'too_big', no GPU advice."""
    _patch(monkeypatch, _est(81920))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=0, gpu_free_mb=0,
        ram_total_mb=32768, ram_available_mb=24576,
    )
    assert r["level"] == "too_big"
    assert all(s["id"] != "dense-partial" for s in r["suggestions"])


def test_no_gpu_fits_total_but_not_available_ram(monkeypatch):
    """Fits in total RAM but other apps hold too much right now: level stays
    'fits' with a warn message (remedy = close apps; no offload can help)."""
    _patch(monkeypatch, _est(24576))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=0, gpu_free_mb=0,
        ram_total_mb=32768, ram_available_mb=8192,
    )
    assert r["level"] == "fits"
    assert any(m["severity"] == "warn" for m in r["messages"])


def test_no_gpu_turkish_strings(monkeypatch):
    """The tr catalog must cover every string the no-GPU path emits."""
    _patch(monkeypatch, _est(8192))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=0, gpu_free_mb=0,
        ram_total_mb=65536, ram_available_mb=49152, lang="tr",
    )
    assert r["level"] == "fits"
    assert r["headline"]


# ---- GPU machines, pure-CPU plan (n_gpu_layers=0) ---------------------------

def test_gpu_machine_ngl0_ignores_busy_gpu(monkeypatch):
    """ngl=0 needs no VRAM, so a busy GPU must not block it."""
    _patch(monkeypatch, _est(8192))
    r = fit_check.check_fit(
        _cfg(n_gpu_layers=0), gpu_total_mb=24576, gpu_free_mb=500,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["level"] == "fits"
    assert r["plan"]["gpu_need_mb"] == 0


def test_gpu_machine_ngl0_ram_overflow_suggests_gpu(monkeypatch):
    """ngl=0 but RAM can't hold it while the GPU could: keep the existing
    offload machinery (suggest putting layers back on the GPU)."""
    _patch(monkeypatch, _est(20480))
    r = fit_check.check_fit(
        _cfg(n_gpu_layers=0), gpu_total_mb=24576, gpu_free_mb=24000,
        ram_total_mb=16384, ram_available_mb=12288,
    )
    assert r["level"] in ("needs_offload", "too_big")


# ---- GPU machines, regression guards on existing behavior -------------------

def test_gpu_machine_full_offload_fits(monkeypatch):
    _patch(monkeypatch, _est(8192))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=24576, gpu_free_mb=20000,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["level"] == "fits"


def test_gpu_machine_needs_offload_dense(monkeypatch):
    _patch(monkeypatch, _est(20480))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=12288, gpu_free_mb=11000,
        ram_total_mb=65536, ram_available_mb=49152,
    )
    assert r["level"] == "needs_offload"
    assert any(s["id"] == "dense-partial" for s in r["suggestions"])


def test_gpu_machine_busy_gpu_fits_if_alone(monkeypatch):
    _patch(monkeypatch, _est(8192))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=24576, gpu_free_mb=500,
        ram_total_mb=65536, ram_available_mb=49152,
        gpu_budget_mb=24576,
    )
    assert r["level"] == "fits_if_alone"


def test_needs_offload_headline_names_the_safety_margin():
    """"~30.5 GB needed, 31.4 GB card" reads like it fits — the 2 GB allocator
    margin is what actually decided it, so the message has to say so."""
    from lld.fit_check import HEADROOM_MB, _STRINGS, _gb

    for lang in ("en", "tr"):
        text = _STRINGS[lang]["needs_offload"].format(
            need="30.5", headroom=_gb(HEADROOM_MB), free="30.9", total="31.4",
        )
        assert "30.9" in text and str(_gb(HEADROOM_MB)) in text


# ---- how big the safety margin should be -----------------------------------
# The margin covers two different things, and only one of them survives a
# measurement. These pin the difference on the case that prompted it: a preset
# measured at 31264 MiB against the 32149 MiB CUDA hands out on a 32 GB card.

def _measured(est: VramEstimate) -> VramEstimate:
    est.details["measured"] = True
    return est


def test_a_measured_model_is_budgeted_against_what_it_really_used(monkeypatch):
    """Holding the full 2 GB back here refused a preset that demonstrably runs
    — and a panel that says "won't fit" about a model the user has been using
    all week is a panel they stop reading."""
    _patch(monkeypatch, _measured(_est(29000, kv_mb=1264, compute_mb=1000)))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=32149, gpu_free_mb=32149,
        ram_total_mb=90000, ram_available_mb=70000,
    )
    assert r["level"] == "fits"
    assert r["plan"]["headroom_mb"] == fit_check.MEASURED_HEADROOM_MB
    assert r["plan"]["measured"] is True


def test_the_same_model_unmeasured_keeps_the_full_margin(monkeypatch):
    """Nothing has checked the formula against this model, and on real models
    here it has run 2.4 GB high and 0.8 GB low. That is the margin."""
    _patch(monkeypatch, _est(29000, kv_mb=1264, compute_mb=1000))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=32149, gpu_free_mb=32149,
        ram_total_mb=90000, ram_available_mb=70000,
    )
    assert r["level"] == "needs_offload"
    assert r["plan"]["headroom_mb"] == fit_check.HEADROOM_MB
    assert r["plan"]["measured"] is False


def test_a_measurement_does_not_wave_through_a_card_that_is_too_small(monkeypatch):
    """The smaller margin is not a licence: 31264 MiB still does not fit in
    what is left once another model holds half the card."""
    _patch(monkeypatch, _measured(_est(29000, kv_mb=1264, compute_mb=1000)))
    r = fit_check.check_fit(
        _cfg(), gpu_total_mb=32149, gpu_free_mb=16000,
        ram_total_mb=90000, ram_available_mb=70000, gpu_budget_mb=16000,
    )
    assert r["level"] == "needs_offload"
