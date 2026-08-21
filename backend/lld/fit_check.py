"""Fit-check — answers "will this preset run on this machine?".

vram_estimate.py computes the total memory need; this module compares it
against the hardware (GPU VRAM + system RAM), models the scenario of moving
MoE expert tensors to CPU/RAM, and produces actionable end-user suggestions:

  * Does the model fit in VRAM? Right now, or only once the GPU is freed?
  * If MoE: how many expert layers should move to RAM (--n-cpu-moe N)?
  * If dense: what should -ngl drop to?
  * If RAM is not enough either: a clear "won't run on this machine" warning.

The expert/core tensor split is read from the GGUF tensor table (header only —
no tensor data loaded) and cached by (path, mtime).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .settings import LlamaServerConfig
from .vram_estimate import (
    estimate_vram,
    missing_shards,
    parse_cpu_moe_offload,
    read_model_profile,
    rival_split_prefix,
    split_shards,
)

log = logging.getLogger(__name__)

# Safety margin for a model nobody has measured yet. It covers two different
# things: the allocator/driver slack that always claims GPU space, and the
# estimator's own error — which on real models here has run from 2.4 GB high
# (DeepSeek-V4 MLA) to 0.8 GB low (Qwen3.8-27B), so it is the larger half.
HEADROOM_MB = 2048
# Once a model has actually run on this kind of card, its memory number is a
# measurement rather than a formula, and only the allocator half of that
# margin is still real: the driver's own working set, fragmentation, and the
# fact that llama.cpp asks for large contiguous blocks (a 0.9 GB projector in
# one piece, in the case that prompted this). Keeping the full 2 GB there
# refused presets the user had already run — which teaches them to ignore the
# panel, and that is worse than being slightly bold.
MEASURED_HEADROOM_MB = 512
# RAM-side safety margin left for the OS + other applications.
RAM_SAFETY = 0.92


# ---- Message catalog -------------------------------------------------------
# User-facing strings in both languages; `lang` comes from settings.ui_language
# (an unknown value falls back to English). API error details stay English by
# design — only fit-check output is bilingual.
_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "ctx_exceeds": "ctx_size ({ctx:,}) exceeds the model's supported maximum ({max:,}) — the server will fail at startup. Lower it to {max:,} or less.",
        "split_missing": "Incomplete model: this is a {count}-part GGUF and {n} part(s) are missing, starting with {first}.",
        "split_missing_hint": "llama.cpp derives the other parts' names from part 1's filename, so every part must sit in the same folder under the same prefix. Download the missing part, or rename part 1 to match the others.",
        "split_rival": "The same folder holds a more complete set named \"{rival}\" — the parts come from two different releases. Use one release's files throughout; mixing them loads without complaint and then produces garbage.",
        "kv_q8_label": "Switch KV cache to q8_0 (halves it)",
        "kv_q8_expl": "At this context length the KV cache takes ~{kv} GB. q8_0 halves that with no quality loss.",
        "ram_warn": "The portion moving to RAM ({need} GB) is larger than currently free RAM ({avail} GB). Close other applications, or the model will spill to disk and get VERY slow.",
        "ram_err": "The portion moving to RAM ({need} GB) exceeds total RAM ({total} GB) — with these settings the model will not run properly on this machine. Download a smaller (more compressed) version.",
        "fits_hybrid": "Fits (hybrid): GPU ~{gpu} GB + RAM ~{ram} GB",
        "hybrid_info": "Part of the model will run in RAM — token speed will be lower than a fully-GPU model; this is normal.",
        "fits": "Fits: within GPU memory (~{need} GB / {free} GB free)",
        "fits_if_alone": "Fits, but the GPU is busy right now: ~{need} GB needed, {free} GB free",
        "gpu_busy_warn": "Another model is using the GPU right now. Stop the running model (Server page) before starting this preset, or it will hit an out-of-memory error at startup.",
        "needs_offload": "Doesn't fit with these settings: ~{need} GB needed plus a {headroom} GB safety margin, and {free} GB is free on a {total} GB card.",
        "core_too_big": "Doesn't fit: even with experts in RAM, the core part (~{fixed} GB) exceeds the GPU",
        "core_too_big_hint": "Try lowering the context length (ctx_size) and switching the KV cache to q8_0; if it still doesn't fit, a smaller model version is needed.",
        "too_big_ram": "Doesn't fit: the model exceeds VRAM + RAM combined (~{cpu} GB needed in RAM, total RAM {total} GB)",
        "too_big_hint": "This version is too big for this machine. Download a smaller / more compressed GGUF.",
        "moe_desc": "MoE with {n} experts",
        "moe_desc_active": ", {m} active per request",
        "moe_label": "Move expert layers to RAM ({flags})",
        "moe_expl": "This model is {desc} — expert tensors ({exps} GB) make up most of the model. Moving {ncpu}/{ntot} expert layers to RAM uses ~{gpu} GB GPU and ~{ram} GB RAM, and the model runs. It gets slower, but stays usable because MoE only computes the active experts.",
        "moe_ram_warn": "Even with the suggestion applied, currently free RAM ({avail} GB) is below the needed {need} GB — close other applications/models before starting.",
        "dense_label": "Move some layers to RAM (n_gpu_layers={ngl})",
        "dense_expl": "{ngl} of the model's {layers} layers stay on the GPU, the rest runs in RAM (~{ram} GB). In dense models every layer in RAM noticeably cuts speed — prefer a smaller version if possible.",
        "too_big2": "Doesn't fit: the model exceeds VRAM + RAM combined",
        "fits_cpu": "Fits in system RAM: ~{ram} GB needed / {avail} GB free",
        "no_gpu_info": "No GPU memory telemetry on this machine — the model is planned entirely in system RAM. CPU-only builds ignore -ngl.",
        "cpu_info": "n_gpu_layers=0 — the whole model runs on the CPU; token speed depends on CPU cores and memory bandwidth.",
        "unified_info": "Unified memory: the GPU and the CPU share the same RAM, so GPU and CPU shares are counted against one pool of {total} GB. Offloading layers to the CPU frees no memory here — it only changes who computes them.",
        "unified_cap_warn": "The GPU-resident part (~{need} GB) is above what this system lets the GPU map ({cap} GB). On macOS raise it with `sudo sysctl iogpu.wired_limit_mb={mb}`; on an AMD APU raise the UMA/VRAM share in the BIOS — or lower n_gpu_layers.",
    },
    "tr": {
        "ctx_exceeds": "ctx_size ({ctx:,}) modelin desteklediği üst sınırın ({max:,}) üzerinde — sunucu açılışta hata verir. Değeri {max:,} veya altına indirin.",
        "split_missing": "Model eksik: bu {count} parçalı bir GGUF ve {n} parça yok, ilki {first}.",
        "split_missing_hint": "llama.cpp diğer parçaların adını 1. parçanın adından türetir; bu yüzden tüm parçalar aynı klasörde aynı önekle durmalı. Eksik parçayı indirin ya da 1. parçanın adını diğerlerine uyacak şekilde düzeltin.",
        "split_rival": "Aynı klasörde \"{rival}\" adıyla daha eksiksiz bir set var — parçalar iki farklı sürümden geliyor. Tek bir sürümün dosyalarını kullanın; karıştırılmış set sorunsuz yüklenir ama saçma çıktı üretir.",
        "kv_q8_label": "KV cache'i q8_0 yap (yarıya iner)",
        "kv_q8_expl": "Bu bağlam uzunluğunda KV cache ~{kv} GB tutuyor. q8_0 kalite kaybı olmadan bunu yarıya indirir.",
        "ram_warn": "RAM'e taşınacak kısım ({need} GB) şu an boş olan RAM'den ({avail} GB) büyük. Diğer uygulamaları kapatın, yoksa model disk'e taşar ve ÇOK yavaşlar.",
        "ram_err": "RAM'e taşınacak kısım ({need} GB) toplam RAM'i ({total} GB) aşıyor — bu ayarlarla model bu makinede sağlıklı çalışmaz. Daha küçük (daha çok sıkıştırılmış) bir sürüm indirin.",
        "fits_hybrid": "Uygun (karma): GPU ~{gpu} GB + RAM ~{ram} GB",
        "hybrid_info": "Modelin bir kısmı RAM'de çalışacak — token hızı tamamen GPU'da çalışan bir modele göre düşük olur; bu normaldir.",
        "fits": "Uygun: ekran kartına sığıyor (~{need} GB / {free} GB boş)",
        "fits_if_alone": "Sığar ama şu an GPU dolu: ~{need} GB gerekli, boş olan {free} GB",
        "gpu_busy_warn": "GPU'yu şu an başka bir model kullanıyor. Bu preset'i başlatmadan önce çalışan modeli durdurun (Server sayfası), yoksa açılışta bellek hatası alırsınız.",
        "needs_offload": "Bu ayarlarla sığmıyor: ~{need} GB + {headroom} GB güvenlik payı gerekiyor, boş olan {free} GB ({total} GB kart).",
        "core_too_big": "Sığmıyor: uzmanlar RAM'e taşınsa bile çekirdek kısım (~{fixed} GB) ekran kartını aşıyor",
        "core_too_big_hint": "Bağlam uzunluğunu (ctx_size) düşürmeyi ve KV cache'i q8_0 yapmayı deneyin; yine sığmazsa daha küçük bir model sürümü gerekir.",
        "too_big_ram": "Sığmıyor: model VRAM + RAM toplamını aşıyor (RAM'e ~{cpu} GB gerekir, toplam RAM {total} GB)",
        "too_big_hint": "Bu sürüm bu makine için çok büyük. Daha küçük/daha çok sıkıştırılmış bir GGUF indirin.",
        "moe_desc": "{n} uzmanlı MoE",
        "moe_desc_active": ", istek başına {m} aktif",
        "moe_label": "Uzman katmanlarını RAM'e taşı ({flags})",
        "moe_expl": "Bu model {desc} — uzman tensörleri ({exps} GB) modelin büyük kısmı. {ncpu}/{ntot} uzman katmanı RAM'e taşınırsa GPU ~{gpu} GB, RAM ~{ram} GB kullanır ve model çalışır. Hız düşer ama MoE'de yalnız aktif uzmanlar hesaplandığı için kullanılabilir kalır.",
        "moe_ram_warn": "Öneri uygulansa da şu an boş RAM ({avail} GB) gereken {need} GB'ın altında — başlatmadan önce diğer uygulamaları/modelleri kapatın.",
        "dense_label": "Katmanların bir kısmını RAM'e taşı (n_gpu_layers={ngl})",
        "dense_expl": "Modelin {layers} katmanından {ngl} tanesi GPU'da kalır, kalanı RAM'de çalışır (~{ram} GB). Yoğun (dense) modellerde RAM'deki her katman hızı belirgin düşürür — mümkünse daha küçük bir sürüm tercih edin.",
        "too_big2": "Sığmıyor: model VRAM + RAM toplamını aşıyor",
        "fits_cpu": "Sistem RAM'ine sığıyor: ~{ram} GB gerekli / {avail} GB boş",
        "no_gpu_info": "Bu makinede GPU bellek telemetrisi yok — model tamamen sistem RAM'ine göre planlandı. CPU-only derlemeler -ngl'yi yok sayar.",
        "cpu_info": "n_gpu_layers=0 — modelin tamamı CPU'da çalışır; token hızı CPU çekirdeklerine ve bellek bant genişliğine bağlıdır.",
        "unified_info": "Ortak bellek (unified memory): GPU ve CPU aynı RAM'i paylaşır, bu yüzden GPU ve CPU payları tek bir {total} GB'lık havuzdan sayılır. Katmanları CPU'ya taşımak burada bellek kazandırmaz — sadece hesabı kimin yaptığını değiştirir.",
        "unified_cap_warn": "GPU'da kalacak kısım (~{need} GB), sistemin GPU'ya izin verdiği sınırın ({cap} GB) üzerinde. macOS'ta `sudo sysctl iogpu.wired_limit_mb={mb}` ile yükseltin; AMD APU'da BIOS'tan UMA/VRAM payını artırın — ya da n_gpu_layers'ı düşürün.",
    },
}

def _gb(mb: float) -> float:
    return round(mb / 1024, 1)


def check_fit(
    cfg: LlamaServerConfig,
    gpu_total_mb: int,
    gpu_free_mb: int,
    ram_total_mb: int,
    ram_available_mb: int,
    gpu_budget_mb: int | None = None,
    lang: str = "en",
    unified: bool = False,
) -> dict[str, Any]:
    """Config + hardware → level, messages and one-click applicable suggestions.

    level: "fits" | "fits_if_alone" | "needs_offload" | "too_big" | "unknown"
    Suggestions carry patches the frontend can apply directly:
      {"add_flags": [...], "remove_flags": [...], "set": {field: value}}

    gpu_budget_mb: suggestions are sized against this budget. Pass total VRAM
    when another (stoppable) LLM is running, or the actually-free amount when
    the GPU is already idle — the desktop/compositor's permanent share never
    comes back from the total. None → gpu_total_mb.

    unified: the GPU draws from system RAM (Apple Silicon, AMD APUs such as
    the Ryzen AI Max). VRAM and RAM are then the same physical bytes, so the
    two budgets must not be spent twice — see below.
    """
    S = _STRINGS["tr" if lang == "tr" else "en"]
    if gpu_budget_mb is None:
        gpu_budget_mb = gpu_total_mb
    if unified:
        # The GPU can never hold more than the shared pool currently has, no
        # matter what the driver advertises as "VRAM".
        gpu_free_mb = min(gpu_free_mb, ram_available_mb)
        gpu_budget_mb = min(gpu_budget_mb, ram_total_mb)
    est = estimate_vram(cfg)
    if est is None:
        return {"available": False, "level": "unknown", "messages": [], "suggestions": []}

    # A split model whose siblings can't be found never gets far enough for
    # memory to matter: llama.cpp aborts during load and the user sees a bare
    # "exited with code 1". Say which file is missing instead.
    shards = split_shards(cfg.model_path)
    # Single-file models need no such check — estimate_vram already bailed out
    # above if the one path we were given isn't there.
    absent = missing_shards(cfg.model_path) if len(shards) > 1 else []
    if absent:
        messages = [{"severity": "error", "text": S["split_missing_hint"]}]
        rival = rival_split_prefix(cfg.model_path)
        if rival:
            messages.append({"severity": "error", "text": S["split_rival"].format(rival=rival)})
        return {
            "available": True,
            "level": "broken",
            "headline": S["split_missing"].format(
                count=len(shards), n=len(absent), first=absent[0].rsplit("/", 1)[-1]
            ),
            "estimate": est.to_dict(),
            "messages": messages,
            "suggestions": [],
        }

    profile = read_model_profile(cfg.model_path) or {}
    exps_mb = int(profile.get("exps_mb") or 0)
    n_exp_layers = int(profile.get("n_exp_layers") or 0)
    per_layer_mb = float(profile.get("exps_per_layer_mb") or 0.0)
    is_moe = exps_mb > 0 and n_exp_layers > 0
    n_layers = int(profile.get("n_layers") or est.details.get("n_layers") or 0)

    model_mb = est.model_mb
    kv_mb = est.kv_cache_mb
    compute_mb = est.compute_mb

    # --- GPU/RAM split of the current config -------------------------------
    cpu_moe_layers = parse_cpu_moe_offload(cfg, n_exp_layers) if is_moe else 0
    ngl = cfg.n_gpu_layers

    # No GPU telemetry (gpu_total_mb == 0): a CPU-only box, or an Apple
    # Silicon / non-NVIDIA host where VRAM is unified or invisible to us.
    # GPU-side budgeting is meaningless there — plan the whole model against
    # system RAM regardless of -ngl (CPU-only builds ignore the flag, and
    # Metal draws from the same unified pool).
    no_gpu = gpu_total_mb <= 0
    if no_gpu or ngl == 0:
        gpu_need_mb = 0
        ram_need_mb = model_mb + kv_mb + compute_mb
    elif cpu_moe_layers > 0:
        cpu_w = int(cpu_moe_layers * per_layer_mb)
        gpu_need_mb = (model_mb - cpu_w) + kv_mb + compute_mb
        ram_need_mb = cpu_w
    elif n_layers and 0 < ngl < n_layers:
        gpu_w = int(model_mb * ngl / n_layers)
        gpu_need_mb = gpu_w + kv_mb + compute_mb
        ram_need_mb = model_mb - gpu_w
    else:
        gpu_need_mb = model_mb + kv_mb + compute_mb
        ram_need_mb = 0

    # Shift by what this model actually measured last time it ran. Without it
    # the formula runs high on some architectures and refuses to start a preset
    # the user has already had running — which teaches them to ignore the panel.
    calibration_mb = int(est.details.get("calibration_mb") or 0)
    if calibration_mb and gpu_need_mb > 0:
        gpu_need_mb = max(0, gpu_need_mb - calibration_mb)

    # A measured plan earns the smaller margin (see MEASURED_HEADROOM_MB).
    headroom_mb = MEASURED_HEADROOM_MB if est.details.get("measured") else HEADROOM_MB

    messages: list[dict[str, str]] = []
    suggestions: list[dict[str, Any]] = []

    # --- context length check ----------------------------------------------
    ctx_len = profile.get("context_length")
    if ctx_len and cfg.ctx_size > ctx_len:
        messages.append({
            "severity": "error",
            "text": S["ctx_exceeds"].format(ctx=cfg.ctx_size, max=ctx_len),
        })

    # --- KV cache suggestion -------------------------------------------------
    if kv_mb > 3072 and (cfg.cache_type_k or "f16").lower() in ("f16", "bf16", "f32"):
        suggestions.append({
            "id": "kv-q8",
            "label": S["kv_q8_label"],
            "explanation": S["kv_q8_expl"].format(kv=_gb(kv_mb)),
            "add_flags": [], "remove_flags": [],
            "set": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        })

    # --- level decision -------------------------------------------------------
    # A pure-CPU plan needs no VRAM, so GPU headroom/busy checks don't apply.
    cpu_plan = gpu_need_mb == 0
    fits_now = cpu_plan or gpu_need_mb + headroom_mb <= gpu_free_mb
    # "Fits if freed": does it fit once stoppable LLMs' VRAM is reclaimed
    # (gpu_budget_mb)? The desktop's permanent share is already off-budget.
    fits_alone = cpu_plan or gpu_need_mb + headroom_mb <= gpu_budget_mb
    # On unified memory the GPU share comes out of the very same RAM, so the
    # CPU share only gets what is left after it.
    ram_avail_eff = max(0, ram_available_mb - gpu_need_mb) if unified else ram_available_mb
    ram_total_eff = max(0, ram_total_mb - gpu_need_mb) if unified else ram_total_mb
    ram_ok = ram_need_mb <= ram_avail_eff * RAM_SAFETY
    ram_ok_total = ram_need_mb <= ram_total_eff * RAM_SAFETY

    hybrid = ram_need_mb > 0
    if hybrid:
        if ram_ok:
            pass  # RAM side is fine
        elif ram_ok_total:
            messages.append({
                "severity": "warn",
                "text": S["ram_warn"].format(need=_gb(ram_need_mb), avail=_gb(ram_available_mb)),
            })
        else:
            messages.append({
                "severity": "error",
                "text": S["ram_err"].format(need=_gb(ram_need_mb), total=_gb(ram_total_mb)),
            })

    # For a pure-CPU plan the only remedy for tight-but-sufficient RAM is
    # closing other apps — the ram_warn message already says exactly that,
    # so keep the level at "fits" instead of derailing into offload advice.
    if fits_now and (not hybrid or ram_ok or (cpu_plan and ram_ok_total)):
        level = "fits"
        if cpu_plan:
            headline = S["fits_cpu"].format(ram=_gb(ram_need_mb), avail=_gb(ram_available_mb))
            messages.append({
                "severity": "info",
                "text": S["no_gpu_info"] if no_gpu else S["cpu_info"],
            })
        elif hybrid:
            headline = S["fits_hybrid"].format(gpu=_gb(gpu_need_mb), ram=_gb(ram_need_mb))
            messages.append({
                "severity": "info",
                "text": S["hybrid_info"],
            })
        else:
            headline = S["fits"].format(need=_gb(gpu_need_mb), free=_gb(gpu_free_mb))
    elif fits_alone and (not hybrid or ram_ok):
        level = "fits_if_alone"
        headline = S["fits_if_alone"].format(need=_gb(gpu_need_mb), free=_gb(gpu_free_mb))
        messages.append({
            "severity": "warn",
            "text": S["gpu_busy_warn"],
        })
    elif no_gpu:
        # No GPU to offload to and RAM can't hold it — no flag can fix this.
        level = "too_big"
        headline = S["too_big_ram"].format(cpu=_gb(ram_need_mb), total=_gb(ram_total_mb))
        messages.append({
            "severity": "error",
            "text": S["too_big_hint"],
        })
    else:
        # Doesn't fit with these settings — produce a fix.
        level = "needs_offload"
        # Naming the margin matters: "30.5 needed, 31.4 GB card" reads like it
        # fits, and the user is left thinking the check is broken.
        headline = S["needs_offload"].format(
            need=_gb(gpu_need_mb),
            headroom=_gb(headroom_mb),
            free=_gb(gpu_free_mb),
            total=_gb(gpu_total_mb),
        )

        if is_moe:
            # Everything except the experts should stay on the GPU (for speed).
            fixed_mb = (model_mb - exps_mb) + kv_mb + compute_mb
            expert_budget = gpu_budget_mb - headroom_mb - fixed_mb
            gpu_exp_layers = max(0, int(expert_budget // per_layer_mb)) if per_layer_mb else 0
            gpu_exp_layers = min(gpu_exp_layers, n_exp_layers)
            n_cpu = n_exp_layers - gpu_exp_layers
            cpu_mb = int(n_cpu * per_layer_mb)
            new_gpu_mb = fixed_mb + int(gpu_exp_layers * per_layer_mb)

            if fixed_mb + headroom_mb > gpu_budget_mb:
                # Even with all experts on CPU the core part doesn't fit.
                level = "too_big"
                headline = S["core_too_big"].format(fixed=_gb(fixed_mb))
                messages.append({
                    "severity": "error",
                    "text": S["core_too_big_hint"],
                })
            elif not (cpu_mb <= ram_total_mb * RAM_SAFETY):
                level = "too_big"
                headline = S["too_big_ram"].format(cpu=_gb(cpu_mb), total=_gb(ram_total_mb))
                messages.append({
                    "severity": "error",
                    "text": S["too_big_hint"],
                })
            else:
                exp_used = profile.get("expert_used_count")
                exp_total = profile.get("expert_count")
                moe_desc = S["moe_desc"].format(n=exp_total) + (S["moe_desc_active"].format(m=exp_used) if exp_used else "")
                flag = ["--cpu-moe"] if n_cpu >= n_exp_layers else ["--n-cpu-moe", str(n_cpu)]
                suggestions.insert(0, {
                    "id": "moe-offload",
                    "label": S["moe_label"].format(flags=" ".join(flag)),
                    "explanation": S["moe_expl"].format(
                        desc=moe_desc, exps=_gb(exps_mb), ncpu=n_cpu, ntot=n_exp_layers,
                        gpu=_gb(new_gpu_mb), ram=_gb(cpu_mb),
                    ),
                    "add_flags": flag,
                    "remove_flags": ["--cpu-moe", "-cmoe", "--n-cpu-moe", "-ncmoe"],
                    "set": {"n_gpu_layers": 999},
                })
                if not ram_ok and ram_ok_total:
                    messages.append({
                        "severity": "warn",
                        "text": S["moe_ram_warn"].format(avail=_gb(ram_available_mb), need=_gb(cpu_mb)),
                    })
        else:
            # Dense model: lower -ngl.
            if n_layers and model_mb:
                per_dense_layer = model_mb / n_layers
                budget = gpu_budget_mb - headroom_mb - kv_mb - compute_mb
                new_ngl = max(0, min(n_layers, int(budget // per_dense_layer)))
                cpu_mb = int(model_mb * (n_layers - new_ngl) / n_layers)
                if cpu_mb <= ram_total_mb * RAM_SAFETY:
                    suggestions.insert(0, {
                        "id": "dense-partial",
                        "label": S["dense_label"].format(ngl=new_ngl),
                        "explanation": S["dense_expl"].format(ngl=new_ngl, layers=n_layers, ram=_gb(cpu_mb)),
                        "add_flags": [], "remove_flags": [],
                        "set": {"n_gpu_layers": new_ngl},
                    })
                else:
                    level = "too_big"
                    headline = S["too_big2"]
                    messages.append({
                        "severity": "error",
                        "text": S["too_big_hint"],
                    })

    if unified and not no_gpu:
        messages.append({
            "severity": "info",
            "text": S["unified_info"].format(total=_gb(ram_total_mb)),
        })
        # Apple caps how much of the pool Metal may wire down (and an AMD APU
        # caps it in the BIOS). The model can fit in RAM yet still fail to
        # load because that cap is lower.
        if gpu_need_mb > gpu_total_mb > 0:
            messages.append({
                "severity": "warn",
                "text": S["unified_cap_warn"].format(
                    need=_gb(gpu_need_mb), cap=_gb(gpu_total_mb),
                    mb=int(gpu_need_mb + headroom_mb),
                ),
            })

    return {
        "available": True,
        "level": level,
        "headline": headline,
        "estimate": est.to_dict(),
        "hardware": {
            "gpu_total_mb": gpu_total_mb,
            "gpu_free_mb": gpu_free_mb,
            "ram_total_mb": ram_total_mb,
            "ram_available_mb": ram_available_mb,
            "unified_memory": unified,
        },
        "plan": {
            "gpu_need_mb": int(gpu_need_mb),
            "ram_need_mb": int(ram_need_mb),
            "cpu_moe_layers": cpu_moe_layers,
            # Already applied to gpu_need_mb; exposed so the client-side offload
            # slider predicts the same numbers this panel shows.
            "calibration_mb": calibration_mb,
            "headroom_mb": headroom_mb,
            # True once this model has been measured on this kind of card, and
            # the plan is therefore budgeted against what it really used.
            "measured": bool(est.details.get("measured")),
        },
        "model": {
            "is_moe": is_moe,
            "expert_count": profile.get("expert_count"),
            "expert_used_count": profile.get("expert_used_count"),
            "n_layers": n_layers or None,
            "n_exp_layers": n_exp_layers or None,
            "exps_mb": exps_mb or None,
            "context_length": ctx_len,
        },
        "messages": messages,
        "suggestions": suggestions,
    }


async def check_fit_async(
    cfg: LlamaServerConfig,
    gpu_total_mb: int,
    gpu_free_mb: int,
    ram_total_mb: int,
    ram_available_mb: int,
    gpu_budget_mb: int | None = None,
    lang: str = "en",
    unified: bool = False,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        check_fit, cfg, gpu_total_mb, gpu_free_mb, ram_total_mb, ram_available_mb,
        gpu_budget_mb, lang, unified,
    )
