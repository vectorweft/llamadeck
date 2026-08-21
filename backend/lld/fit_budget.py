"""What a plan may count on: VRAM free now, and VRAM one stop away.

`fit_check` answers "does this preset fit"; this module answers the question
underneath it — fit against *what*. The editor's fit panel and the start
preflight both need the same numbers for the cards a preset is pinned to:
total, free right now, and the budget the plan would have once the
llama-servers LlamaDeck itself runs are stopped. They used to compute those
inline, in two copies that had already drifted apart.

Both vendors can now answer "who is holding what" — nvidia-smi for NVIDIA, DRM
fdinfo for amdgpu — so the budget is normally built from real per-process
numbers, narrowed to the cards the plan actually runs on. A row on the other
card must never count: freeing the 5090 buys an R9700-pinned preset nothing.

Where no such reading exists (Metal, an older kernel, another user's process),
the attribution falls back the other way round: our own running presets are
matched to the pinned cards through their device pin, each one's VRAM comes
from the estimate the supervisor already carries for it, and the sum is capped
by what those cards report as *used*. That cap is what keeps the fallback
honest — it can never promise back more than the hardware says is allocated,
no matter who allocated it.

Getting this wrong is not cosmetic. While nothing outside NVIDIA could be
attributed, a busy Radeon looked permanently full: a 27 GB preset pinned to a
32 GB R9700 that already had a model on it came back "doesn't fit — even with
the experts in RAM the core part exceeds the card", i.e. "download a smaller
model", when the true answer was "stop the model that is already on it".
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .devices import LlamaDevice, _normalized_name, probe_devices
from .settings import LlamaServerConfig, load_settings
from .vram import GpuInfo, GpuProcess, offload_gpus


@dataclass
class PlanBudget:
    """The hardware side of one preset's fit-check."""

    total_mb: int
    # Free right now, plus the VRAM of presets the caller is about to stop on
    # purpose (a switch/restart) — that memory is back before the new process
    # asks for it.
    free_mb: int
    # free_mb plus the VRAM held by the *other* llama-servers we could stop.
    # Memory that never comes back (desktop/compositor, other applications)
    # drops out by construction: it is neither free nor ours to stop.
    budget_mb: int
    # The plan runs on memory shared with the CPU (Apple Silicon, AMD APU).
    unified: bool


def _canon(ids: Iterable[str], by_id: Mapping[str, LlamaDevice]) -> set[str]:
    """Device ids with backend aliases collapsed onto one card.

    A CUDA+Vulkan build lists the 5090 as both CUDA0 and Vulkan2, so a preset
    pinned to one and a plan pinned to the other compete for the same 32 GB —
    comparing the raw ids would call them unrelated.
    """
    out: set[str] = set()
    for i in ids or ():
        dev = by_id.get(i)
        out.add((dev.duplicate_of or dev.id) if dev else i)
    return out


def _os_card_for(dev: LlamaDevice, gpus: Sequence[GpuInfo]) -> GpuInfo | None:
    """The OS probe's row for the same physical card, if it can be identified.

    Matched by name — Vulkan's driver suffix ("… (RADV GFX1201)") is stripped
    the same way duplicate detection strips it. Two identical cards are told
    apart by the ordinal in the device id; anything less certain than that
    returns None rather than a guess.
    """
    key = _normalized_name(dev.name)
    same = [g for g in gpus if _normalized_name(g.name) == key]
    if len(same) == 1:
        return same[0]
    if not same:
        return None
    m = re.search(r"(\d+)$", dev.id)
    if not m:
        return None
    ordinal = int(m.group(1))
    return next((g for g in same if g.index == ordinal), None)


def _os_card_key(dev: LlamaDevice, gpus: Sequence[GpuInfo]) -> tuple[str, int] | None:
    """(vendor, index) of the card behind a device row — the key a GpuProcess
    carries. None when the two views cannot be tied together."""
    match = _os_card_for(dev, gpus)
    return (match.vendor, match.index) if match else None


def _in_scope(proc: GpuProcess, keys: set[tuple[str, int]] | None) -> bool:
    """Whether a process row sits on one of the plan's cards.

    `keys` None is the unpinned plan: the whole machine is in scope. A row
    whose card could not be resolved (`gpu_index` None — nvidia-smi's bus map
    failed) is kept when its vendor is in scope: on the single-card-per-vendor
    box that is the normal shape it is certainly this card, and dropping it
    would silently zero the budget.
    """
    if keys is None:
        return True
    if proc.gpu_index is None:
        return any(vendor == proc.vendor for vendor, _index in keys)
    return proc.card_key in keys


def _card_memory(dev: LlamaDevice, gpus: Sequence[GpuInfo]) -> tuple[int, int]:
    """(total_mb, free_mb) for one pinned card.

    llama.cpp prints `CUDA0: … (0 MiB, 0 MiB free)` when its cudaMemGetInfo
    call fails — which is precisely what a card that is already full does.
    Taken at face value that says "this machine has no GPU", and check_fit
    then plans a 30 GB preset entirely into system RAM and calls it a fit. So
    a zero row is treated as a failed reading and replaced by the OS probe's
    numbers for the same card when it can be identified.
    """
    if dev.total_mb > 0:
        return dev.total_mb, dev.free_mb
    match = _os_card_for(dev, gpus)
    if match is None:
        return dev.total_mb, dev.free_mb
    return match.total_mb, match.free_mb


def _runs_on(row: dict, on_ids: set[str] | None, by_id: Mapping[str, LlamaDevice]) -> bool:
    """Whether a running preset occupies the cards in `on_ids`.

    `on_ids` None means "the whole machine" — there is no pin to narrow it
    down. A running preset with no pin of its own counts either way: an
    unpinned model spreads over every visible card, so it is on this one too.
    """
    if on_ids is None:
        return True
    pin = _canon((row.get("config") or {}).get("devices") or [], by_id)
    return not pin or bool(pin & on_ids)


def _own_vram_mb(statuses: Mapping[str, dict], names: Iterable[str]) -> int:
    """Estimated VRAM the named presets hold, from the supervisor's own rows."""
    total = 0
    for name in names:
        row = statuses.get(name) or {}
        # Router presets and unreadable GGUFs have no estimate — never invent one.
        total += max(0, int((row.get("vram_estimate") or {}).get("gpu_mb") or 0))
    return total


async def plan_budget(
    cfg: LlamaServerConfig,
    *,
    gpus: Sequence[GpuInfo],
    procs: Sequence[GpuProcess],
    statuses: Mapping[str, dict],
    freeing: Iterable[str] = (),
) -> PlanBudget:
    """VRAM totals to plan `cfg` against.

    `freeing` names presets the caller stops as part of this start (switch or
    restart): their memory counts as free rather than merely reclaimable.
    """
    planned_on = offload_gpus(list(gpus))
    total_mb = sum(g.total_mb for g in planned_on)
    free_mb = sum(g.free_mb for g in planned_on)
    # "Unified" is a property of the cards the plan runs on, not of the
    # machine: `any(g.unified ...)` over every GPU let a desktop Ryzen's iGPU
    # put a discrete 5090 plan on shared-memory math.
    unified = any(g.unified for g in planned_on)

    picked: list[LlamaDevice] = []
    by_id: dict[str, LlamaDevice] = {}
    if cfg.devices:
        by_id = {d.id: d for d in await probe_devices(load_settings().llama_bin or "")}
        resolved = [by_id[i] for i in cfg.devices if i in by_id]
        # Ids that do not resolve mean "no opinion" — plan against the whole
        # machine exactly as an unpinned preset does.
        if resolved and len(resolved) == len(cfg.devices):
            picked = resolved
            # A pinned preset is planned against its own cards, not the
            # machine total: budgeting a one-card model against both would
            # call it a fit and then OOM at load.
            memory = [_card_memory(d, planned_on) for d in picked]
            total_mb = sum(t for t, _ in memory)
            free_mb = sum(f for _, f in memory)
            # A pin can also point AT the iGPU, which puts the plan back on
            # shared memory even on a box that has a discrete card.
            unified = all(d.integrated for d in picked)

    on_ids = _canon([d.id for d in picked], by_id) if picked else None
    running = {
        n for n, r in statuses.items() if r.get("running") and _runs_on(r, on_ids, by_id)
    }
    stopping = running & {n for n in freeing if n}
    others = running - stopping
    pids = {int(statuses[n]["pid"]) for n in stopping if statuses[n].get("pid")}

    # Per-process VRAM, narrowed to the cards this plan runs on. A row on the
    # other card must not count — freeing the 5090 buys the R9700 nothing.
    keys = {k for k in (_os_card_key(d, planned_on) for d in picked) if k} if picked else None
    on_scope = [p for p in procs if _in_scope(p, keys)]
    our_pids = {int(statuses[n]["pid"]) for n in running if statuses[n].get("pid")}
    # The reading covers this scope when our own servers show up in it — or
    # when we have none running there to show up in the first place.
    covered = bool({p.pid for p in on_scope} & our_pids) or not running

    if covered:
        freed = sum(p.used_mb for p in on_scope if p.pid in pids)
        reclaimable = sum(
            p.used_mb
            for p in on_scope
            if "llama" in (p.process_name or "").lower() and p.pid not in pids
        )
    else:
        # Nothing can attribute memory on these cards. Fall back to what our
        # own running presets estimate, capped by the memory the cards report
        # as used.
        used = (
            sum(max(0, t - f) for t, f in (_card_memory(d, planned_on) for d in picked))
            if picked
            else sum(g.used_mb for g in planned_on)
        )
        freed = min(used, _own_vram_mb(statuses, stopping))
        reclaimable = min(max(0, used - freed), _own_vram_mb(statuses, others))

    free_mb = min(total_mb, free_mb + freed) if total_mb else free_mb + freed
    budget_mb = min(total_mb, free_mb + reclaimable) if total_mb else 0
    return PlanBudget(
        total_mb=total_mb, free_mb=free_mb, budget_mb=budget_mb, unified=unified
    )
