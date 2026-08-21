<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { api, formatUptime, offloadGpus, type ModelDefaults, type PresetStatus, type VramReport } from '$lib/api';
  import { fetchSnapshot, openMetricsStream, sparklinePath, type MetricsFrame, type SlotInfo } from '$lib/metrics';
  import { modelLabel } from '$lib/ui';
  import { confirmDialog } from '$lib/confirm';
  import { t, tLang } from '$lib/i18n.svelte';
  import Skeleton from '$lib/components/Skeleton.svelte';
  import Donut from '$lib/components/Donut.svelte';
  import Gauge from '$lib/components/Gauge.svelte';
  import Meter from '$lib/components/Meter.svelte';
  import StatusPill from '$lib/components/StatusPill.svelte';
  import SetupCard from '$lib/components/SetupCard.svelte';

  const SPARK_LEN = 60; // 30 seconds at 2 Hz

  let statuses = $state<Record<string, PresetStatus>>({});
  let frames = $state<Record<string, MetricsFrame | null>>({});
  let history = $state<Record<string, MetricsFrame[]>>({});
  let recDefaults = $state<Record<string, ModelDefaults | null>>({});
  let vram = $state<VramReport | null>(null);
  let error = $state<string | null>(null);
  const streams = new Map<string, EventSource>();
  let statusTimer: ReturnType<typeof setInterval> | null = null;
  let loaded = $state(false);

  // Two switchable dashboard layouts: 'classic' (donut panels + big stat tiles
  // + tok/s sparkline) and 'compact' (an instrument cluster: dials and
  // segmented meters, every reading led by its number, no time series).
  // Persisted per browser.
  let layout = $state<'classic' | 'compact'>('classic');
  function setLayout(l: 'classic' | 'compact') {
    layout = l;
    try { localStorage.setItem('dash_layout', l); } catch { /* private mode */ }
  }

  const running = $derived.by(() => Object.values(statuses).filter(s => s.running).sort((a, b) => a.port - b.port));
  const idle = $derived.by(() => Object.values(statuses).filter(s => !s.running).sort((a, b) => a.port - b.port));

  // One-line machine summary for the first-run setup card, so "download a
  // model" can say what this box can actually hold. Built from the VRAM report
  // the dashboard already polls — null until that first response lands, which
  // the card renders as a shorter sentence.
  const hwSummary = $derived.by<string | null>(() => {
    if (!vram) return null;
    const gb = (mb: number) => Math.round(mb / 1024);
    const parts: string[] = [];
    const gpus = offloadGpus(vram.gpus);
    if (gpus.length > 0) {
      // Identical multi-GPU rigs read better as "2× RTX 5090" than a repeated list.
      const names = [...new Set(gpus.map(g => g.name))];
      parts.push(names.length === 1 && gpus.length > 1 ? `${gpus.length}× ${names[0]}` : names.join(' + '));
    } else if (vram.platform?.cpu_name) {
      parts.push(vram.platform.cpu_name);
    }
    const totalVram = gpus.reduce((s, g) => s + g.total_mb, 0);
    // Unified memory is system RAM — reporting it as both VRAM and RAM would
    // double-count the same pool, so say it once.
    if (vram.unified_memory) {
      if (totalVram > 0) parts.push(t('{n} GB unified memory', { n: gb(totalVram) }));
    } else {
      if (totalVram > 0) parts.push(t('{n} GB VRAM', { n: gb(totalVram) }));
      if (vram.ram?.total_mb) parts.push(t('{n} GB RAM', { n: gb(vram.ram.total_mb) }));
    }
    return parts.length > 0 ? parts.join(' · ') : null;
  });

  async function refreshVram() {
    try {
      const v = await api.serverVram();
      vram = v;
    } catch (e) {
      console.error('serverVram failed', e);
    }
  }

  async function refreshStatuses() {
    try {
      const r = await api.serverStatuses();
      // Merge keys in place so existing object refs are preserved where the
      // backend state hasn't actually changed (prevents flicker / re-mount).
      const next: Record<string, PresetStatus> = { ...statuses };
      for (const [name, s] of Object.entries(r.presets)) next[name] = s;
      for (const name of Object.keys(next)) if (!(name in r.presets)) delete next[name];
      statuses = next;
      loaded = true;

      const runningNames = new Set(Object.values(r.presets).filter(s => s.running).map(s => s.name));
      for (const [name, es] of streams) {
        if (!runningNames.has(name)) { es.close(); streams.delete(name); }
      }
      for (const name of runningNames) {
        if (!streams.has(name)) openStream(name);
        // Cache recommended defaults per running preset (fetch once per run).
        if (!(name in recDefaults)) {
          const cfg = r.presets[name]?.config;
          if (cfg?.model_path) {
            api.modelDefaults(cfg.model_path, name)
              .then(d => { recDefaults[name] = d; })
              .catch(() => { recDefaults[name] = null; });
          }
        }
      }
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function openStream(name: string) {
    const es = openMetricsStream(name, (f) => {
      frames[name] = f;
      const h = history[name] ?? [];
      h.push(f);
      if (h.length > SPARK_LEN) h.splice(0, h.length - SPARK_LEN);
      history[name] = h;
    });
    es.onerror = () => {
      // auto-reconnect: EventSource reconnects on its own; if it closes fully, reopen next poll
    };
    streams.set(name, es);
  }

  onMount(async () => {
    try {
      if (localStorage.getItem('dash_layout') === 'compact') layout = 'compact';
    } catch { /* private mode */ }
    try {
      const snap = await fetchSnapshot(SPARK_LEN);
      for (const [name, entry] of Object.entries(snap.presets)) {
        history[name] = entry.history;
        frames[name] = entry.latest;
      }
    } catch { /* ignore */ }
    await Promise.all([refreshStatuses(), refreshVram()]);
    statusTimer = setInterval(() => { if (!restarting) { refreshStatuses(); refreshVram(); } }, 3000);
  });

  onDestroy(() => {
    if (statusTimer) clearInterval(statusTimer);
    for (const es of streams.values()) es.close();
  });

  let restarting = $state(false);
  let restartMsg = $state<string | null>(null);
  let restartFailed = $state(false);
  let restartElapsed = $state(0);

  async function restartBackend() {
    const ok = await confirmDialog(
      t('Reloads code changes. Managed llama-server child processes stay alive across the exec; the supervisor reattaches via the scan/adopt flow.'),
      { title: t('Restart the LlamaDeck backend?'), confirmLabel: t('Restart') }
    );
    if (!ok) return;
    restarting = true;
    restartFailed = false;
    restartElapsed = 0;
    error = null; // expected connection drops must not surface as page errors
    restartMsg = t('Sending restart request…');
    // The endpoint replies 202 ~0.4 s BEFORE execv fires, so naive /health
    // polling races that window: the first poll hits the OLD process, looks
    // "ready", and reload() lands mid-exec on the browser's connection-refused
    // page. Discriminate on boot_id, NOT pid — execv keeps the pid (that's how
    // child llama-servers survive), so the pid is identical across a restart.
    let oldBoot: string | null = null;
    try {
      const h = await fetch('/health', { cache: 'no-store' });
      oldBoot = (await h.json())?.boot_id ?? null;
    } catch { /* proceed without a baseline; falls back to the time heuristic */ }
    try {
      await api.systemRestart();
    } catch (e) {
      // Connection drop is expected as the process re-execs; ignore.
      console.warn('restart triggered, expecting reconnect', e);
    }
    restartMsg = t('Backend is restarting, waiting for connection…');
    const started = Date.now();
    // Cold boot runs the full model rescan inside the lifespan before /health
    // answers; on a large model tree that can take a while, so allow 90 s.
    const deadline = started + 90_000;
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 300));
      const elapsed = Date.now() - started;
      restartElapsed = Math.floor(elapsed / 1000);
      try {
        const r = await fetch('/health', { cache: 'no-store' });
        if (r.ok) {
          const body = await r.json().catch(() => ({}));
          const newProcess = typeof body.boot_id === 'string' && oldBoot != null
            ? body.boot_id !== oldBoot
            : elapsed > 1500; // old backend without boot_id: exec fires at ~0.4 s, so a late OK is the new process
          if (newProcess) {
            restartMsg = t('Backend ready — reloading the page.');
            await new Promise(r2 => setTimeout(r2, 400)); // let the user read it
            location.reload();
            return;
          }
        }
      } catch { /* still down, keep polling */ }
    }
    restartFailed = true;
    restartMsg = t('Backend did not come back within 90 s.');
  }

  function fmt(n: number | null | undefined, d = 0): string {
    if (n == null || isNaN(n)) return '—';
    return d === 0 ? Math.round(n).toString() : n.toFixed(d);
  }

  function slotPct(slot: SlotInfo | undefined): number {
    if (!slot || !slot.is_processing) return 0;
    const total = slot.n_decoded + slot.n_remain;
    if (total <= 0) return 0;
    return (slot.n_decoded / total) * 100;
  }

  /** Compare actual vs recommended sampling value; return drift tag
   * (empty if close enough; 'off' if noticeably different). */
  function driftTag(actual: number | null | undefined, recommended: number | null | undefined, tol = 0.02): '' | 'off' {
    if (actual == null || recommended == null) return '';
    return Math.abs(actual - recommended) > tol ? 'off' : '';
  }

  /** Rolling tok/s over the last `windowSec` of history. The backend already
   * picks the best signal per frame (per-slot n_decoded delta during active
   * inference, falls back to Prometheus counter delta at completion); we just
   * average its `instant_*_tps` values over the window for a smooth readout.
   * Frames with null instant_*_tps (swap point, no prev) are skipped. */
  function rollingTps(hist: MetricsFrame[], field: 'instant_decode_tps' | 'instant_prompt_tps', windowSec = 3): number {
    if (hist.length < 1) return 0;
    const last = hist[hist.length - 1];
    const cutoff = last.ts - windowSec;
    let sum = 0, n = 0;
    for (let i = hist.length - 1; i >= 0; i--) {
      if (hist[i].ts < cutoff) break;
      const v = (hist[i] as any)[field];
      if (typeof v === 'number' && Number.isFinite(v)) { sum += v; n++; }
    }
    return n > 0 ? sum / n : 0;
  }

  /** The same 3 s smoothing as `rollingTps`, evaluated at every frame.
   * The compact gauges mark their peak with a tick, and that tick has to be a
   * value the needle actually reached — the peak of the raw 2 Hz samples is a
   * single-frame spike the reading never showed. */
  function rollingSeries(hist: MetricsFrame[], field: 'instant_decode_tps' | 'instant_prompt_tps', windowSec = 3): number[] {
    const out: number[] = [];
    for (let i = 0; i < hist.length; i++) {
      const cutoff = hist[i].ts - windowSec;
      let sum = 0, n = 0;
      for (let j = i; j >= 0; j--) {
        if (hist[j].ts < cutoff) break;
        const v = (hist[j] as any)[field];
        if (typeof v === 'number' && Number.isFinite(v)) { sum += v; n++; }
      }
      out.push(n > 0 ? sum / n : 0);
    }
    return out;
  }

  /** Full-scale deflection for a gauge: the next round number above `v`.
   * Snapping to 1/1.5/2/3/5/7.5 keeps the dial still while a rate wanders,
   * instead of re-scaling under the needle on every frame. */
  function niceMax(v: number): number {
    if (!Number.isFinite(v) || v <= 0) return 10;
    const base = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.5, 2, 3, 5, 7.5]) {
      if (v <= m * base) return m * base;
    }
    return 10 * base;
  }

  /** A reading the hardware does not report is "—", never 0: a fanless card
   * and a stopped fan are different facts, and only one of them is a problem. */
  function sensor(v: number | null | undefined, d = 0): string {
    return v == null || !Number.isFinite(v) ? '—' : v.toFixed(d);
  }

  /** Map 30–100 °C onto a bar. Room temperature lights nothing; a card at its
   * thermal limit fills it. (A raw 0–100 scale would sit a third full at idle.) */
  function tempPct(c: number | null | undefined): number {
    if (c == null) return 0;
    return Math.min(100, Math.max(0, ((c - 30) / 70) * 100));
  }

  /** Thresholds are the card's, not the bar's: 72 °C is warm for silicon and
   * 85 °C is where consumer GPUs start pulling their own clocks down. */
  function tempTone(c: number | null | undefined): 'emerald' | 'amber' | 'rose' | 'slate' {
    if (c == null) return 'slate';
    return c >= 85 ? 'rose' : c >= 72 ? 'amber' : 'emerald';
  }

  /** 1.2M / 34.5k — a lifetime token counter is a magnitude, not a number you
   * read digit by digit. */
  function compactNum(n: number): string {
    if (!Number.isFinite(n)) return '—';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(Math.round(n));
  }
</script>

{#snippet probeWarning(message: string)}
  <!-- A GPU the machine has, missing from the list with no explanation, is the
       worst version of this panel: the card is simply gone and the user is
       left to wonder whether it died. The backend knows exactly why the probe
       came back empty — nearly always a driver package upgraded without a
       reboot, leaving the loaded kernel module behind its userspace — so say
       so here rather than quietly showing one card fewer. -->
  <div class="mb-3 flex items-start gap-2 rounded border border-amber-800/60 bg-amber-900/20 px-3 py-2 text-xs text-amber-200">
    <span aria-hidden="true">⚠</span>
    <span>
      <span class="font-semibold">{t('A GPU may be missing from this list.')}</span>
      <span class="text-amber-200/80">
        {t('nvidia-smi is installed but is not answering:')}
        <span class="font-mono" lang="en">{message}</span>
      </span>
    </span>
  </div>
{/snippet}

<div class="max-w-6xl space-y-6">
  <div class="flex items-baseline gap-3 flex-wrap">
    <h1 class="text-2xl font-semibold">{t('Dashboard')}</h1>
    <span class="text-xs text-slate-500 font-mono">{t('{n} active · {m} idle', { n: running.length, m: idle.length })}</span>
    <div class="ml-auto flex items-center gap-3">
      <div class="flex rounded border border-slate-700 overflow-hidden text-xs font-mono" role="group" aria-label={t('Dashboard layout')}>
        <button
          onclick={() => setLayout('classic')}
          title={t('Classic layout: donut panels and large stat tiles')}
          class="px-2.5 py-1 {layout === 'classic' ? 'bg-slate-700/70 text-slate-100' : 'bg-slate-800/40 text-slate-500 hover:text-slate-300'}"
        >{t('Classic')}</button>
        <button
          onclick={() => setLayout('compact')}
          title={t('Compact layout: dials and big readouts — speeds, VRAM, temperature, fan, load')}
          class="px-2.5 py-1 border-l border-slate-700 {layout === 'compact' ? 'bg-slate-700/70 text-slate-100' : 'bg-slate-800/40 text-slate-500 hover:text-slate-300'}"
        >{t('Compact')}</button>
      </div>
      <button
        disabled={restarting}
        onclick={restartBackend}
        title={t('Restarts the LlamaDeck backend (FastAPI process). Managed llama-server children stay alive.')}
        class="rounded border border-slate-700 bg-slate-800/60 hover:bg-slate-700/70 px-3 py-1 text-xs font-mono text-slate-300 disabled:opacity-50"
      >⟳ {restarting ? t('restarting…') : t('Restart backend')}</button>
    </div>
  </div>

  {#if error && !restarting}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-2 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  <!-- First-run guide. Renders nothing once the install is complete, so it
       costs a fresh user one card and an established user nothing. Gated on
       `loaded` to avoid flashing "step 3" before the statuses arrive. -->
  {#if loaded}
    <SetupCard {statuses} hardware={hwSummary} />
  {/if}

  <!-- VRAM + Tokens panels (2 columns) + full-width System panel -->
  {#if vram}
    {@const gpuList = offloadGpus(vram.gpus)}
    {@const totalMb = gpuList.reduce((s, g) => s + g.total_mb, 0)}
    {@const hasGpu = gpuList.length > 0}
    {@const tokensRows = running.map(s => {
      const f = frames[s.name];
      const decoded = f?.prom?.['llamacpp:tokens_predicted_total'] ?? 0;
      const prompt = f?.prom?.['llamacpp:prompt_tokens_total'] ?? 0;
      return { name: s.name, label: modelLabel(s.config, s.name), decoded, prompt, live: rollingTps(history[s.name] ?? [], 'instant_decode_tps', 3) };
    })}
    {@const totDecoded = tokensRows.reduce((a, r) => a + r.decoded, 0)}
    {@const totPrompt = tokensRows.reduce((a, r) => a + r.prompt, 0)}
    {@const totLive = tokensRows.reduce((a, r) => a + r.live, 0)}
    {@const loadedModels = running.flatMap(s => {
      const m = frames[s.name]?.loaded_model_id;
      return m ? [{ preset: s.name, model: m }] : [];
    })}
    {#if layout === 'classic'}
    <div class="grid gap-4 lg:grid-cols-2">
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
      {#if vram.probe_warning}{@render probeWarning(vram.probe_warning)}{/if}
      {#if hasGpu}
      <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">
          VRAM
          {#if vram.unified_memory}
            <span
              class="ml-1.5 rounded border border-cyan-800 bg-cyan-900/30 px-1.5 py-0.5 text-[10px] normal-case tracking-normal text-cyan-300"
              title={t('GPU memory is shared with system RAM — the two gauges show one pool.')}
            >{t('unified')}</span>
          {/if}
        </h2>
        {#if vram.active_estimate_mb > 0}
          <span class="text-[11px] font-mono text-slate-500">
            <span class="inline-block h-2 w-2 rounded-sm bg-emerald-500 align-middle"></span> used
            · <span class="inline-block h-2 w-2 rounded-sm bg-amber-400 align-middle ml-1"></span> {t('active preset estimate')}
          </span>
        {/if}
      </div>
      <div class="grid gap-4" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr))">
        {#each gpuList as gpu (gpu.vendor + '' + gpu.index)}
          {@const usedPct = gpu.total_mb > 0 ? (gpu.used_mb / gpu.total_mb) * 100 : 0}
          {@const estPct = gpu.total_mb > 0 ? Math.min(100, (vram.active_estimate_mb / gpu.total_mb) * 100) : 0}
          <div class="flex items-center gap-4">
            <Donut pct={usedPct} innerPct={estPct} label="VRAM {gpu.name}"
              sub="{(gpu.used_mb / 1024).toFixed(1)}/{(gpu.total_mb / 1024).toFixed(0)} GB" />
            <div class="min-w-0 flex-1 space-y-1">
              <div class="text-sm font-mono font-semibold text-slate-200 truncate" title={gpu.name}>{gpu.name}</div>
              <div class="text-sm font-mono text-slate-500">
                used <span class="text-slate-100">{(gpu.used_mb / 1024).toFixed(1)} GB</span>
              </div>
              <div class="text-sm font-mono text-slate-500">
                free <span class="text-slate-100">{(gpu.free_mb / 1024).toFixed(1)} GB</span>
              </div>
              {#if vram.active_estimate_mb > 0}
                <div class="text-sm font-mono text-amber-400/90">
                  est. <span>{(vram.active_estimate_mb / 1024).toFixed(1)} GB</span>
                  <span class="text-slate-600 text-xs">{t('({p}% of total)', { p: estPct.toFixed(0) })}</span>
                </div>
              {/if}
            </div>
          </div>
        {/each}
      </div>

      <!-- Process breakdown: who is using VRAM -->
      {#if vram.processes && vram.processes.length > 0}
        {@const procSum = vram.processes.reduce((s, p) => s + p.used_mb, 0)}
        {@const otherMb = Math.max(0, vram.used_mb - procSum)}
        <div class="mt-4 pt-3 border-t border-slate-800">
          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2 font-mono">{t('VRAM consumers')}</div>
          <div class="space-y-1.5">
            {#each vram.processes as p (p.pid)}
              {@const pct = totalMb > 0 ? (p.used_mb / totalMb) * 100 : 0}
              {@const labelColor = p.preset ? (p.adopted ? 'text-amber-300' : 'text-emerald-300') : 'text-slate-300'}
              <div class="flex items-center gap-3 text-xs font-mono">
                <div class="min-w-0 flex-1">
                  <div class="flex items-baseline gap-2 flex-wrap">
                    <span class={labelColor}>{p.model ?? p.process_name.split('/').pop()}</span>
                    {#if p.preset}
                      <span class="text-[11px] text-slate-600">preset: {p.preset}{p.adopted ? ' · adopted' : ''}</span>
                    {:else}
                      <span class="text-[11px] text-slate-600">{t('PID {pid} · external', { pid: p.pid })}</span>
                    {/if}
                  </div>
                  <div class="h-1 rounded-full bg-slate-800 overflow-hidden mt-1">
                    <div class="h-full {p.preset ? (p.adopted ? 'bg-amber-500' : 'bg-emerald-500') : 'bg-slate-500'}" style="width: {pct}%"></div>
                  </div>
                </div>
                <div class="shrink-0 text-right tabular-nums w-28">
                  <span class="text-slate-200">{(p.used_mb / 1024).toFixed(2)} GB</span>
                  <span class="text-slate-600 text-[11px]"> · {pct.toFixed(1)}%</span>
                </div>
              </div>
            {/each}
            {#if otherMb > 200}
              {@const pct = totalMb > 0 ? (otherMb / totalMb) * 100 : 0}
              <div class="flex items-center gap-3 text-xs font-mono opacity-70">
                <div class="min-w-0 flex-1">
                  <div class="flex items-baseline gap-2">
                    <span class="text-slate-400">{t('other (driver/desktop/…)')}</span>
                  </div>
                  <div class="h-1 rounded-full bg-slate-800 overflow-hidden mt-1">
                    <div class="h-full bg-slate-600" style="width: {pct}%"></div>
                  </div>
                </div>
                <div class="shrink-0 text-right tabular-nums w-28">
                  <span class="text-slate-300">{(otherMb / 1024).toFixed(2)} GB</span>
                  <span class="text-slate-600 text-[11px]"> · {pct.toFixed(1)}%</span>
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/if}

      <!-- Which model is actually loaded right now (router presets) — big and
           front-and-center, in the otherwise-empty bottom of the VRAM card. -->
      {#if loadedModels.length > 0}
        <div class="mt-4 pt-3 border-t border-slate-800">
          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2 font-mono">{t('Loaded model')}</div>
          <div class="space-y-1.5">
            {#each loadedModels as lm (lm.preset)}
              <div class="flex items-baseline gap-3 flex-wrap">
                <span class="text-xl font-mono font-semibold text-cyan-300 break-all">{lm.model}</span>
                <span class="text-xs font-mono text-slate-500">{lm.preset}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
      {:else}
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">VRAM</h2>
      <div class="text-xs text-slate-500 font-mono leading-relaxed">
        {t('No GPU memory telemetry on this machine — running in CPU mode. Models are planned against system RAM below.')}
      </div>
      {/if}
    </section>

    <!-- Tokens panel: total prompt + decoded across active presets -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
      <div class="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">Tokens</h2>
        {#if tokensRows.length > 0}
          <span class="text-[11px] font-mono text-slate-500">{tokensRows.length === 1 ? t('1 active preset · lifetime') : t('{n} active presets · lifetime', { n: tokensRows.length })}</span>
        {/if}
      </div>
      {#if tokensRows.length === 0}
        <div class="text-xs text-slate-500 font-mono">{t('No active presets.')}</div>
      {:else}
        <div class="grid grid-cols-[repeat(auto-fit,minmax(8rem,1fr))] gap-3 mb-4">
          <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
            <div class="text-[11px] uppercase tracking-wider text-slate-500">decoded</div>
            <div class="text-2xl font-mono text-emerald-400 leading-tight tabular-nums">{totDecoded.toLocaleString()}</div>
            <div class="text-[11px] text-slate-600 font-mono">{t('tokens generated')}</div>
          </div>
          <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
            <div class="text-[11px] uppercase tracking-wider text-slate-500">prompt</div>
            <div class="text-2xl font-mono text-cyan-400 leading-tight tabular-nums">{totPrompt.toLocaleString()}</div>
            <div class="text-[11px] text-slate-600 font-mono">{t('tokens processed')}</div>
          </div>
          <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
            <div lang="en" class="text-[11px] uppercase tracking-wider text-slate-500">live</div>
            <div class="text-2xl font-mono text-violet-400 leading-tight tabular-nums">{totLive.toFixed(1)}</div>
            <div class="text-[11px] text-slate-600 font-mono">{t('decode tok/s right now')}</div>
          </div>
        </div>

        <div class="pt-3 border-t border-slate-800">
          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2 font-mono">{t('By preset')}</div>
          <div class="space-y-1.5">
            {#each tokensRows as r (r.name)}
              {@const decPct = totDecoded > 0 ? (r.decoded / totDecoded) * 100 : 0}
              <div class="flex items-center gap-3 text-xs font-mono">
                <div class="min-w-0 flex-1">
                  <div class="flex items-baseline gap-2 flex-wrap">
                    <span class="text-emerald-300 truncate">{r.label}</span>
                    <span class="text-[11px] text-slate-600">{r.live > 0.5 ? `${r.live.toFixed(1)} tok/s` : 'idle'}</span>
                  </div>
                  <div class="h-1 rounded-full bg-slate-800 overflow-hidden mt-1">
                    <div class="h-full bg-emerald-500" style="width: {decPct}%"></div>
                  </div>
                </div>
                <div class="shrink-0 text-right tabular-nums w-32 text-[11px]">
                  <div class="text-emerald-200">{r.decoded.toLocaleString()}<span class="text-slate-600"> dec</span></div>
                  <div class="text-cyan-300/80">{r.prompt.toLocaleString()}<span class="text-slate-600"> pr</span></div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Power + cumulative energy (only counted while busy) -->
      {#if vram.power}
        {@const p = vram.power}
        {@const bs = p.busy_seconds ?? 0}
        {@const wh = p.energy_wh ?? 0}
        {@const avgW = bs > 0 ? (wh * 3600) / bs : 0}
        <div class="mt-4 pt-3 border-t border-slate-800">
          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2 font-mono">{t('Power draw')}</div>
          <div class="grid grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] gap-3 mb-3">
            <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
              <div class="text-[11px] uppercase tracking-wider text-slate-500">GPU</div>
              <div class="text-2xl font-mono text-emerald-400 leading-tight tabular-nums">{p.gpu_w != null ? p.gpu_w.toFixed(0) : '—'}<span class="text-sm text-slate-500"> W</span></div>
            </div>
            <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
              <div class="text-[11px] uppercase tracking-wider text-slate-500">CPU</div>
              {#if p.cpu_w != null}
                <div class="text-2xl font-mono text-cyan-400 leading-tight tabular-nums">{p.cpu_w.toFixed(0)}<span class="text-sm text-slate-500"> W</span></div>
              {:else}
                <div class="text-2xl font-mono text-slate-600 leading-tight tabular-nums">—</div>
              {/if}
            </div>
            <div class="rounded bg-slate-900/60 border border-amber-900/60 p-3">
              <div class="text-[11px] uppercase tracking-wider text-amber-500/80">{t('Total')}</div>
              <div class="text-2xl font-mono text-amber-300 leading-tight tabular-nums">{p.total_w != null ? p.total_w.toFixed(0) : '—'}<span class="text-sm text-slate-500"> W</span></div>
            </div>
          </div>
          <!-- RAPL counters ship mode 0400 on most distros, so CPU watts read
               as a bare "—" on hardware that does support them. Say what to do
               about it in the panel rather than in a hover tooltip. -->
          {#if p.cpu_status === 'denied'}
            <div class="mb-3 rounded bg-slate-900/60 border border-slate-800 px-3 py-2 text-[11px] text-slate-400">
              {t('CPU watts need readable RAPL counters:')}
              <code class="block mt-1 font-mono text-cyan-300 select-all break-all">sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj</code>
            </div>
          {/if}
          <div class="rounded bg-slate-900/60 border border-slate-800 p-3 space-y-1.5">
            <div class="flex items-baseline justify-between text-xs font-mono">
              <span class="text-slate-400">{t('Cumulative energy')} <span class="text-[11px] text-slate-600">{t('(only while busy)')}</span></span>
              <span class="tabular-nums">
                {#if wh >= 1000}
                  <span class="text-amber-300 text-lg">{(wh / 1000).toFixed(3)}</span><span class="text-slate-500"> kWh</span>
                {:else}
                  <span class="text-amber-300 text-lg">{wh.toFixed(2)}</span><span class="text-slate-500"> Wh</span>
                {/if}
              </span>
            </div>
            <div class="flex items-baseline justify-between text-xs font-mono">
              <span class="text-slate-500">{t('Active busy time')}</span>
              <span class="tabular-nums text-slate-200">{formatUptime(bs)}</span>
            </div>
            <div class="flex items-baseline justify-between text-xs font-mono">
              <span class="text-slate-500">{t('Average power (busy)')}</span>
              <span class="tabular-nums text-slate-200">{bs > 0 ? avgW.toFixed(0) + ' W' : '—'}</span>
            </div>
          </div>
        </div>
      {/if}
    </section>

    <!-- CPU usage % + RAM donut (system-wide) — own full-width card -->
    {#if vram.cpu_percent != null || vram.ram}
      {@const cpuPct = vram.cpu_percent ?? 0}
      {@const ram = vram.ram}
      {@const ramPct = ram?.percent ?? 0}
      <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 lg:col-span-2">
        <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-3">{t('System')}</h2>
        <div class="grid gap-4 grid-cols-1 sm:grid-cols-2">
          <!-- CPU -->
          {#if vram.cpu_percent != null}
            <div class="flex items-center gap-4">
              <Donut pct={cpuPct} base="cyan" label="CPU" sub="CPU" />
              <div class="min-w-0 flex-1 space-y-1">
                <div class="text-sm font-mono font-semibold text-slate-200">CPU</div>
                <div class="text-sm font-mono text-slate-500">
                  load <span class="text-slate-100">{cpuPct.toFixed(1)}%</span>
                </div>
                <div class="text-xs font-mono text-slate-600">{t('2 Hz sampling · all-core average')}</div>
              </div>
            </div>
          {/if}
          <!-- RAM -->
          {#if ram}
            <div class="flex items-center gap-4">
              <Donut pct={ramPct} base="violet" label="RAM"
                sub="{(ram.used_mb / 1024).toFixed(1)}/{(ram.total_mb / 1024).toFixed(0)} GB" />
              <div class="min-w-0 flex-1 space-y-1">
                <div class="text-sm font-mono font-semibold text-slate-200">RAM</div>
                <div class="text-sm font-mono text-slate-500">
                  used <span class="text-slate-100">{(ram.used_mb / 1024).toFixed(1)} GB</span>
                </div>
                <div class="text-sm font-mono text-slate-500">
                  free <span class="text-slate-100">{(ram.free_mb / 1024).toFixed(1)} GB</span>
                </div>
              </div>
            </div>
          {/if}
        </div>
      </section>
    {/if}
    </div>
    {:else}
    <!-- ============== Compact layout: the instrument cluster ==============
         One tile per piece of hardware, each led by the number it is about.
         No time series anywhere: see the note at the top of Gauge.svelte. -->
    {@const igpus = vram.gpus.filter(g => g.integrated && !gpuList.includes(g))}
    {@const pw = vram.power}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      {#if vram.probe_warning}{@render probeWarning(vram.probe_warning)}{/if}
      <!-- Accelerators: the cards a model is actually planned onto. -->
      {#if hasGpu}
        <div class="grid gap-3" style="grid-template-columns: repeat(auto-fit, minmax(330px, 1fr))">
          {#each gpuList as gpu (gpu.vendor + '' + gpu.index)}
            {@const usedPct = gpu.total_mb > 0 ? (gpu.used_mb / gpu.total_mb) * 100 : 0}
            {@const estPct = gpu.total_mb > 0 ? Math.min(100, (vram.active_estimate_mb / gpu.total_mb) * 100) : 0}
            {@const hasSensors = gpu.util_percent != null || gpu.temp_c != null || gpu.fan_percent != null || gpu.power_w != null}
            <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5 space-y-3">
              <div class="flex items-baseline justify-between gap-2">
                <span class="text-xs font-mono text-slate-200 truncate" title={gpu.name}>{gpu.name}</span>
                <span class="shrink-0 text-[10px] font-mono uppercase tracking-widest text-slate-600">
                  <span lang="en">{gpu.vendor ?? 'gpu'} · #{gpu.index}</span>{gpu.unified ? ` · ${t('unified')}` : ''}
                </span>
              </div>
              <Meter
                size="lg" segments={28} escalate label="VRAM"
                value={(gpu.used_mb / 1024).toFixed(1)}
                unit="/ {(gpu.total_mb / 1024).toFixed(0)} GB"
                pct={usedPct} mark={estPct}
                sub="{usedPct.toFixed(0)}% · free {(gpu.free_mb / 1024).toFixed(1)} GB"
                title={vram.active_estimate_mb > 0 ? t('The amber mark is what the active presets are estimated to need.') : ''}
              />
              {#if hasSensors}
                <div class="grid grid-cols-[repeat(auto-fit,minmax(5.5rem,1fr))] gap-3 border-t border-slate-800 pt-3">
                  <Meter size="sm" segments={8} tone="cyan" label="load"
                    value={sensor(gpu.util_percent)} unit="%" pct={gpu.util_percent ?? 0} />
                  <Meter size="sm" segments={8} tone={tempTone(gpu.temp_c ?? gpu.hotspot_c)} label="temp"
                    value={sensor(gpu.temp_c ?? gpu.hotspot_c)} unit="°C" pct={tempPct(gpu.temp_c ?? gpu.hotspot_c)} />
                  <Meter size="sm" segments={8} tone="violet" label="fan"
                    value={sensor(gpu.fan_percent)} unit="%" pct={gpu.fan_percent ?? 0} />
                  <Meter size="sm" segments={0} tone="amber" label="power"
                    value={sensor(gpu.power_w)} unit="W" />
                </div>
                {#if gpu.hotspot_c != null || gpu.mem_temp_c != null || gpu.clock_mhz != null || gpu.fan_rpm != null}
                  <div class="flex flex-wrap gap-x-4 text-[10px] font-mono text-slate-600 tabular-nums">
                    {#if gpu.hotspot_c != null}
                      <span title={t('The junction probe — the one the card throttles on.')}>hotspot <span class="text-slate-400">{sensor(gpu.hotspot_c)} °C</span></span>
                    {/if}
                    {#if gpu.mem_temp_c != null}<span>mem <span class="text-slate-400">{sensor(gpu.mem_temp_c)} °C</span></span>{/if}
                    {#if gpu.fan_rpm != null}<span>fan <span class="text-slate-400">{gpu.fan_rpm} rpm</span></span>{/if}
                    {#if gpu.clock_mhz != null}<span>clock <span class="text-slate-400">{gpu.clock_mhz} MHz</span></span>{/if}
                  </div>
                {/if}
              {:else}
                <div class="text-[10px] font-mono text-slate-600">{t('This card reports no sensors.')}</div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5">
          <div class="text-[10px] font-mono uppercase tracking-widest text-slate-500">VRAM</div>
          <div class="mt-1 text-[11px] font-mono text-slate-500">{t('No GPU memory telemetry on this machine — running in CPU mode. Models are planned against system RAM below.')}</div>
        </div>
      {/if}

      <!-- The rest of the box: host memory, host CPU, what the wall sees. -->
      <div class="grid gap-3" style="grid-template-columns: repeat(auto-fit, minmax(215px, 1fr))">
        {#if vram.cpu_percent != null}
          <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5 space-y-3">
            <Meter size="md" segments={20} escalate tone="cyan" label="CPU load"
              value={vram.cpu_percent.toFixed(0)} unit="%" pct={vram.cpu_percent}
              sub={t('all-core')} />
            <div class="grid grid-cols-2 gap-3 border-t border-slate-800 pt-3">
              <Meter size="sm" segments={0} tone={tempTone(vram.cpu_temp_c)} label="temp"
                value={sensor(vram.cpu_temp_c)} unit="°C" />
              <Meter size="sm" segments={0} tone="amber" label="power"
                value={sensor(pw?.cpu_w)} unit="W" />
            </div>
          </div>
        {/if}
        {#if vram.ram}
          {@const ram = vram.ram}
          <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5 space-y-3">
            <Meter size="md" segments={20} escalate tone="violet" label="RAM"
              value={(ram.used_mb / 1024).toFixed(1)} unit="/ {(ram.total_mb / 1024).toFixed(0)} GB"
              pct={ram.percent ?? 0} sub="{(ram.percent ?? 0).toFixed(0)}%" />
            <div class="grid grid-cols-2 gap-3 border-t border-slate-800 pt-3">
              <Meter size="sm" segments={0} tone="slate" label="free"
                value={(ram.free_mb / 1024).toFixed(1)} unit="GB" />
              <Meter size="sm" segments={0} tone="slate" label="used"
                value={(ram.used_mb / 1024).toFixed(1)} unit="GB" />
            </div>
          </div>
        {/if}
        {#if pw}
          {@const wh = pw.energy_wh ?? 0}
          <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5 space-y-3">
            <Meter size="md" segments={0} tone="amber"
              label={t('Power draw')} labelLang={tLang('Power draw')}
              value={sensor(pw.total_w)} unit="W" sub={t('GPU + CPU')} />
            <div class="grid grid-cols-[repeat(auto-fit,minmax(5.5rem,1fr))] gap-3 border-t border-slate-800 pt-3">
              <Meter size="sm" segments={0} tone="emerald" label="gpu" value={sensor(pw.gpu_w)} unit="W" />
              <Meter size="sm" segments={0} tone="cyan" label="cpu" value={sensor(pw.cpu_w)} unit="W" />
              <Meter size="sm" segments={0} tone="amber"
                label={t('energy')} labelLang={tLang('energy')}
                value={wh >= 1000 ? (wh / 1000).toFixed(2) : wh.toFixed(1)} unit={wh >= 1000 ? 'kWh' : 'Wh'}
                title={t('Counted only while a model is actually working.')} />
            </div>
          </div>
        {/if}
        {#each igpus as ig (ig.vendor + '' + ig.index)}
          <div class="rounded border border-slate-800 bg-slate-900/60 p-3.5 space-y-3"
            title={t('Drives the display. Its memory is system RAM, so it is never counted as a VRAM budget.')}>
            <Meter size="md" segments={20} tone="slate" escalate label="iGPU load"
              value={sensor(ig.util_percent)} unit="%" pct={ig.util_percent ?? 0}
              sub={t('display')} />
            <div class="grid grid-cols-2 gap-3 border-t border-slate-800 pt-3">
              <Meter size="sm" segments={0} tone={tempTone(ig.temp_c)} label="temp"
                value={sensor(ig.temp_c)} unit="°C" />
              <Meter size="sm" segments={0} tone="slate" label="shared"
                value={(ig.used_mb / 1024).toFixed(1)} unit="GB" />
            </div>
          </div>
        {/each}
      </div>

      <!-- Detail that answers "why is VRAM full" — one click, still no charts. -->
      {#if (vram.processes && vram.processes.length > 0) || tokensRows.length > 0}
        <details class="pt-1 group">
          <summary class="cursor-pointer list-none text-[10px] uppercase tracking-widest text-slate-500 font-mono hover:text-slate-300 select-none">
            <span class="inline-block transition-transform group-open:rotate-90">▸</span>
            {t('VRAM consumers')} · {t('By preset')}
          </summary>
          <div class="mt-3 grid gap-6 lg:grid-cols-2">
            <div class="space-y-1.5">
              {#if vram.processes && vram.processes.length > 0}
                {#each vram.processes as p (p.pid)}
                  {@const pct = totalMb > 0 ? (p.used_mb / totalMb) * 100 : 0}
                  {@const labelColor = p.preset ? (p.adopted ? 'text-amber-300' : 'text-emerald-300') : 'text-slate-300'}
                  <div class="flex items-center gap-3 text-xs font-mono">
                    <span class="min-w-0 flex-1 truncate {labelColor}">{p.model ?? p.process_name.split('/').pop()}</span>
                    <span class="text-slate-600 text-[11px] shrink-0">{p.preset ? `preset: ${p.preset}` : t('PID {pid} · external', { pid: p.pid })}</span>
                    <span class="shrink-0 tabular-nums text-slate-200 w-20 text-right">{(p.used_mb / 1024).toFixed(2)} GB</span>
                    <span class="shrink-0 tabular-nums text-slate-600 text-[11px] w-12 text-right">{pct.toFixed(1)}%</span>
                  </div>
                {/each}
              {:else}
                <div class="text-xs text-slate-600 font-mono">—</div>
              {/if}
            </div>
            <div class="space-y-1.5">
              {#if tokensRows.length > 0}
                {#each tokensRows as r (r.name)}
                  <div class="flex items-center gap-3 text-xs font-mono">
                    <span class="min-w-0 flex-1 truncate text-emerald-300">{r.label}</span>
                    <span class="text-slate-600 text-[11px] shrink-0">{r.live > 0.5 ? `${r.live.toFixed(1)} tok/s` : 'idle'}</span>
                    <span class="shrink-0 tabular-nums text-emerald-200 text-[11px]">{r.decoded.toLocaleString()}<span class="text-slate-600"> dec</span></span>
                    <span class="shrink-0 tabular-nums text-cyan-300/80 text-[11px]">{r.prompt.toLocaleString()}<span class="text-slate-600"> pr</span></span>
                  </div>
                {/each}
              {:else}
                <div class="text-xs text-slate-600 font-mono">{t('No active presets.')}</div>
              {/if}
            </div>
          </div>
        </details>
      {/if}
    </section>
    {/if}
  {/if}

  {#if !loaded}
    <div class="grid gap-4 lg:grid-cols-2">
      <Skeleton class="h-40 w-full" />
      <Skeleton class="h-40 w-full" />
    </div>
  {/if}

  {#if loaded && running.length === 0}
    <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-6 text-sm text-slate-400">
      {t('No running presets.')} {t('To start a preset or adopt an externally running llama-server, use the')} <a href="/server" class="text-emerald-400 hover:underline">Server</a> {t('page.')}
    </div>
  {/if}

  <!-- Active preset cards -->
  {#each running as s (s.name)}
    {@const f = frames[s.name]}
    {@const hist = history[s.name] ?? []}
    {@const rec = recDefaults[s.name] ?? null}
    {@const recT = rec?.sampling?.temperature}
    {@const recK = rec?.sampling?.top_k}
    {@const recP = rec?.sampling?.top_p}
    {@const liveDecode = rollingTps(hist, 'instant_decode_tps', 3)}
    {@const rollingVals = hist.map((_, i, a) => rollingTps(a.slice(0, i + 1), 'instant_decode_tps', 3))}
    {@const activeJobs = f?.active_jobs ?? []}
    {@const recentJobs = f?.recent_jobs ?? []}
    {@const sparkMax = Math.max(1, ...rollingVals)}
    {@const kvPct = f?.kv_cache_max_tokens ? Math.min(100, (f.kv_cache_used_tokens / f.kv_cache_max_tokens) * 100) : 0}
    <!-- Slot count follows the live server (its actual -np), falling back to the
         preset's `parallel` only before metrics arrive. Never Math.max over a
         stale config — that showed 8 slots for a 1-slot preset. -->
    {@const slotCount = (f?.total_slots ?? 0) > 0 ? f!.total_slots : (f?.slots?.length || s.config.parallel || 1)}
    {@const perSlotCtx = Math.round(s.config.ctx_size / Math.max(1, slotCount))}
    {@const cardLabel = modelLabel(s.config, s.name)}
    <!-- Router presets: the loaded model is the headline; the preset name demotes to a
         small chip. Never repeat the same string as title + preset + port. -->
    {@const cardTitle = f?.loaded_model_id ?? cardLabel}
    {@const presetChip = cardTitle !== s.name}
    {@const portShown = !(presetChip ? s.name : cardTitle).includes(String(s.port))}
    {@const busyIdx = Array.from({ length: slotCount }, (_, i) => i).filter(i => f?.slots?.[i]?.is_processing)}
    {@const idleCount = slotCount - busyIdx.length}
    {@const idleCtx = f?.slots?.find(sl => !sl.is_processing)?.n_ctx ?? perSlotCtx}
    {#if layout === 'compact'}
    <!-- Compact preset card: two dials and a block of numbers. The tok/s
         sparkline that used to be the hero here is gone — the reading is the
         point, and the only history worth keeping is the peak tick. -->
    {@const livePrompt = rollingTps(hist, 'instant_prompt_tps', 3)}
    {@const peakDecode = Math.max(0, ...rollingSeries(hist, 'instant_decode_tps', 3))}
    {@const peakPrompt = Math.max(0, ...rollingSeries(hist, 'instant_prompt_tps', 3))}
    {@const decodedTot = f?.prom?.['llamacpp:tokens_predicted_total'] ?? 0}
    {@const promptTot = f?.prom?.['llamacpp:prompt_tokens_total'] ?? 0}
    {@const parallel = s.config.parallel || slotCount}
    <!-- The prompt the busy slot is on. A long chat re-sends its whole
         history, so what took time is `processed`, not `n_prompt_tokens` —
         and when the cache covers nearly all of it, that is the reason a
         160k-token turn came back instantly. -->
    {@const busySlot = f?.slots?.find(sl => sl.is_processing)}
    {@const promptTokens = busySlot?.n_prompt_tokens ?? 0}
    {@const cachedTokens = busySlot?.n_prompt_tokens_cache ?? 0}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3 flex-wrap">
          <h2 class="text-base font-mono text-emerald-400 break-all" title={f?.loaded_model_id ? t('Model currently loaded on this router') : undefined}>{cardTitle}</h2>
          {#if presetChip}
            <span class="text-xs text-slate-500 font-mono">{s.config.mode === 'router' ? 'router' : 'preset'}: {s.name}</span>
          {/if}
          {#if portShown}
            <span class="text-xs text-slate-500 font-mono">:{s.port}</span>
          {/if}
          <StatusPill adopted={s.adopted} pid={s.pid} />
          <span class="text-xs text-slate-500 font-mono">up {formatUptime(s.uptime_seconds)}</span>
        </div>
        {#if f?.error}
          <span class="text-xs text-amber-400/90 font-mono">{f.error}</span>
        {/if}
      </div>

      <div class="flex flex-wrap items-center gap-x-6 gap-y-4">
        <!-- The two rates that describe every llama.cpp run: how fast it reads
             the prompt, how fast it writes the answer. -->
        <div class="flex gap-2">
          <Gauge
            value={liveDecode} peak={peakDecode}
            max={niceMax(Math.max(peakDecode, liveDecode) * 1.05)}
            label="decode · tg" unit="tok/s" tone="emerald" digits={1}
            sub={t('peak {p} · last {n} s', { p: fmt(peakDecode, 1), n: (hist.length / 2).toFixed(0) })}
            title={t('Tokens the model is writing, averaged over 3 s. The tick marks the fastest it went in the last 30 s.')}
          />
          <Gauge
            value={livePrompt} peak={peakPrompt}
            max={niceMax(Math.max(peakPrompt, livePrompt) * 1.05)}
            label="prompt · pp" unit="tok/s" tone="cyan" digits={0}
            sub={t('peak {p} · last {n} s', { p: fmt(peakPrompt, 0), n: (hist.length / 2).toFixed(0) })}
            title={t('Prompt tokens being ingested. It sits at zero between requests — prefill is bursty, and the peak is the rate that burst reached.')}
          />
        </div>
        <!-- auto-fit, so a wide card gets four narrow columns instead of two
             wide ones with their captions stranded at the far edge. -->
        <div class="grid gap-x-5 gap-y-3.5 min-w-[240px] flex-1"
          style="grid-template-columns: repeat(auto-fit, minmax(165px, 1fr))">
          <Meter size="md" segments={10} label="active" tone="emerald"
            value={String(f?.requests_processing ?? 0)} unit="/ {parallel}"
            pct={parallel > 0 ? ((f?.requests_processing ?? 0) / parallel) * 100 : 0}
            sub={t('processing now')} />
          <Meter size="md" segments={0} label="queued"
            tone={(f?.requests_deferred ?? 0) > 0 ? 'amber' : 'slate'}
            value={String(f?.requests_deferred ?? 0)} sub={t('waiting in queue')} />
          <Meter size="md" segments={10} escalate tone="violet" label="kv-cache"
            value={fmt(kvPct, 1)} unit="%" pct={kvPct}
            sub="{compactNum(f?.kv_cache_used_tokens ?? 0)}/{compactNum(f?.kv_cache_max_tokens ?? 0)}" />
          <Meter size="md" segments={0} tone="cyan" label="tokens"
            value={compactNum(decodedTot)} unit="dec"
            sub="{compactNum(promptTot)} pr" />
        </div>
      </div>

      {#if promptTokens > 0}
        <div class="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[11px] font-mono text-slate-500 tabular-nums">
          <span>prompt <span class="text-slate-200">{compactNum(promptTokens)}</span> tok</span>
          <span>cached <span class="text-emerald-300">{((cachedTokens / promptTokens) * 100).toFixed(0)}%</span>
            <span class="text-slate-600">({compactNum(cachedTokens)})</span></span>
          <span>to run <span class="text-cyan-300">{compactNum(busySlot?.n_prompt_tokens_processed ?? 0)}</span></span>
        </div>
      {/if}

      <div class="flex items-center gap-2 flex-wrap border-t border-slate-800 pt-3">
        <span class="text-[10px] uppercase tracking-widest text-slate-500 font-mono mr-1">
          slots <span class="text-slate-300 tabular-nums">{f?.busy_slots ?? 0}/{f?.total_slots ?? slotCount}</span>
        </span>
        {#each Array(slotCount) as _, i}
          {@const slot = f?.slots?.[i]}
          {@const busy = slot?.is_processing ?? false}
          {@const pct = slotPct(slot)}
          <span class="inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-mono {busy ? 'border-emerald-700 bg-emerald-950/30 text-emerald-300' : 'border-slate-800 bg-slate-900/40 text-slate-500'}"
            title={busy && slot ? `task ${slot.task_id} · ${slot.n_decoded} tok` : `ctx ${(slot?.n_ctx ?? perSlotCtx).toLocaleString()}`}>
            <span class="h-1.5 w-1.5 rounded-full {busy ? 'bg-emerald-400 animate-pulse' : 'bg-slate-700'}"></span>
            {slot?.id ?? i}
            {#if busy}
              <span class="inline-block h-1 w-10 rounded-full bg-slate-800 overflow-hidden"><span class="block h-full bg-emerald-500" style="width: {pct}%"></span></span>
              <span class="tabular-nums text-emerald-200">{slot?.n_decoded ?? 0}</span>
            {/if}
          </span>
        {/each}
      </div>
    </section>
    {:else}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3 flex-wrap">
          <h2 class="text-lg font-mono text-emerald-400 break-all" title={f?.loaded_model_id ? t('Model currently loaded on this router') : undefined}>{cardTitle}</h2>
          {#if presetChip}
            <span class="text-xs text-slate-500 font-mono">{s.config.mode === 'router' ? 'router' : 'preset'}: {s.name}</span>
          {/if}
          {#if portShown}
            <span class="text-xs text-slate-500 font-mono">:{s.port}</span>
          {/if}
          <StatusPill adopted={s.adopted} pid={s.pid} />
          <span class="text-xs text-slate-500 font-mono">up {formatUptime(s.uptime_seconds)}</span>
        </div>
        {#if f?.error}
          <span class="text-xs text-rose-400 font-mono">{t('metrics error: {e}', { e: f.error })}</span>
        {/if}
      </div>

      <!-- Stat row -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
          <div class="text-xs uppercase tracking-wider text-slate-500">decode tok/s <span class="text-[11px] text-slate-600">{t('(3 s avg)')}</span></div>
          <div class="text-2xl font-mono text-emerald-400 leading-tight">{fmt(liveDecode, 1)}</div>
          {#if activeJobs.length > 0}
            {@const j = activeJobs[0]}
            <div class="text-[11px] text-slate-500 font-mono">
              {t('active task')} <span class="text-slate-300">#{j.task_id}</span> ·
              <span class="text-slate-300">{j.tokens_decoded}</span> tok ·
              avg <span class="text-slate-300">{fmt(j.avg_decode_tps, 1)}</span>
            </div>
          {:else if recentJobs.length > 0}
            {@const j = recentJobs[recentJobs.length - 1]}
            <div class="text-[11px] text-slate-500 font-mono">
              {t('last task')} <span class="text-slate-300">#{j.task_id}</span> ·
              <span class="text-slate-300">{j.tokens_decoded}</span> tok ·
              avg <span class="text-slate-300">{fmt(j.avg_decode_tps, 1)}</span>
            </div>
          {:else}
            <div class="text-[11px] text-slate-500 font-mono">{t('no jobs yet')}</div>
          {/if}
          <div class="text-[11px] text-slate-600 font-mono">lifetime {fmt(f?.lifetime_decode_tps, 1)}</div>
        </div>
        <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
          <div lang="en" class="text-xs uppercase tracking-wider text-slate-500">active</div>
          <div class="text-2xl font-mono text-slate-200 leading-tight">{f?.requests_processing ?? 0}<span class="text-sm text-slate-500">/{s.config.parallel}</span></div>
          <div class="text-[11px] text-slate-500 font-mono">{t('processing now')}</div>
        </div>
        <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
          <div class="text-xs uppercase tracking-wider text-slate-500">queued</div>
          <div class="text-2xl font-mono leading-tight {(f?.requests_deferred ?? 0) > 0 ? 'text-amber-400' : 'text-slate-200'}">{f?.requests_deferred ?? 0}</div>
          <div class="text-[11px] text-slate-500 font-mono">{t('waiting in queue')}</div>
        </div>
        <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
          <div class="text-xs uppercase tracking-wider text-slate-500">kv-cache</div>
          <div class="text-2xl font-mono text-violet-400 leading-tight">{fmt(kvPct, 1)}%</div>
          <div class="text-[11px] text-slate-500 font-mono">{fmt(f?.kv_cache_used_tokens ?? 0)} / {fmt(f?.kv_cache_max_tokens ?? 0)} tok</div>
        </div>
      </div>

      <!-- Sparkline (rolling decode tok/s over time) -->
      <div class="rounded bg-slate-900/60 border border-slate-800 p-3">
        <div class="flex items-center justify-between text-xs text-slate-500 font-mono mb-1.5">
          <span>{t('last {n} s ·', { n: (hist.length / 2).toFixed(0) })} <span class="text-emerald-400">decode tok/s</span> {t('(3 s rolling avg)')}</span>
          <span class="text-slate-600">peak {fmt(sparkMax, 1)} · {t('{a} active · {b} completed', { a: activeJobs.length, b: recentJobs.length })}</span>
        </div>
        <div class="relative">
          <svg viewBox="0 0 600 80" width="100%" height="80" class="block" preserveAspectRatio="none">
            {#each [0.25, 0.5, 0.75] as g}
              <line x1="0" x2="600" y1={80 * g} y2={80 * g} class="stroke-slate-800" stroke-width="1" />
            {/each}
            <path d={sparklinePath(rollingVals, 600, 80, sparkMax)} fill="none" class="stroke-emerald-400" stroke-width="2" vector-effect="non-scaling-stroke" />
          </svg>
          <span class="pointer-events-none absolute left-1 top-0 text-[10px] font-mono text-slate-500 tabular-nums bg-slate-900/70 px-1 rounded-sm">{fmt(sparkMax, 1)}</span>
          <span class="pointer-events-none absolute left-1 top-1/2 -translate-y-1/2 text-[10px] font-mono text-slate-600 tabular-nums bg-slate-900/70 px-1 rounded-sm">{fmt(sparkMax / 2, 1)}</span>
          <span class="pointer-events-none absolute left-1 bottom-0 text-[10px] font-mono text-slate-600 tabular-nums bg-slate-900/70 px-1 rounded-sm">0</span>
        </div>
      </div>

      <!-- Slot grid — detailed cards -->
      <div>
        <div class="flex items-center justify-between mb-2 flex-wrap gap-2">
          <div class="text-xs uppercase tracking-wider text-slate-500">slots · {f?.busy_slots ?? 0}/{f?.total_slots ?? slotCount} busy</div>
          {#if rec && rec.source !== 'none'}
            <div class="text-[11px] font-mono text-slate-500">
              {t('recommended')}
              <span class="text-slate-300">T={recT != null ? recT.toFixed(2) : '—'}</span>
              <span class="text-slate-600">·</span>
              <span class="text-slate-300">k={recK != null ? Math.round(recK) : '—'}</span>
              <span class="text-slate-600">·</span>
              <span class="text-slate-300">p={recP != null ? recP.toFixed(2) : '—'}</span>
              <span class="ml-1 rounded border px-1 py-0 text-[11px] {rec.source === 'gguf' ? 'border-emerald-800 text-emerald-400 bg-emerald-950/40' : rec.source === 'props' ? 'border-cyan-800 text-cyan-400 bg-cyan-950/40' : 'border-amber-800 text-amber-400 bg-amber-950/40'}">{rec.source}</span>
            </div>
          {/if}
        </div>
        <!-- Busy slots get full cards; idle slots collapse into one summary line.
             An all-idle 16-slot router used to print a screenful of "idle" cards. -->
        {#if busyIdx.length > 0}
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {#each busyIdx as i (i)}
            {@const slot = f?.slots?.[i]}
            {@const pct = slotPct(slot)}
            <div class="rounded border border-emerald-700 bg-emerald-950/30 p-3 space-y-2">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="text-xs font-mono text-emerald-300">slot {slot?.id ?? i}</span>
                  <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                </div>
                <span class="text-xs font-mono text-slate-600">task {slot?.task_id}</span>
              </div>

              {#if slot}
                <div class="flex justify-between text-sm font-mono">
                  <span class="text-emerald-300">{slot.n_decoded} tok</span>
                  <span class="text-slate-500">/ {slot.max_tokens ?? (slot.n_decoded + slot.n_remain)}</span>
                </div>
                <div class="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div class="h-full bg-emerald-500 transition-all" style="width: {pct}%"></div>
                </div>
                <div class="flex justify-between text-xs font-mono">
                  <span class={driftTag(slot.temperature, recT) === 'off' ? 'text-amber-400' : 'text-slate-500'} title={recT != null ? t('recommended: {v}', { v: recT.toFixed(2) }) : ''}>T={slot.temperature?.toFixed(2) ?? '—'}</span>
                  <span class={driftTag(slot.top_k, recK, 0.5) === 'off' ? 'text-amber-400' : 'text-slate-500'} title={recK != null ? t('recommended: {v}', { v: Math.round(recK) }) : ''}>k={slot.top_k ?? '—'}</span>
                  <span class={driftTag(slot.top_p, recP) === 'off' ? 'text-amber-400' : 'text-slate-500'} title={recP != null ? t('recommended: {v}', { v: recP.toFixed(2) }) : ''}>p={slot.top_p?.toFixed(2) ?? '—'}</span>
                </div>
              {/if}
            </div>
          {/each}
          {#if idleCount > 0}
            <div class="rounded border border-slate-800 bg-slate-900/40 p-3 flex items-center gap-2 text-xs font-mono text-slate-500">
              <span class="h-2 w-2 rounded-full bg-slate-700"></span>
              {t('{n} idle · ctx {c}/slot', { n: idleCount, c: idleCtx.toLocaleString() })}
            </div>
          {/if}
        </div>
        {:else}
        <div class="rounded border border-slate-800 bg-slate-900/40 px-3 py-2 flex items-center gap-2 text-xs font-mono text-slate-500">
          <span class="h-2 w-2 rounded-full bg-slate-700"></span>
          {t('all idle · ctx {c}/slot', { c: idleCtx.toLocaleString() })}
        </div>
        {/if}
      </div>
    </section>
    {/if}
  {/each}
</div>

{#if restarting}
  <div class="fixed inset-0 z-[70] bg-black/80 backdrop-blur-sm flex items-center justify-center p-6" role="alertdialog" aria-modal="true" aria-label={t('Restarting backend')}>
    <div class="w-full max-w-sm rounded-lg border border-slate-700 bg-slate-900 p-6 text-center space-y-4 shadow-2xl">
      {#if !restartFailed}
        <div class="mx-auto h-9 w-9 animate-spin rounded-full border-2 border-slate-700 border-t-emerald-400"></div>
        <div class="text-sm text-slate-200">{restartMsg}</div>
        <div class="text-xs font-mono text-slate-500">{restartElapsed}s · {t('managed llama-server processes stay alive')}</div>
      {:else}
        <div class="mx-auto flex h-9 w-9 items-center justify-center rounded-full border-2 border-rose-800 text-rose-300">✕</div>
        <div class="text-sm text-rose-200">{restartMsg}</div>
        <div class="text-xs text-slate-500">{t('The backend may still be finishing a model scan. Try reloading, or check the service with: systemctl status llamadeck (or your launcher).')}</div>
        <div class="flex justify-center gap-2 pt-1">
          <button
            onclick={() => { restarting = false; restartFailed = false; }}
            class="rounded bg-slate-700/40 border border-slate-600 px-4 py-1.5 text-sm hover:bg-slate-700/60"
          >{t('Close')}</button>
          <button
            onclick={() => location.reload()}
            class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60"
          >{t('Reload now')}</button>
        </div>
      {/if}
    </div>
  </div>
{/if}
