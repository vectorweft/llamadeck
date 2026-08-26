<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { api, fitBlock, formatUptime, offloadGpus, type PresetStatus, type ScanEntry, type VramReport } from '$lib/api';
  import { modelLabel } from '$lib/ui';
  import { confirmDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';
  import Skeleton from '$lib/components/Skeleton.svelte';
  import StatusPill from '$lib/components/StatusPill.svelte';

  let statuses = $state<Record<string, PresetStatus>>({});
  let scanned = $state<ScanEntry[]>([]);
  let vram = $state<VramReport | null>(null);
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;
  let loaded = $state(false);

  const order = $derived.by(() => {
    return Object.values(statuses).sort((a, b) => a.port - b.port);
  });
  // Hidden presets ("Hide" on the Presets page) sit in a collapsed section
  // while stopped; a running preset consumes VRAM so it is always visible.
  const mainList = $derived(order.filter(s => s.running || !s.config.ui_hidden));
  const hiddenIdle = $derived(order.filter(s => !s.running && s.config.ui_hidden));
  let showHidden = $state(false);
  // Router kartı rozeti: çalışan router'ın bellekteki model sayısı.
  let routerLoadedCount = $state<number | null>(null);
  // Prefer the live estimate (scales with ctx/np/kv-quant + GGUF geometry);
  // fall back to the preset's static value when GGUF can't be parsed.
  function estMb(s: PresetStatus): number {
    // GPU share only — weights parked in host RAM (--n-cpu-moe / low ngl)
    // don't count against the VRAM budget.
    const e = s.vram_estimate;
    if (e) return e.gpu_mb ?? e.total_mb;
    return s.config.estimated_vram_mb ?? 0;
  }
  const activeTotalMb = $derived(
    order.filter(s => s.running).reduce((sum, s) => sum + estMb(s), 0)
  );

  async function refresh() {
    try {
      const [s, sc, v] = await Promise.all([api.serverStatuses(), api.serverScan(), api.serverVram()]);
      statuses = s.presets;
      scanned = sc.found;
      vram = v;
      error = null;
      loaded = true;
      const routerRunning = Object.values(s.presets).some(p => p.running && p.config.mode === 'router');
      if (routerRunning) {
        try {
          const rm = await api.routerModels();
          routerLoadedCount = (rm.data ?? []).filter(m => ['loaded', 'sleeping', 'loading'].includes(m.status?.value ?? '')).length;
        } catch { routerLoadedCount = null; }
      } else {
        routerLoadedCount = null;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function act(name: string, fn: () => Promise<unknown>, okMsg?: string) {
    busy = name;
    error = null;
    try {
      await fn();
      if (okMsg) toast(okMsg, 'success');
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
    finally { busy = null; await refresh(); }
  }

  // Start/switch/restart, but honour the backend's fit-check preflight: if it
  // blocks the start (the model would very likely OOM), show check_fit's
  // actionable headline and let the user push it through with force=true.
  async function actGuarded(name: string, run: (force: boolean) => Promise<unknown>, okMsg?: string) {
    busy = name;
    error = null;
    try {
      await run(false);
      if (okMsg) toast(okMsg, 'success');
    } catch (e) {
      const fb = fitBlock(e);
      if (!fb) { error = e instanceof Error ? e.message : String(e); return; }
      busy = null;
      const extra = (fb.messages ?? []).map(m => m.text).join('\n');
      const ok = await confirmDialog(fb.headline + (extra ? '\n\n' + extra : ''), {
        title: t('Start anyway?'), confirmLabel: t('Start'), danger: fb.level === 'too_big',
      });
      if (!ok) return;
      busy = name;
      try {
        await run(true);
        if (okMsg) toast(okMsg, 'success');
      } catch (e2) { error = e2 instanceof Error ? e2.message : String(e2); }
    } finally { busy = null; await refresh(); }
  }

  async function toggle(s: PresetStatus) {
    if (s.running) {
      if (s.adopted) {
        // Adopted presets were started outside LlamaDeck. The toggle now behaves
        // like a user would expect — "turn off" means kill the process and
        // free its VRAM. Release (stop tracking but leave the process alive)
        // is still available as a separate action.
        const ok = await confirmDialog(
          t('The llama-server on :{port} will be stopped and its VRAM freed.\n\nClients using port :{port} will lose their connection.', { port: s.port }),
          { title: t('Kill adopted PID {pid}?', { pid: s.pid ?? '?' }), danger: true, confirmLabel: 'Kill' }
        );
        if (!ok) return;
        await act(s.name, () => api.serverStop(s.name), t('Killed PID {pid}', { pid: s.pid ?? '?' }));
      } else {
        await act(s.name, () => api.serverStop(s.name), t('Stopped {name}', { name: s.name }));
      }
    } else {
      // The backend fit-check preflight now gates the start with a real
      // GPU+RAM plan (MoE offload, KV quant, ctx limits) — far better than the
      // old free_mb-vs-static-estimate guess this used to do client-side.
      await actGuarded(s.name, (force) => api.serverStart(s.name, force), t('Started {name}', { name: s.name }));
    }
  }

  async function releaseAdopted(s: PresetStatus) {
    const ok = await confirmDialog(
      t('LlamaDeck stops tracking PID {pid} but the process keeps running and VRAM stays occupied.\nTo free VRAM, use the red "Kill" button (or the toggle).', { pid: s.pid ?? '?' }),
      { title: t('Release {name}?', { name: s.name }), confirmLabel: t('Release') }
    );
    if (!ok) return;
    await act(s.name, () => api.serverRelease(s.name), t('Released {name}', { name: s.name }));
  }

  async function killAdopted(s: PresetStatus) {
    const ok = await confirmDialog(
      t('Clients using port :{port} will lose their connection.', { port: s.port }),
      { title: t('Kill PID {pid}?', { pid: s.pid ?? '?' }), danger: true, confirmLabel: 'Kill' }
    );
    if (!ok) return;
    await act(s.name, () => api.serverStop(s.name), t('Killed PID {pid}', { pid: s.pid ?? '?' }));
  }

  async function adoptInto(pid: number, preset: string | null) {
    await act(preset ?? `pid-${pid}`, () => api.serverAdopt(pid, preset), t('Adopted PID {pid} as {name}', { pid, name: preset ?? '?' }));
  }

  onMount(async () => {
    await refresh();
    timer = setInterval(refresh, 3000);
  });
  onDestroy(() => { if (timer) clearInterval(timer); });
</script>

<div class="max-w-5xl space-y-6">
  <div class="flex items-center gap-3">
    <h1 class="text-2xl font-semibold">Server</h1>
    <span class="text-xs text-slate-500 font-mono">{t('{a} / {b} active', { a: order.filter(s => s.running).length, b: order.length })}</span>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  <!-- VRAM panel. One bar per offload GPU: this rendered gpus[0] alone, so a
       second card was simply absent from the page — and on a two-card box the
       one it dropped is as likely to be the one filling up. -->
  {#if vram && offloadGpus(vram.gpus).length > 0}
    {@const gpus = offloadGpus(vram.gpus)}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      {#each gpus as gpu (gpu.vendor + ':' + gpu.index)}
        {@const usedPct = gpu.total_mb > 0 ? (gpu.used_mb / gpu.total_mb) * 100 : 0}
        <!-- The estimate is a machine-wide sum over running presets, with no
             per-device split behind it. Drawing it on one card of several would
             claim an attribution nobody computed, so the marker is for the
             single-GPU case and the number below stands on its own otherwise. -->
        {@const estPct = gpus.length === 1 && gpu.total_mb > 0 ? (activeTotalMb / gpu.total_mb) * 100 : 0}
        <div>
          <div class="flex items-baseline justify-between mb-2 gap-3 flex-wrap">
            <h2 class="text-sm uppercase tracking-wider text-slate-400">VRAM · {gpu.name}</h2>
            <span class="text-xs font-mono text-slate-400">
              {(gpu.used_mb / 1024).toFixed(1)} GB used · {(gpu.free_mb / 1024).toFixed(1)} GB free · {(gpu.total_mb / 1024).toFixed(1)} GB total
            </span>
          </div>
          <div class="relative h-6 w-full rounded bg-slate-800 overflow-hidden">
            <div class="absolute inset-y-0 left-0 bg-emerald-600/70" style="width: {usedPct}%"></div>
            {#if estPct > 0}
              <div class="absolute inset-y-0 border-l-2 border-amber-400" style="left: {Math.min(estPct, 99.5)}%" title={t('Active preset estimate: {mb} MB', { mb: activeTotalMb })}></div>
            {/if}
          </div>
        </div>
      {/each}
      {#if activeTotalMb > 0}
        <div class="text-xs text-slate-500 font-mono">
          {t('active preset estimate:')} {activeTotalMb} MB ({(activeTotalMb / 1024).toFixed(1)} GB)
          {#if gpus.length > 1}
            <span class="text-slate-600">· {t('across all GPUs')}</span>
          {/if}
        </div>
      {/if}
    </section>
  {/if}

  <!-- Adopt suggestions -->
  {#if scanned.length > 0}
    <section class="rounded-lg border border-amber-900 bg-amber-950/20 p-5 space-y-3">
      <h2 class="text-sm uppercase tracking-wider text-amber-300">{t('External llama-server found')}</h2>
      {#each scanned as p}
        <div class="rounded border border-amber-900/50 bg-slate-900/40 p-3 text-sm font-mono text-slate-300 flex items-center justify-between gap-4">
          <div class="min-w-0 flex-1 truncate">
            <span class="text-emerald-300">{p.config.model_path ? p.config.model_path.split('/').pop()!.replace(/\.gguf$/i, '') : t('(no model)')}</span> ·
            <span class="text-amber-300">PID {p.pid}</span> ·
            :{p.config.port} · ctx {p.config.ctx_size} · np {p.config.parallel}
            {#if p.suggested_preset}
              <span class="text-emerald-400"> → preset <b>{p.suggested_preset}</b></span>
            {:else}
              <span class="text-slate-500"> {t('(no preset matches port {port})', { port: p.config.port })}</span>
            {/if}
          </div>
          <button
            disabled={busy != null || !p.suggested_preset}
            onclick={() => adoptInto(p.pid, p.suggested_preset)}
            class="shrink-0 rounded bg-amber-700/40 border border-amber-600 px-3 py-1 text-xs hover:bg-amber-700/60 disabled:opacity-40"
          >{t('Adopt')}: {p.suggested_preset ?? '?'}</button>
        </div>
      {/each}
    </section>
  {/if}

  <!-- Preset list -->
  {#snippet serverCard(s: PresetStatus)}
      <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <div class="flex items-start gap-4">
          <!-- Toggle -->
          <button
            disabled={busy === s.name}
            onclick={() => toggle(s)}
            role="switch"
            aria-checked={s.running}
            aria-label="{s.running ? t('Stop') : t('Start')} preset {s.name}"
            class="mt-1 relative inline-flex h-6 w-12 shrink-0 items-center rounded-full border transition
                   {s.running ? 'bg-emerald-600 border-emerald-500' : 'bg-slate-800 border-slate-700'}
                   {busy === s.name ? 'opacity-50' : ''}"
            title={s.running ? (s.adopted ? t('Stop + free VRAM (kills PID {pid})', { pid: s.pid ?? '?' }) : t('Stop')) : t('Start')}
          >
            <span class="inline-block h-5 w-5 rounded-full bg-white shadow transition
                         {s.running ? 'translate-x-6' : 'translate-x-0.5'}"></span>
          </button>

          <div class="min-w-0 flex-1">
            <div class="flex items-center gap-3 flex-wrap">
              <h3 class="text-lg font-mono break-all {s.running ? 'text-emerald-400' : 'text-slate-300'}">{modelLabel(s.config, s.name)}</h3>
              <span class="text-xs text-slate-500 font-mono">preset: {s.name}</span>
              <span class="text-xs text-slate-500 font-mono">:{s.port}</span>
              {#if s.running}
                <StatusPill adopted={s.adopted} pid={s.pid} />
                <span class="text-xs text-slate-500 font-mono">· up {formatUptime(s.uptime_seconds)}</span>
                {#if s.rss_mb}
                  <span class="text-xs text-slate-500 font-mono">· rss {s.rss_mb.toFixed(0)} MB</span>
                {/if}
              {/if}
            </div>
            <!-- The diagnosis is a sentence with a next step in it, not an exit
                 code — give it its own full-width row so it stays readable. -->
            {#if !s.running && s.last_error}
              <div class="mt-2 rounded border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">
                {s.last_error}
              </div>
            {/if}
            <div class="text-xs text-slate-500 font-mono mt-1 truncate">
              {s.config.model_path ? s.config.model_path.substring(0, s.config.model_path.lastIndexOf('/')) + '/' : (s.config.hf_repo ?? t('(no model)'))}
            </div>
            <div class="text-xs text-slate-500 mt-1 flex gap-4 flex-wrap">
              <span>ctx {s.config.ctx_size}</span>
              <span>ngl {s.config.n_gpu_layers}</span>
              <span>np {s.config.parallel}</span>
              <span>fa {s.config.flash_attn}</span>
              <span>kv {s.config.cache_type_k}</span>
              {#if s.vram_estimate}
                {@const e = s.vram_estimate}
                <span
                  class="text-amber-400"
                  class:text-amber-500={e.source === 'approx'}
                  title={`model ${(e.model_mb/1024).toFixed(1)} GB · KV ${(e.kv_cache_mb/1024).toFixed(1)} GB · compute ${(e.compute_mb/1024).toFixed(1)} GB\nctx ${e.details.ctx_size} · kv ${e.details.cache_type_k}/${e.details.cache_type_v}${e.details.n_layers ? ` · layers ${e.details.n_layers} · kv_heads ${e.details.n_kv_heads}` : ''}\nsource: ${e.source}`}
                >~{((e.gpu_mb ?? e.total_mb) / 1024).toFixed(1)} GB VRAM{#if e.ram_mb}<span class="text-sky-400"> +{(e.ram_mb / 1024).toFixed(1)} GB RAM</span>{/if}{e.source === 'approx' ? '*' : ''}</span>
              {:else if s.config.estimated_vram_mb}
                <span class="text-amber-400" title={t('static value from the preset (no live geometry)')}>~{(s.config.estimated_vram_mb / 1024).toFixed(1)} GB VRAM</span>
              {/if}
              {#if s.config.jinja}<span class="text-emerald-400">jinja</span>{/if}
              {#if s.config.mode === 'router'}
                <a
                  href="/router"
                  class="text-sky-400 hover:underline"
                  title={t('Models inside this router process are managed on the Router page')}
                >{s.running && routerLoadedCount != null ? (routerLoadedCount === 1 ? t('1 model loaded') : t('{n} models loaded', { n: routerLoadedCount })) + ' · ' : ''}Router →</a>
              {/if}
            </div>
          </div>

          <!-- Actions -->
          <div class="flex gap-1 shrink-0">
            {#if s.running && !s.adopted}
              <button
                disabled={busy != null}
                onclick={() => actGuarded(s.name, (force) => api.serverRestart(s.name, force), t('Restarted {name}', { name: s.name }))}
                class="rounded bg-slate-700/40 border border-slate-600 px-2 py-1 text-xs hover:bg-slate-700/60 disabled:opacity-40"
              >Restart</button>
            {/if}
            {#if s.running && s.adopted}
              <button
                disabled={busy != null}
                onclick={() => releaseAdopted(s)}
                title={t('LlamaDeck stops tracking but the process keeps running (VRAM stays occupied)')}
                class="rounded bg-slate-700/40 border border-slate-600 px-2 py-1 text-xs hover:bg-slate-700/60 disabled:opacity-40"
              >{t('Release')}</button>
              <button
                disabled={busy != null}
                onclick={() => killAdopted(s)}
                class="rounded bg-rose-900/40 border border-rose-800 px-2 py-1 text-xs hover:bg-rose-900/60 disabled:opacity-40"
              >Kill</button>
            {/if}
          </div>
        </div>
      </div>
  {/snippet}

  <section class="space-y-3">
    {#if !loaded}
      <Skeleton class="h-24 w-full" />
      <Skeleton class="h-24 w-full" />
    {/if}
    {#each mainList as s (s.name)}
      {@render serverCard(s)}
    {:else}
      {#if loaded}<div class="text-sm text-slate-500 italic">
        {order.length > 0 ? t('All presets are hidden — expand them below.') : t('No presets yet. Create one on the Presets page.')}
      </div>{/if}
    {/each}

    {#if hiddenIdle.length > 0}
      <button
        onclick={() => showHidden = !showHidden}
        class="flex items-center gap-2 text-xs font-mono text-slate-500 hover:text-slate-300"
      >
        <span>{showHidden ? '▾' : '▸'}</span>
        {t('Hidden presets ({n})', { n: hiddenIdle.length })}
      </button>
      {#if showHidden}
        {#each hiddenIdle as s (s.name)}
          <div class="opacity-70">{@render serverCard(s)}</div>
        {/each}
      {/if}
    {/if}
  </section>
</div>
