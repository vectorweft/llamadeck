<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { page } from '$app/stores';
  import { api, type BenchJob, type BenchRecord, type BenchResult, type BenchRunBody, type ModelEntry, type PresetStatus, type VramReport } from '$lib/api';
  import { confirmDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';
  import ModelSelect from '$lib/components/ModelSelect.svelte';

  let models = $state<ModelEntry[]>([]);
  let statuses = $state<Record<string, PresetStatus>>({});
  let vram = $state<VramReport | null>(null);
  let historyRows = $state<BenchRecord[]>([]);
  let active = $state<BenchJob | { status: 'idle' } | null>(null);
  let log = $state<string[]>([]);
  let source: EventSource | null = null;
  let logContainer: HTMLDivElement | null = $state(null);
  let autoScroll = $state(true);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let activePoll: ReturnType<typeof setInterval> | null = null;

  // Form state
  let modelPath = $state<string>('');
  let nPrompts = $state('512, 2048');
  let nGens = $state('128');
  let pgPairs = $state('');       // "512,128 ; 2048,256"
  let ngl = $state(999);
  let batchSize = $state(2048);
  let ubatchSize = $state(512);
  let threads = $state<string>(''); // empty = auto
  let flashAttn = $state(true);
  let cacheTypeK = $state('q8_0');
  let cacheTypeV = $state('q8_0');
  let nDepth = $state(0);
  let repetitions = $state(3);
  let extraFlags = $state('');

  const isRunning = $derived(active && 'status' in active && active.status === 'running');

  // llama-bench has its own narrow flag set; many llama-server flags either
  // crash bench or are silently ignored. Catch the common offenders before
  // hitting the subprocess so the user sees an inline warning instead of an
  // exit-code-1 in the live log.
  const SERVER_ONLY_FLAGS = new Set([
    '--spec-default', '--jinja', '--metrics', '--slots',
    '--reasoning-format', '--no-context-shift',
    '--api-key', '--host', '--port', '-hf', '--hf-repo', '-hff', '--hf-file',
    '--mmproj', '--alias', '--model-alias',
    '--cont-batching', '--no-cont-batching', '-cb',
  ]);
  const flagWarnings = $derived.by<string[]>(() => {
    const tokens = extraFlags.split(/\s+/).filter(Boolean);
    const offenders: string[] = [];
    for (const t of tokens) {
      if (t.startsWith('-') && SERVER_ONLY_FLAGS.has(t) && !offenders.includes(t)) {
        offenders.push(t);
      }
    }
    return offenders;
  });

  const selectedModel = $derived(models.find(m => m.path === modelPath) ?? null);
  const runningWouldConflict = $derived.by(() => {
    // Any preset that's running on a process — running a bench loads the model
    // into VRAM again, which will OOM if we're close to full.
    return Object.values(statuses).filter(s => s.running);
  });
  const vramHeadroomOK = $derived.by(() => {
    if (!vram || !selectedModel) return true;
    // Rough: need size_gb + 1.5GB compute buffer
    const needed = selectedModel.size_gb * 1024 + 1500;
    return vram.free_mb >= needed;
  });

  async function refreshAll() {
    try {
      const [ms, sts, vr, a, h] = await Promise.all([
        api.listModels(),
        api.serverStatuses(),
        api.serverVram(),
        api.benchActive(),
        api.benchHistory(50, modelPath || null),
      ]);
      models = ms;
      statuses = sts.presets;
      vram = vr;
      active = a;
      historyRows = h;
      error = null;
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
  }

  function parseIntList(s: string): number[] {
    return s.split(/[,\s]+/).map(x => x.trim()).filter(Boolean).map(Number).filter(n => Number.isFinite(n) && n > 0);
  }
  function parsePgPairs(s: string): [number, number][] {
    const out: [number, number][] = [];
    for (const group of s.split(/;+/).map(x => x.trim()).filter(Boolean)) {
      const [pp, tg] = group.split(/[,\s]+/).map(Number);
      if (Number.isFinite(pp) && Number.isFinite(tg)) out.push([pp, tg]);
    }
    return out;
  }

  async function runBench() {
    if (!modelPath) { error = t('Pick a model first'); return; }
    if (!vramHeadroomOK) {
      const ok = await confirmDialog(
        t('The selected model may not fit in free VRAM.'),
        { title: t('Run anyway?'), confirmLabel: t('Run') }
      );
      if (!ok) return;
    }
    if (runningWouldConflict.length > 0) {
      const ok = await confirmDialog(
        t('There are still running presets ({names}).\nllama-bench will load the model once more; VRAM may overflow.', { names: runningWouldConflict.map(s => s.name).join(', ') }),
        { title: t('Continue anyway?'), confirmLabel: t('Continue') }
      );
      if (!ok) return;
    }
    busy = true;
    error = null;
    log = [];
    try {
      const body: BenchRunBody = {
        model_path: modelPath,
        n_prompts: parseIntList(nPrompts),
        n_gens: parseIntList(nGens),
        pg_pairs: parsePgPairs(pgPairs),
        n_gpu_layers: ngl,
        batch_size: batchSize,
        ubatch_size: ubatchSize,
        threads: threads.trim() ? Number(threads) : null,
        flash_attn: flashAttn,
        cache_type_k: cacheTypeK,
        cache_type_v: cacheTypeV,
        n_depth: nDepth,
        repetitions,
        extra_flags: extraFlags.split(/\s+/).filter(Boolean),
      };
      await api.benchRun(body);
      toast(t('Benchmark started'), 'success');
      openStream();
      await refreshAll();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally { busy = false; }
  }

  async function stopBench() {
    const ok = await confirmDialog(
      t('The running llama-bench process will be terminated. Partial results are discarded.'),
      { title: t('Stop the benchmark?'), confirmLabel: t('Stop') }
    );
    if (!ok) return;
    try {
      await api.benchCancel();
      toast(t('Benchmark cancelled'), 'info');
      await refreshAll();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function openStream() {
    source?.close();
    source = new EventSource('/api/bench/stream');
    source.onmessage = async (ev) => {
      try {
        const data = JSON.parse(ev.data);
        log = [...log, data.line];
        if (log.length > 5000) log = log.slice(-5000);
        if (autoScroll) {
          await tick();
          logContainer?.scrollTo({ top: logContainer.scrollHeight });
        }
      } catch { /* ignore */ }
    };
  }

  onMount(() => {
    // Support ?model=<path> query param from /models page
    const qp = $page.url.searchParams.get('model');
    if (qp) modelPath = qp;
    refreshAll();
    openStream();
    activePoll = setInterval(refreshAll, 4000);
  });
  onDestroy(() => {
    source?.close();
    if (activePoll) clearInterval(activePoll);
  });

  function fmtTs(ts: number | null): string {
    if (!ts) return '—';
    return new Date(ts * 1000).toLocaleString();
  }
  function fmtDur(s: number): string {
    if (s < 60) return `${s.toFixed(0)}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${Math.round(s % 60)}s`;
  }
  function fmtTps(r: BenchResult): string {
    const v = r.avg_ts;
    if (v == null) return '—';
    const sd = r.stddev_ts;
    return sd != null ? `${v.toFixed(1)} ± ${sd.toFixed(1)}` : v.toFixed(1);
  }
  function shortPath(p: string): string {
    const parts = p.split('/');
    return parts.slice(-3).join('/');
  }

  function testName(r: BenchResult): string {
    if (r.test) return r.test as string;
    const pp = r.n_prompt ?? 0, tg = r.n_gen ?? 0;
    if (pp > 0 && tg === 0) return `pp${pp}`;
    if (tg > 0 && pp === 0) return `tg${tg}`;
    if (pp > 0 && tg > 0) return `pp${pp}+tg${tg}`;
    return '—';
  }
</script>

<div class="max-w-6xl space-y-6">
  <div class="flex items-center gap-3">
    <h1 class="text-2xl font-semibold">Benchmark</h1>
    <span class="text-xs text-slate-500 font-mono">llama-bench · prompt processing + token generation</span>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  <!-- Config form -->
  <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="block lg:col-span-2">
        <span class="text-xs uppercase tracking-wider text-slate-400">model</span>
        <div class="mt-1">
          <ModelSelect {models} bind:value={modelPath} disabled={!!isRunning} />
        </div>
      </div>

      <label class="block">
        <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Prompt sizes to benchmark prompt-processing (pp) speed. Comma-separated; each value runs as its own test.')}>-p (prompt sizes)</span>
        <input bind:value={nPrompts} disabled={isRunning} placeholder="512, 2048, 4096"
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono" />
      </label>
      <label class="block">
        <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Generation lengths to benchmark token-generation (tg) speed. Comma-separated; each value runs as its own test.')}>-n (gen sizes)</span>
        <input bind:value={nGens} disabled={isRunning} placeholder="128, 256"
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono" />
      </label>
      <label class="block lg:col-span-2">
        <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Combined test: process pp prompt tokens then generate tg tokens, measuring end-to-end throughput. Separate multiple pairs with ;')}>-pg (combined pp,tg pairs; separate pairs with ;)</span>
        <input bind:value={pgPairs} disabled={isRunning} placeholder="512,128 ; 2048,256"
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono" />
      </label>

      <div class="grid grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] gap-2 lg:col-span-2">
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Number of model layers offloaded to the GPU. 999 = offload everything.')}>-ngl</span>
          <input type="number" bind:value={ngl} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Logical batch size: max tokens submitted to the model in one batch.')}>-b</span>
          <input type="number" bind:value={batchSize} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Physical (micro) batch size: tokens computed per forward pass. Must be ≤ -b.')}>-ub</span>
          <input type="number" bind:value={ubatchSize} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('CPU threads to use. Leave empty to auto-detect.')}>-t (threads, empty=auto)</span>
          <input bind:value={threads} disabled={isRunning} placeholder="auto" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('KV-cache data type for keys. f16 = full precision; q8_0/q4_0 quantize to save VRAM at some quality cost.')}>-ctk</span>
          <select bind:value={cacheTypeK} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono">
            <option>f16</option><option>q8_0</option><option>q4_0</option>
          </select>
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('KV-cache data type for values. f16 = full precision; q8_0/q4_0 quantize to save VRAM at some quality cost.')}>-ctv</span>
          <select bind:value={cacheTypeV} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono">
            <option>f16</option><option>q8_0</option><option>q4_0</option>
          </select>
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Context depth: pre-fill the KV cache with this many tokens before measuring, to test speed at longer contexts. 0 = disabled.')}>-d (ctx depth)</span>
          <input type="number" bind:value={nDepth} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="block">
          <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Repetitions per test. Results are averaged; more reps are more stable but slower.')}>-r (reps)</span>
          <input type="number" min="1" bind:value={repetitions} disabled={isRunning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono" />
        </label>
        <label class="flex items-center gap-2 mt-5 text-sm cursor-help" title={t('FlashAttention kernels: faster and lower-memory attention on supported GPUs.')}>
          <input type="checkbox" bind:checked={flashAttn} disabled={isRunning} />
          <span class="decoration-dotted underline underline-offset-2 decoration-slate-600">flash-attn</span>
        </label>
      </div>
      <label class="block lg:col-span-2">
        <span class="text-xs uppercase tracking-wider text-slate-400 cursor-help decoration-dotted underline underline-offset-2 decoration-slate-600" title={t('Any extra llama-bench flags passed verbatim. Server-only flags are rejected — see the warning below.')}>extra llama-bench flags</span>
        <input bind:value={extraFlags} disabled={isRunning} placeholder="--progress --numa distribute"
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono" />
        {#if flagWarnings.length > 0}
          <div class="mt-1 rounded border border-amber-700 bg-amber-900/20 px-2 py-1 text-xs text-amber-200">
            {t('⚠ llama-bench rejects these server-only flags:')}
            <code class="font-mono text-amber-100">{flagWarnings.join(' ')}</code>
            {t('— they belong to llama-server, not bench. Remove them before running.')}
          </div>
        {/if}
      </label>
    </div>

    <div class="flex items-center justify-between flex-wrap gap-3 pt-2 border-t border-slate-800">
      <div class="flex items-center gap-3 text-xs font-mono text-slate-500 flex-wrap">
        {#if vram}
          <span>free VRAM: <span class={vramHeadroomOK ? 'text-emerald-400' : 'text-amber-400'}>{(vram.free_mb / 1024).toFixed(1)} GB</span></span>
        {/if}
        {#if selectedModel}
          <span>· model: <span class="text-slate-300">{selectedModel.size_gb.toFixed(1)} GB</span></span>
        {/if}
        {#if runningWouldConflict.length > 0}
          <span class="text-amber-400">{t('· running: {names}', { names: runningWouldConflict.map(s => s.name).join(', ') })}</span>
        {/if}
      </div>
      <div class="flex items-center gap-2">
        {#if isRunning}
          <button
            onclick={stopBench}
            class="rounded bg-rose-800/40 border border-rose-700 px-4 py-1.5 text-sm text-rose-200 hover:bg-rose-800/60"
          >{t('Stop')}</button>
        {/if}
        <button
          onclick={runBench}
          disabled={busy || isRunning || !modelPath || flagWarnings.length > 0}
          title={flagWarnings.length > 0 ? t('remove server-only flags: {flags}', { flags: flagWarnings.join(' ') }) : ''}
          class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
        >{isRunning ? t('Running…') : t('Start benchmark')}</button>
      </div>
    </div>
  </section>

  <!-- Active job card -->
  {#if active && 'id' in active}
    <section class="rounded-lg border {active.status === 'running' ? 'border-emerald-800 bg-emerald-950/20' : active.status === 'failed' ? 'border-rose-800 bg-rose-950/20' : active.status === 'cancelled' ? 'border-amber-900/60 bg-amber-950/10' : 'border-slate-800 bg-slate-900/40'} p-4">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3">
          <span class="text-xs uppercase tracking-wider text-slate-500">bench #{active.id}</span>
          <span class="font-mono text-sm {active.status === 'running' ? 'text-emerald-400' : active.status === 'success' ? 'text-emerald-300' : active.status === 'failed' ? 'text-rose-400' : active.status === 'cancelled' ? 'text-amber-400' : 'text-slate-400'}">
            {active.status}
          </span>
          <span class="text-xs font-mono text-slate-400">· {active.model_name}</span>
          {#if active.build_number}
            <span class="text-xs font-mono text-slate-500">· build {active.build_number} {active.build_commit ?? ''}</span>
          {/if}
          {#if active.status === 'running'}
            <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
          {/if}
        </div>
        <div class="text-xs font-mono text-slate-500">{fmtDur(active.duration_seconds)}</div>
      </div>
      {#if active.results.length > 0}
        <div class="mt-3 overflow-x-auto">
          <table class="w-full text-xs font-mono">
            <thead class="text-slate-500 uppercase tracking-wider">
              <tr class="border-b border-slate-800">
                <th class="px-2 py-1 text-left">test</th>
                <th class="px-2 py-1 text-right">n_prompt</th>
                <th class="px-2 py-1 text-right">n_gen</th>
                <th class="px-2 py-1 text-right">n_depth</th>
                <th class="px-2 py-1 text-right">ngl</th>
                <th class="px-2 py-1 text-right">b/ub</th>
                <th class="px-2 py-1 text-right">t/s (avg ± stdev)</th>
              </tr>
            </thead>
            <tbody>
              {#each active.results as r}
                <tr class="border-b border-slate-800/50">
                  <td class="px-2 py-1 text-emerald-400">{testName(r)}</td>
                  <td class="px-2 py-1 text-right text-slate-400">{r.n_prompt ?? '—'}</td>
                  <td class="px-2 py-1 text-right text-slate-400">{r.n_gen ?? '—'}</td>
                  <td class="px-2 py-1 text-right text-slate-500">{r.n_depth ?? 0}</td>
                  <td class="px-2 py-1 text-right text-slate-500">{r.n_gpu_layers ?? '—'}</td>
                  <td class="px-2 py-1 text-right text-slate-500">{r.n_batch}/{r.n_ubatch}</td>
                  <td class="px-2 py-1 text-right text-slate-200">{fmtTps(r)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
      {#if active.status === 'failed' && active.error}
        <div class="mt-2 text-xs text-rose-400 font-mono">{active.error}</div>
      {/if}
    </section>
  {/if}

  <!-- Live log -->
  <section class="space-y-2">
    <div class="flex items-center justify-between">
      <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('live log')}</h2>
      <div class="flex items-center gap-3">
        <label class="flex items-center gap-1.5 text-xs text-slate-400">
          <input type="checkbox" bind:checked={autoScroll} />
          {t('auto-scroll')}
        </label>
        <button onclick={() => { log = []; }} class="rounded bg-slate-700/40 border border-slate-600 px-2 py-1 text-xs hover:bg-slate-700/60">{t('Clear')}</button>
      </div>
    </div>
    <div
      bind:this={logContainer}
      class="rounded-lg border border-slate-800 bg-black/60 p-3 font-mono text-[11px] leading-tight text-slate-300 overflow-y-auto whitespace-pre transition-[height] duration-200 {log.length > 0 || isRunning ? 'h-[calc(35*var(--vh))]' : 'h-14'}"
    >
      {#each log as line}
        <div class="{line.includes('[LlamaDeck]') ? 'text-amber-400' : line.match(/error|failed/i) ? 'text-rose-400' : line.startsWith('$') ? 'text-emerald-400' : ''}">{line}</div>
      {:else}
        <div class="text-slate-600 italic">{isRunning ? t('Waiting for output…') : t('Idle. Configure and click "Start benchmark".')}</div>
      {/each}
    </div>
  </section>

  <!-- History -->
  {#if historyRows.length > 0}
    <section>
      <h2 class="text-sm uppercase tracking-wider text-slate-400 mb-2">
        {t('History')} {modelPath ? t('· filter: {p}', { p: shortPath(modelPath) }) : ''}
      </h2>
      <div class="rounded-lg border border-slate-800 bg-slate-900/40 overflow-x-auto">
        <table class="w-full text-xs font-mono">
          <thead class="bg-slate-900/80 text-slate-500 uppercase tracking-wider">
            <tr>
              <th class="px-3 py-2 text-left">#</th>
              <th class="px-3 py-2 text-left">started</th>
              <th class="px-3 py-2 text-left">status</th>
              <th class="px-3 py-2 text-left">model</th>
              <th class="px-3 py-2 text-left">build</th>
              <th class="px-3 py-2 text-left">top t/s</th>
            </tr>
          </thead>
          <tbody>
            {#each historyRows as r}
              {@const pp = r.results?.find(x => (x.n_prompt ?? 0) > 0 && (x.n_gen ?? 0) === 0)}
              {@const tg = r.results?.find(x => (x.n_gen ?? 0) > 0 && (x.n_prompt ?? 0) === 0)}
              <tr class="border-t border-slate-800/60 hover:bg-slate-900/60">
                <td class="px-3 py-1.5 text-slate-500">{r.id}</td>
                <td class="px-3 py-1.5 text-slate-300">{fmtTs(r.started_at)}</td>
                <td class="px-3 py-1.5 {r.status === 'success' ? 'text-emerald-400' : r.status === 'failed' ? 'text-rose-400' : r.status === 'running' ? 'text-amber-400' : r.status === 'cancelled' ? 'text-amber-500/80' : 'text-slate-400'}">{r.status}</td>
                <td class="px-3 py-1.5 text-slate-400">{r.model_name ?? '—'}</td>
                <td class="px-3 py-1.5 text-slate-500">{r.build_number ?? '—'} {r.build_commit ?? ''}</td>
                <td class="px-3 py-1.5 text-slate-300">
                  {#if pp}pp{pp.n_prompt}: {(pp.avg_ts ?? 0).toFixed(0)}{/if}
                  {#if pp && tg} · {/if}
                  {#if tg}tg{tg.n_gen}: {(tg.avg_ts ?? 0).toFixed(0)}{/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/if}
</div>
