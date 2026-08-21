<script lang="ts">
  import { goto } from '$app/navigation';
  import { api, type HFSearchResult, type HFFile, type DownloadJob } from '$lib/api';
  import { alertDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';
  import Skeleton from '$lib/components/Skeleton.svelte';

  // --- state ---
  let query = $state('');
  let searching = $state(false);
  let results = $state<HFSearchResult[]>([]);
  let searchError = $state<string | null>(null);

  // Expanded repo for file listing
  let expanded = $state<string | null>(null);
  let expandedFiles = $state<HFFile[]>([]);
  let expandedLoading = $state(false);

  // Inline brand/series/base_model editing (target path: models/<brand>/<series>/<base_model>/)
  let editBrand = $state('');
  let editSeries = $state('');
  let editBaseModel = $state('');

  // Jobs
  let jobs = $state<DownloadJob[]>([]);
  let jobsTimer: ReturnType<typeof setInterval> | null = null;

  // Active SSE streams: job_id → EventSource
  const streams = new Map<string, EventSource>();

  import { onDestroy, onMount } from 'svelte';

  onMount(async () => {
    await refreshJobs();
    jobsTimer = setInterval(refreshJobs, 5000);
  });

  onDestroy(() => {
    if (jobsTimer) clearInterval(jobsTimer);
    for (const es of streams.values()) es.close();
  });

  // Terminal states (no SSE needed)
  function isSettled(s: DownloadJob['status']): boolean {
    return s === 'done' || s === 'failed' || s === 'paused';
  }

  async function refreshJobs() {
    try {
      const r = await api.hfJobs();
      jobs = r.jobs;
      // open SSE for in-progress jobs
      for (const j of jobs) {
        if ((j.status === 'in_progress' || j.status === 'queued') && !streams.has(j.job_id)) {
          openJobStream(j.job_id);
        }
        if (isSettled(j.status) && streams.has(j.job_id)) {
          streams.get(j.job_id)!.close();
          streams.delete(j.job_id);
        }
      }
    } catch { /* ignore */ }
  }

  function openJobStream(job_id: string) {
    const es = new EventSource(`/api/hf/stream/${encodeURIComponent(job_id)}`);
    es.onmessage = (ev) => {
      try {
        const updated: DownloadJob = JSON.parse(ev.data);
        jobs = jobs.map(j => j.job_id === updated.job_id ? updated : j);
        if (isSettled(updated.status)) {
          es.close();
          streams.delete(job_id);
        }
      } catch { /* ignore */ }
    };
    streams.set(job_id, es);
  }

  async function pauseJob(j: DownloadJob) {
    try {
      await api.hfPause(j.job_id);
      toast(t('Paused: {file}', { file: j.filename }), 'info');
      await refreshJobs();
    } catch (e) {
      await alertDialog(t('Could not pause: {e}', { e: e instanceof Error ? e.message : String(e) }), { title: t('Error') });
    }
  }

  async function resumeJob(j: DownloadJob) {
    try {
      const job = await api.hfResume(j.job_id);
      toast(t('Resumed: {file}', { file: j.filename }), 'success');
      jobs = jobs.map(x => x.job_id === job.job_id ? job : x);
      if (!streams.has(job.job_id)) openJobStream(job.job_id);
    } catch (e) {
      await alertDialog(t('Could not resume: {e}', { e: e instanceof Error ? e.message : String(e) }), { title: t('Error') });
    }
  }

  async function removeJob(j: DownloadJob) {
    try {
      await api.hfJobDelete(j.job_id);
      toast(t('Removed from list'), 'info');
      jobs = jobs.filter(x => x.job_id !== j.job_id);
    } catch (e) {
      await alertDialog(t('Could not remove: {e}', { e: e instanceof Error ? e.message : String(e) }), { title: t('Error') });
    }
  }

  async function doSearch() {
    if (!query.trim()) return;
    searching = true;
    searchError = null;
    results = [];
    try {
      const r = await api.hfSearch(query.trim(), 20);
      results = r.results;
    } catch (e) {
      searchError = e instanceof Error ? e.message : String(e);
    } finally {
      searching = false;
    }
  }

  async function toggleExpand(m: HFSearchResult) {
    if (expanded === m.repo_id) {
      expanded = null;
      return;
    }
    expanded = m.repo_id;
    editBrand = m.brand;
    editSeries = m.series;
    editBaseModel = '';
    expandedLoading = true;
    expandedFiles = [];
    try {
      const r = await api.hfFiles(m.repo_id);
      expandedFiles = r.files;
      // update brand/series from fresh classify
      editBrand = r.brand;
      editSeries = r.series;
      // Pick a representative file (first non-mmproj .gguf) to derive base_model
      const rep = r.files.find(f => !f.name.toLowerCase().startsWith('mmproj')) ?? r.files[0];
      if (rep) {
        try {
          const c = await api.hfClassify(m.repo_id, rep.name);
          editBaseModel = c.base_model;
        } catch { /* fallback empty */ }
      }
    } catch { /* ignore */ } finally {
      expandedLoading = false;
    }
  }

  async function download(repo_id: string, filename: string) {
    try {
      const job = await api.hfDownload(
        repo_id, filename,
        editBrand || undefined, editSeries || undefined,
        editBaseModel || undefined,
      );
      jobs = [job, ...jobs.filter(j => j.job_id !== job.job_id)];
      if (!streams.has(job.job_id)) openJobStream(job.job_id);
      toast(t('Download started: {file}', { file: filename }), 'success');
      // The Downloads section sits right under the search — scroll there on start
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e) {
      await alertDialog(t('Could not start download: {e}', { e: e instanceof Error ? e.message : String(e) }), { title: t('Error') });
    }
  }

  function fmtBytes(n: number | null | undefined): string {
    if (n == null) return '—';
    if (n < 1e6) return (n / 1e3).toFixed(0) + ' KB';
    if (n < 1e9) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e9).toFixed(1) + ' GB';
  }

  function fmtSpeed(bps: number): string {
    if (bps < 1e3) return bps.toFixed(0) + ' B/s';
    if (bps < 1e6) return (bps / 1e3).toFixed(0) + ' KB/s';
    return (bps / 1e6).toFixed(1) + ' MB/s';
  }

  // Hours matter here: a 96 GB GGUF on a home line is a multi-hour download,
  // and "139m 52s" makes the reader do the division. Rolls up to h/m/s.
  function fmtEta(s: number | null): string {
    if (s == null) return '';
    const total = Math.ceil(s);
    if (total < 60) return `${total}s`;
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    return `${m}m ${sec}s`;
  }

  const quantOrder = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'F16', 'F32', 'MXFP', 'UD', 'IQ'];

  function groupByQuant(files: HFFile[]): Map<string, HFFile[]> {
    const groups = new Map<string, HFFile[]>();
    for (const f of files) {
      const key = inferQuant(f.name);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(f);
    }
    // sort by quantOrder
    const sorted = new Map<string, HFFile[]>();
    for (const k of [...groups.keys()].sort((a, b) => {
      const ai = quantOrder.findIndex(q => a.toUpperCase().startsWith(q));
      const bi = quantOrder.findIndex(q => b.toUpperCase().startsWith(q));
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    })) {
      sorted.set(k, groups.get(k)!);
    }
    return sorted;
  }

  function inferQuant(name: string): string {
    const m = name.match(/[-._]((?:UD-?)?(?:IQ|Q|F)\d+(?:[._]K[._][A-Z]+)?(?:[-_]XL)?)/i);
    if (m) return m[1].toUpperCase();
    if (/MXFP4/i.test(name)) return 'MXFP4';
    if (/fp16/i.test(name)) return 'F16';
    if (/f32/i.test(name)) return 'F32';
    return 'other';
  }

  const statusLabels: Record<DownloadJob['status'], string> = {
    queued: 'queued',
    in_progress: 'downloading',
    paused: 'paused',
    done: 'done',
    failed: 'failed',
  };

  function statusChip(status: DownloadJob['status']): string {
    if (status === 'done') return 'border-emerald-800 bg-emerald-950/50 text-emerald-400';
    if (status === 'failed') return 'border-rose-800 bg-rose-950/50 text-rose-400';
    if (status === 'in_progress') return 'border-cyan-800 bg-cyan-950/50 text-cyan-300';
    if (status === 'paused') return 'border-amber-800 bg-amber-950/50 text-amber-300';
    return 'border-slate-700 bg-slate-800/60 text-slate-400';
  }

  function fmtClock(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
  }

  // Active (queued / downloading) jobs — used to lock file buttons
  const activeKeys = $derived(new Set(
    jobs
      .filter(j => j.status === 'queued' || j.status === 'in_progress')
      .map(j => `${j.repo_id}::${j.filename}`)
  ));
  // Paused jobs — used to show "resumable" on the file button
  const pausedKeys = $derived(new Set(
    jobs
      .filter(j => j.status === 'paused')
      .map(j => `${j.repo_id}::${j.filename}`)
  ));
  const activeCount = $derived(jobs.filter(j => j.status === 'in_progress' || j.status === 'queued').length);
</script>

<div class="max-w-5xl space-y-6">
  <h1 class="text-2xl font-semibold">HuggingFace Download</h1>

  <!-- Search -->
  <form class="flex gap-2" onsubmit={(e) => { e.preventDefault(); doSearch(); }}>
    <input
      bind:value={query}
      placeholder={t('Search GGUF models… (e.g. Qwen2.5-7B, Llama-3)')}
      class="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-slate-500"
    />
    <button
      type="submit"
      disabled={searching}
      class="rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 px-4 py-2 text-sm font-mono text-emerald-100"
    >
      {searching ? t('Searching…') : t('Search')}
    </button>
  </form>

  {#if searchError}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-2 text-sm text-rose-200 font-mono">{searchError}</div>
  {/if}

  <!-- Active & recent jobs (right under the search, visible once a download starts) -->
  {#if jobs.length > 0}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-xs uppercase tracking-wider text-slate-500">{t('Downloads')}</h3>
        {#if activeCount > 0}
          <span class="text-[11px] font-mono text-cyan-400">
            <span class="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse align-middle mr-1"></span>{t('{n} active', { n: activeCount })}
          </span>
        {/if}
      </div>
      {#each jobs as j (j.job_id)}
        <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-1.5">
          <!-- top row: file name + status chip + actions -->
          <div class="flex items-center justify-between flex-wrap gap-2">
            <div class="flex items-center gap-2 min-w-0">
              <span class="font-mono text-xs text-slate-200 truncate" title={j.filename}>{j.filename.split('/').pop()}</span>
              <span class="rounded-full border px-2 py-0.5 text-[10px] font-mono shrink-0 {statusChip(j.status)}">{t(statusLabels[j.status])}</span>
            </div>
            <div class="flex items-center gap-1.5 shrink-0">
              {#if j.status === 'in_progress' || j.status === 'queued'}
                <button
                  onclick={() => pauseJob(j)}
                  class="rounded border border-amber-900 bg-amber-950/40 hover:bg-amber-900/50 px-2 py-0.5 text-[11px] font-mono text-amber-300"
                >⏸ {t('pause')}</button>
              {:else if j.status === 'paused' || j.status === 'failed'}
                <button
                  onclick={() => resumeJob(j)}
                  class="rounded border border-emerald-900 bg-emerald-950/40 hover:bg-emerald-900/50 px-2 py-0.5 text-[11px] font-mono text-emerald-300"
                >{j.status === 'paused' ? '▶ ' + t('resume') : '↻ ' + t('retry')}</button>
              {/if}
              {#if j.status === 'done' && j.target_path && !/^mmproj/i.test(j.filename.split('/').pop() ?? '')}
                <button
                  onclick={() => goto('/presets?new_from=' + encodeURIComponent(j.target_path))}
                  title={t('Open the preset editor with this model preselected')}
                  class="rounded border border-emerald-800 bg-emerald-950/50 hover:bg-emerald-900/60 px-2 py-0.5 text-[11px] font-mono text-emerald-300"
                >＋ {t('create preset')}</button>
              {/if}
              {#if isSettled(j.status)}
                <button
                  onclick={() => removeJob(j)}
                  title={t('Remove from list (does not touch the file)')}
                  class="rounded border border-slate-700 bg-slate-800/60 hover:bg-slate-700/70 px-2 py-0.5 text-[11px] font-mono text-slate-400"
                >✕ {t('remove')}</button>
              {/if}
            </div>
          </div>

          <!-- source → destination -->
          <div class="text-[11px] text-slate-600 font-mono truncate" title="{j.repo_id} → {j.target_dir}">
            {j.repo_id} → {j.brand}/{j.series}/{j.base_model} · {t('started {time}', { time: fmtClock(j.created_at) })}
          </div>

          {#if j.status !== 'failed' || j.bytes_downloaded > 0}
            <div class="flex items-center gap-3">
              <div class="h-2 flex-1 rounded-full bg-slate-800 overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-500 {j.status === 'done' ? 'bg-emerald-500' : j.status === 'paused' ? 'bg-amber-500' : j.status === 'failed' ? 'bg-rose-600' : 'bg-cyan-500'}"
                  style="width: {j.pct}%"
                ></div>
              </div>
              <span class="text-xs font-mono {j.status === 'done' ? 'text-emerald-400' : 'text-slate-300'} shrink-0 w-14 text-right">{j.pct.toFixed(1)}%</span>
            </div>
            <div class="flex justify-between flex-wrap gap-x-4 text-[11px] text-slate-500 font-mono">
              <span>{fmtBytes(j.bytes_downloaded)} / {fmtBytes(j.total_bytes)}</span>
              {#if j.status === 'in_progress'}
                <span class="text-cyan-500">{fmtSpeed(j.speed_bps)}{#if j.eta_seconds}&nbsp;· {t('~{eta} left', { eta: fmtEta(j.eta_seconds) })}{/if}</span>
              {:else if j.status === 'paused'}
                <span class="text-amber-500/80">{t('resumable from where it left off')}</span>
              {/if}
            </div>
          {/if}

          {#if j.status === 'failed' && j.error}
            <div class="text-xs text-rose-400 font-mono break-all">{j.error}</div>
          {/if}

          {#if j.status === 'done' && j.target_path}
            <div class="text-[11px] text-emerald-700 font-mono truncate" title={j.target_path}>{j.target_path}</div>
          {/if}
        </div>
      {/each}
    </section>
  {/if}

  <!-- Search results -->
  {#if results.length > 0}
    <div class="space-y-2">
      <div class="text-xs text-slate-500 font-mono">{t('{n} results', { n: results.length })}</div>
      {#each results as m}
        <div class="rounded-lg border border-slate-800 bg-slate-900/40">
          <!-- Header row -->
          <button
            onclick={() => toggleExpand(m)}
            class="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-800/40 rounded-lg"
          >
            <div class="flex items-center gap-3 min-w-0">
              <span class="font-mono text-sm text-slate-200 truncate">{m.repo_id}</span>
              <span class="text-xs text-slate-500 font-mono shrink-0">⭐ {m.likes.toLocaleString()}</span>
              <span class="text-xs text-slate-500 font-mono shrink-0" title={t('total downloads')}>⬇ {m.downloads.toLocaleString()}</span>
              <span class="text-xs text-emerald-700 font-mono shrink-0">{m.brand} / {m.series}</span>
            </div>
            <span class="text-slate-600 text-xs font-mono ml-2">{expanded === m.repo_id ? '▲' : '▼'} {m.files.length} dosya</span>
          </button>

          <!-- Expanded file list -->
          {#if expanded === m.repo_id}
            <div class="border-t border-slate-800 px-4 pb-4 pt-3 space-y-3">
              <!-- Brand / series / base-model override (3-level layout) -->
              <div class="flex items-center gap-2 flex-wrap">
                <span class="text-xs text-slate-500 font-mono">{t('save to:')}</span>
                <input
                  bind:value={editBrand}
                  placeholder="brand"
                  class="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-mono text-slate-200 w-24 focus:outline-none focus:border-slate-500"
                />
                <span class="text-slate-600">/</span>
                <input
                  bind:value={editSeries}
                  placeholder="series"
                  class="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-mono text-slate-200 w-32 focus:outline-none focus:border-slate-500"
                />
                <span class="text-slate-600">/</span>
                <input
                  bind:value={editBaseModel}
                  placeholder="base-model"
                  class="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-mono text-slate-200 w-44 focus:outline-none focus:border-slate-500"
                  title={t('The base-model folder groups quants sharing the same architecture (and mmproj). Different base models get separate folders.')}
                />
                <span class="text-xs text-slate-600 font-mono">→ models/{editBrand || '…'}/{editSeries || '…'}/{editBaseModel || '…'}/</span>
              </div>

              {#if expandedLoading}
                <div class="text-xs text-slate-500 font-mono"><Skeleton class="h-4 w-full mb-1.5" /><Skeleton class="h-4 w-5/6 mb-1.5" /><Skeleton class="h-4 w-2/3" /></div>
              {:else if expandedFiles.length === 0}
                <div class="text-xs text-slate-500 font-mono">{t('No .gguf files found.')}</div>
              {:else}
                {@const groups = groupByQuant(expandedFiles)}
                <div class="space-y-2">
                  {#each [...groups.entries()] as [quant, files]}
                    <div>
                      <div class="text-[11px] uppercase tracking-wider text-slate-600 font-mono mb-1">{quant}</div>
                      <div class="flex flex-wrap gap-1.5">
                        {#each files as f}
                          {@const active = activeKeys.has(`${m.repo_id}::${f.name}`)}
                          {@const paused = pausedKeys.has(`${m.repo_id}::${f.name}`)}
                          <button
                            onclick={() => download(m.repo_id, f.name)}
                            disabled={active}
                            class="flex items-center gap-1.5 rounded border px-2.5 py-1.5 text-xs font-mono {active
                              ? 'border-cyan-800 bg-cyan-950/40 text-cyan-400 cursor-default'
                              : paused
                                ? 'border-amber-800 bg-amber-950/30 hover:bg-amber-900/40 text-amber-300'
                                : 'border-slate-700 bg-slate-800 hover:bg-slate-700 hover:border-slate-600 text-slate-300'}"
                            title={active ? t('Downloading…') : paused ? t('Incomplete — click to resume from where it left off') : f.name}
                          >
                            <span class="truncate max-w-[200px]">{f.name.split('/').pop()}</span>
                            {#if active}
                              <span class="text-cyan-500 shrink-0 animate-pulse">{t('downloading…')}</span>
                            {:else if paused}
                              <span class="text-amber-500 shrink-0">▶ {t('resume')}</span>
                            {:else}
                              <span class="text-slate-500 shrink-0">{fmtBytes(f.size)}</span>
                            {/if}
                          </button>
                        {/each}
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

</div>
