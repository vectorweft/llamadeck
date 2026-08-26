<script lang="ts">
  /**
   * Build page: update and rebuild llama.cpp from source, from the browser.
   *
   * The setup wizard builds llama.cpp once, to get a machine off the ground.
   * This is the page you come back to — what is installed now, what upstream
   * has landed since, which compute backend to build with, and the log of the
   * build actually running. Everything here is the BuildManager's own state
   * (/api/build/*): one build at a time, process-wide, so two open tabs and the
   * wizard all watch the same job rather than each starting their own.
   */
  import { onDestroy, onMount, tick } from 'svelte';
  import {
    api,
    type BuildBackends,
    type BuildCheck,
    type BuildJob,
    type BuildRecord,
    type LlamaVersion,
  } from '$lib/api';
  import { confirmDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';

  let version = $state<LlamaVersion | null>(null);
  let check = $state<BuildCheck | null>(null);
  let backends = $state<BuildBackends | null>(null);
  let active = $state<BuildJob | { status: 'idle' } | null>(null);
  let history = $state<BuildRecord[]>([]);
  let log = $state<string[]>([]);
  let error = $state<string | null>(null);
  let checkError = $state<string | null>(null);
  let checking = $state(false);
  let busy = $state(false);

  let backendId = $state('auto');
  let jobsOverride = $state('');
  let backendTouched = false;   // don't clobber the choice on a refresh

  let source: EventSource | null = null;
  let logContainer: HTMLDivElement | null = $state(null);
  let autoScroll = $state(true);
  let poll: ReturnType<typeof setInterval> | null = null;

  const isRunning = $derived(!!active && 'status' in active && active.status === 'running');
  const job = $derived(active && 'id' in active ? active : null);
  // `ahead` is how many upstream commits the checkout is behind. Null check
  // matters: a failed fetch must not render as "you are up to date".
  const behindBy = $derived(check?.ahead ?? null);

  async function refresh() {
    try {
      const [v, b, a, h] = await Promise.all([
        api.buildVersion(),
        api.buildBackends(),
        api.buildActive(),
        api.buildHistory(15),
      ]);
      version = v;
      backends = b;
      active = a;
      history = h;
      if (!backendTouched && b.current) backendId = b.current;
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  /** Upstream check is a network call and the slowest thing here, so it runs on
   * its own and reports its own failure — an offline box still gets the rest of
   * the page instead of one long spinner. */
  async function refreshCheck() {
    checking = true;
    try {
      check = await api.buildCheck();
      checkError = null;
    } catch (e) {
      checkError = e instanceof Error ? e.message : String(e);
    } finally {
      checking = false;
    }
  }

  function openStream() {
    source?.close();
    // Subscribing primes the queue with the running job's scrollback, so the
    // stream alone is the whole picture — anything already held would double up.
    log = [];
    source = new EventSource('/api/build/stream');
    source.onmessage = async (ev) => {
      try {
        const data = JSON.parse(ev.data);
        log = [...log, data.line];
        if (log.length > 5000) log = log.slice(-5000);
        if (autoScroll) {
          await tick();
          logContainer?.scrollTo({ top: logContainer.scrollHeight });
        }
      } catch { /* ignore malformed frame */ }
    };
  }

  async function rebuild() {
    const label = backends?.backends.find((b) => b.id === backendId)?.label ?? backendId;
    const switching = !!backends?.current && backends.current !== backendId && backendId !== 'auto';
    const ok = await confirmDialog(
      switching
        ? t('Building with {backend} replaces the {current} build. The stale cmake cache is wiped, so this run compiles everything from scratch.',
            { backend: label, current: backends?.current ?? '' })
        : t('This pulls llama.cpp and rebuilds it with {backend}. Running models keep serving from the binary already loaded in memory.',
            { backend: label }),
      { title: t('Rebuild llama.cpp?'), confirmLabel: t('Rebuild') }
    );
    if (!ok) return;
    busy = true;
    error = null;
    try {
      const n = jobsOverride.trim() === '' ? null : Number(jobsOverride);
      await api.buildRebuild(backendId, Number.isFinite(n as number) ? (n as number) : null);
      toast(t('Build started'), 'success');
      openStream();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    refresh();
    refreshCheck();
    openStream();
    poll = setInterval(refresh, 3000);
  });
  onDestroy(() => {
    source?.close();
    if (poll) clearInterval(poll);
  });

  function fmtTs(ts: number | null): string {
    if (!ts) return '—';
    return new Date(ts * 1000).toLocaleString();
  }
  function fmtDur(s: number): string {
    if (s < 60) return `${s.toFixed(0)}s`;
    return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  }
  function recordDur(r: BuildRecord): string {
    if (!r.finished_at) return '—';
    return fmtDur(r.finished_at - r.started_at);
  }
  function statusClass(s: string): string {
    if (s === 'running') return 'text-emerald-400';
    if (s === 'success') return 'text-emerald-300';
    if (s === 'failed') return 'text-rose-400';
    return 'text-slate-400';
  }
</script>

<div class="max-w-6xl space-y-6">
  <div class="flex items-center gap-3 flex-wrap">
    <h1 class="text-2xl font-semibold">{t('Build')}</h1>
    <span class="text-xs text-slate-500 font-mono">
      {t('llama.cpp from source · git pull → cmake → ninja')}
    </span>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  <!-- What is installed, and what upstream has since -->
  <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-3">
    <div class="flex items-baseline gap-3 flex-wrap">
      <span class="text-xs uppercase tracking-wider text-slate-400">{t('installed')}</span>
      {#if version}
        <span class="font-mono text-sm text-slate-200">
          {version.build_number != null ? `build ${version.build_number}` : t('unknown build')}
          {#if version.commit}<span class="text-slate-500"> · {version.commit}</span>{/if}
        </span>
      {:else}
        <span class="font-mono text-sm text-slate-500">…</span>
      {/if}
      <!-- The check runs once on mount and can fail on its own (offline box,
           slow fetch). Without a way to run it again the only retry was a page
           reload, which also throws away the log stream. -->
      <button
        onclick={refreshCheck}
        disabled={checking}
        class="ml-auto rounded border border-slate-700 px-2 py-1 text-xs hover:bg-slate-800 disabled:opacity-50"
      >
        {checking ? t('checking upstream…') : t('Check for updates')}
      </button>
    </div>

    {#if checkError}
      <!-- Deliberately unlabelled as to cause: this fails for a missing
           checkout as often as for a missing network, and the backend's own
           message already says which and what to do about it. Guessing here
           would just contradict it. -->
      <div class="rounded border border-amber-800 bg-amber-950/20 px-3 py-2">
        <span class="text-xs uppercase tracking-wider text-amber-400">{t('update check unavailable')}</span>
        <p class="mt-1 text-xs font-mono text-amber-200">{checkError}</p>
      </div>
    {:else if check}
      <div class="text-sm">
        {#if behindBy === 0}
          <span class="text-emerald-300 font-mono">{t('up to date with {branch}', { branch: check.branch })}</span>
        {:else}
          <span class="text-amber-300 font-mono">
            {t('{n} new commit(s) on {branch}', { n: behindBy ?? 0, branch: check.branch })}
          </span>
          {#if check.commits.length > 0}
            <ul class="mt-2 space-y-0.5 max-h-40 overflow-y-auto">
              {#each check.commits as c (c.sha)}
                <li class="font-mono text-xs text-slate-400">
                  <span class="text-slate-600">{c.sha.slice(0, 8)}</span> {c.subject}
                </li>
              {/each}
            </ul>
          {/if}
        {/if}
      </div>
    {:else}
      <p class="text-xs font-mono text-slate-500">{t('checking upstream…')}</p>
    {/if}
  </section>

  <!-- Rebuild controls -->
  <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <label class="block">
        <span class="text-xs uppercase tracking-wider text-slate-400">{t('compute backend')}</span>
        <select
          bind:value={backendId}
          onchange={() => (backendTouched = true)}
          disabled={isRunning}
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono"
        >
          <option value="auto">
            {t('auto')}{backends ? ` — ${backends.backends.find((b) => b.id === backends!.preferred)?.label ?? backends.preferred}` : ''}
          </option>
          <!-- Only backends this machine can actually build. Offering CUDA on a
               box with no toolkit turns a clear "not available" into a cmake
               error twenty minutes in. -->
          {#each (backends?.backends ?? []).filter((b) => b.supported && b.available) as b (b.id)}
            <option value={b.id}>{b.label}</option>
          {/each}
        </select>
        {#if backends}
          {@const sel = backends.backends.find((b) => b.id === backendId)}
          {#if sel?.detail}
            <span class="mt-1 block text-xs font-mono text-slate-500">{sel.detail}</span>
          {/if}
        {/if}
      </label>

      <label class="block">
        <span class="text-xs uppercase tracking-wider text-slate-400">{t('parallel jobs')}</span>
        <input
          bind:value={jobsOverride}
          disabled={isRunning}
          placeholder={t('all cores')}
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono"
        />
      </label>
    </div>

    <div class="flex items-center justify-between flex-wrap gap-3 pt-2 border-t border-slate-800">
      <div class="text-xs font-mono text-slate-500">
        {#if backends?.current}
          {t('current build: {backend}', { backend: backends.current })}
        {/if}
      </div>
      <button
        onclick={rebuild}
        disabled={busy || isRunning}
        class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
      >{isRunning ? t('Building…') : t('Rebuild')}</button>
    </div>
  </section>

  <!-- Active job + log -->
  {#if job}
    <section class="rounded-lg border {job.status === 'running' ? 'border-emerald-800 bg-emerald-950/20' : job.status === 'failed' ? 'border-rose-800 bg-rose-950/20' : 'border-slate-800 bg-slate-900/40'} p-4 space-y-3">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3 flex-wrap">
          <span class="text-xs uppercase tracking-wider text-slate-500">{t('build')} #{job.id}</span>
          <span class="font-mono text-sm {statusClass(job.status)}">{job.status}</span>
          <span class="text-xs font-mono text-slate-400">· {job.current_step}</span>
          {#if job.status === 'running'}
            <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
          {/if}
          {#if job.from_commit && job.to_commit}
            <span class="text-xs font-mono text-slate-500">· {job.from_commit} → {job.to_commit}</span>
          {/if}
        </div>
        <div class="flex items-center gap-3">
          <label class="flex items-center gap-1.5 text-xs text-slate-500">
            <input type="checkbox" bind:checked={autoScroll} class="accent-emerald-600" />
            {t('follow')}
          </label>
          <span class="text-xs font-mono text-slate-500">{fmtDur(job.duration_seconds)}</span>
        </div>
      </div>

      <div
        bind:this={logContainer}
        class="h-80 overflow-y-auto rounded bg-slate-950 border border-slate-800 p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap"
      >{#each log as line, i (i)}{line}{'\n'}{:else}<span class="text-slate-600">{t('no output yet')}</span>{/each}</div>
    </section>
  {/if}

  <!-- History -->
  <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5">
    <h2 class="text-xs uppercase tracking-wider text-slate-400 mb-3">{t('build history')}</h2>
    {#if history.length === 0}
      <p class="text-sm text-slate-500">{t('No builds yet.')}</p>
    {:else}
      <div class="overflow-x-auto">
        <table class="w-full text-xs font-mono">
          <thead class="text-slate-500 uppercase tracking-wider">
            <tr class="border-b border-slate-800">
              <th class="px-2 py-1 text-left">#</th>
              <th class="px-2 py-1 text-left">{t('started')}</th>
              <th class="px-2 py-1 text-left">{t('status')}</th>
              <th class="px-2 py-1 text-left">{t('took')}</th>
              <th class="px-2 py-1 text-left">{t('commits')}</th>
            </tr>
          </thead>
          <tbody>
            {#each history as r (r.id)}
              <tr class="border-b border-slate-900">
                <td class="px-2 py-1 text-slate-500">{r.id}</td>
                <td class="px-2 py-1 text-slate-400">{fmtTs(r.started_at)}</td>
                <td class="px-2 py-1 {statusClass(r.status)}">{r.status}</td>
                <td class="px-2 py-1 text-slate-400">{recordDur(r)}</td>
                <td class="px-2 py-1 text-slate-500">
                  {r.from_commit && r.to_commit ? `${r.from_commit} → ${r.to_commit}` : '—'}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
</div>
