<script lang="ts">
  /**
   * First-run wizard: takes a machine with nothing on it to a running model.
   *
   * The backend decides which step the user is on (`SetupState.step`) — this
   * page renders that verdict rather than deriving its own, so it can never
   * claim "done" while the app is still unusable. Nothing is persisted as
   * "wizard completed": every step re-checks the real thing (does the binary
   * answer --version, are there GGUFs on disk, is a preset defined), so
   * deleting your models later brings the step back instead of leaving the
   * wizard lying about the state of the machine.
   */
  import { onDestroy, onMount, tick } from 'svelte';
  import { api, type SetupState, type SetupStep } from '$lib/api';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';

  let st = $state<SetupState | null>(null);
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);

  // --- step 1 inputs -------------------------------------------------------
  let repoPath = $state('');
  let backendId = $state('auto');
  let jobsOverride = $state('');
  let manualBin = $state('');
  let showManual = $state(false);
  let repoTouched = false;   // don't clobber what the user is typing on poll

  // --- step 2 inputs -------------------------------------------------------
  let modelsRoot = $state('');
  let modelsRootTouched = false;

  // --- build log -----------------------------------------------------------
  let log = $state<string[]>([]);
  let source: EventSource | null = null;
  let logContainer: HTMLDivElement | null = $state(null);
  let poll: ReturnType<typeof setInterval> | null = null;

  // `required` marks the two steps without which nothing can run. The last two
  // are suggestions — someone who already has GGUFs on an external disk copies
  // them in and never touches the download page, and the wizard must not treat
  // that as an unfinished install.
  const STEPS: { id: SetupStep; title: string; required: boolean }[] = [
    { id: 'llama', title: 'Install llama.cpp', required: true },
    { id: 'models_dir', title: 'Choose a models folder', required: true },
    { id: 'model', title: 'Add a model', required: false },
    { id: 'preset', title: 'Create a preset', required: false },
  ];

  const stepIndex = $derived(st ? STEPS.findIndex((s) => s.id === st!.step) : -1);
  const done = $derived(st?.step === 'done');
  const ready = $derived(st?.required_done === true);
  const building = $derived(st?.build_active?.status === 'running');
  const buildFailed = $derived(st?.build_active?.status === 'failed');
  const missingTools = $derived.by(() => {
    if (!st) return [] as string[];
    const tc = st.toolchain;
    return [
      ...(tc.git ? [] : ['git']),
      ...(tc.cmake ? [] : ['cmake']),
      ...(tc.compiler ? [] : ['build-essential']),
    ];
  });
  const buildableBackends = $derived((st?.backends ?? []).filter((b) => b.supported));
  const autoLabel = $derived(
    st?.backends.find((b) => b.id === st?.preferred_backend)?.label ?? '',
  );

  /** A step is complete when the wizard has moved past it. */
  function stateOf(i: number): 'done' | 'current' | 'todo' {
    if (done) return 'done';
    if (stepIndex < 0) return 'todo';
    return i < stepIndex ? 'done' : i === stepIndex ? 'current' : 'todo';
  }

  async function refresh() {
    try {
      const next = await api.setupState();
      st = next;
      if (!repoTouched) repoPath = next.llama.repo_path || next.llama.default_repo_path;
      if (!modelsRootTouched) modelsRoot = next.models.default_root;
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function openStream() {
    source?.close();
    // The stream primes every new subscriber with the running job's scrollback,
    // so a reconnect re-delivers lines this page already has. Reopening on
    // "Clone and build" therefore printed the first few lines of the job twice.
    // The stream is self-sufficient: drop what we hold and let it re-send.
    log = [];
    source = new EventSource('/api/build/stream');
    source.onmessage = async (ev) => {
      try {
        const data = JSON.parse(ev.data);
        log = [...log, data.line];
        if (log.length > 3000) log = log.slice(-3000);
        await tick();
        logContainer?.scrollTo({ top: logContainer.scrollHeight });
      } catch { /* ignore malformed frame */ }
    };
  }

  async function useBinary(path: string) {
    busy = path;
    error = null;
    try {
      const r = await api.setupUseBinary(path);
      toast(t('Using {path}', { path: r.llama_bin }), 'success');
      showManual = false;
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function installFromSource() {
    busy = 'build';
    error = null;
    try {
      const n = jobsOverride.trim() === '' ? null : Number(jobsOverride);
      const r = await api.setupBuild(repoPath, backendId, Number.isFinite(n as number) ? (n as number) : null);
      toast(r.cloning ? t('Cloning llama.cpp…') : t('Build started'), 'success');
      openStream();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function rescan() {
    busy = 'rescan';
    error = null;
    try {
      const r = await api.setupRescan();
      toast(r.count > 0 ? t('{n} models found', { n: r.count }) : t('No GGUF files found there yet'),
            r.count > 0 ? 'success' : 'info');
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function saveModelsRoot() {
    busy = 'models';
    error = null;
    try {
      // create=true: on a fresh machine the folder usually does not exist yet,
      // and bouncing the user out to a terminal for `mkdir` is the kind of
      // dead end this wizard exists to remove.
      const r = await api.setupModelsRoot(modelsRoot, true);
      toast(t('Models folder set: {path}', { path: r.hf_models_root }), 'success');
      modelsRootTouched = false;
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  onMount(() => {
    refresh();
    openStream();
    // 6 s: /state runs `llama-server --version` in a subprocess, so this is not
    // free. Fast enough that a finished build flips the step on its own.
    poll = setInterval(refresh, 6000);
  });
  onDestroy(() => {
    source?.close();
    if (poll) clearInterval(poll);
  });
</script>

<div class="max-w-3xl space-y-6">
  <div class="flex items-baseline gap-3">
    <h1 class="text-2xl font-semibold">{t('Setup')}</h1>
    <span class="text-xs text-slate-500 font-mono">
      {ready ? t('everything is ready') : t('step {n} of {m}', { n: Math.max(stepIndex + 1, 1), m: STEPS.length })}
    </span>
    <a href="/" class="ml-auto text-sm text-slate-400 underline decoration-dotted hover:text-slate-200">
      {t('Back to Dashboard')}
    </a>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-2 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  {#if !st}
    <div class="text-sm text-slate-500">{t('Checking what this machine already has…')}</div>
  {:else}
    {#if ready}
      <section class="rounded-lg border border-emerald-900/60 bg-emerald-950/20 p-5">
        <h2 class="text-lg font-semibold text-slate-100">{t('You are all set')}</h2>
        <p class="mt-1 text-sm text-slate-400">
          {done
            ? t('llama.cpp is installed, models are indexed and you have a preset. Start it from the Server page.')
            : t('llama.cpp is installed and your models folder is set — LlamaDeck is ready to use. The steps below are suggestions, not requirements.')}
        </p>
        <div class="mt-3 flex gap-3">
          <a href="/server" class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-emerald-700/60">{t('Go to Server')}</a>
          <a href="/" class="rounded border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700/70">{t('Dashboard')}</a>
        </div>
      </section>
    {/if}

    <!-- Machine summary: what the wizard is planning against. -->
    <div class="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3 text-xs font-mono text-slate-400 flex flex-wrap gap-x-4 gap-y-1">
      <span>{st.platform.cpu_name}</span>
      <span>{st.platform.os}/{st.platform.arch}</span>
      {#if autoLabel}<span class="text-cyan-500/90">{t('best backend: {b}', { b: autoLabel })}</span>{/if}
    </div>

    <ol class="space-y-3">
      {#each STEPS as s, i (s.id)}
        {@const state = stateOf(i)}
        <li class="rounded-lg border {state === 'current' ? 'border-cyan-900/60 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/30'} p-5">
          <div class="flex items-start gap-3">
            <span
              class="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs font-mono
                     {state === 'done' ? 'border-emerald-600 bg-emerald-900/40 text-emerald-300'
                      : state === 'current' ? 'border-cyan-600 bg-cyan-900/40 text-cyan-200'
                      : 'border-slate-700 text-slate-600'}"
            >{state === 'done' ? '✓' : i + 1}</span>
            <div class="min-w-0 flex-1">
              <h2 class="text-base font-semibold {state === 'todo' ? 'text-slate-500' : 'text-slate-100'}">
                {t(s.title)}
                {#if !s.required}
                  <span class="ml-2 align-middle rounded border border-slate-700 px-1.5 py-0.5 text-[10px] font-mono font-normal uppercase tracking-wider text-slate-500">{t('optional')}</span>
                {/if}
              </h2>

              <!-- ---------- step 1: llama.cpp ---------- -->
              {#if s.id === 'llama'}
                {#if state === 'done'}
                  <p class="mt-1 text-sm text-slate-400 font-mono break-all">{st.llama.version ?? st.llama.bin_path}</p>
                  <p class="mt-1 text-xs text-slate-500 font-mono break-all">{st.llama.bin_path}</p>
                {:else if building}
                  <p class="mt-1 text-sm text-slate-400">
                    {t('Working: {step}. This takes a while — the clone is ~250 MB and a CUDA build can run 10–20 minutes.', { step: st.build_active?.current_step ?? '' })}
                  </p>
                {:else}
                  {#if st.llama.bin_exists && !st.llama.bin_ok}
                    <p class="mt-1 text-sm text-amber-300/90">
                      {t('There is a file at {path}, but it did not answer --version.', { path: st.llama.bin_path })}
                    </p>
                  {/if}

                  {#if st.llama.candidates.length > 0}
                    <p class="mt-1 text-sm text-slate-400">{t('Found on this machine — use one of these:')}</p>
                    <ul class="mt-2 space-y-1">
                      {#each st.llama.candidates as c (c.path)}
                        <li class="flex items-center gap-3 text-xs font-mono">
                          <button
                            disabled={busy !== null}
                            onclick={() => useBinary(c.path)}
                            class="rounded border border-cyan-700 bg-cyan-800/30 px-2 py-1 text-slate-100 hover:bg-cyan-700/40 disabled:opacity-50"
                          >{busy === c.path ? '…' : t('Use')}</button>
                          <span class="truncate text-slate-300">{c.path}</span>
                          <span class="text-slate-600">{c.source}</span>
                        </li>
                      {/each}
                    </ul>
                    <div class="my-4 border-t border-slate-800"></div>
                  {/if}

                  <p class="text-sm text-slate-400">
                    {st.llama.candidates.length > 0
                      ? t('Or build a fresh one from source:')
                      : t('No llama-server found. LlamaDeck can clone and build it for you:')}
                  </p>

                  {#if missingTools.length > 0}
                    <div class="mt-2 rounded border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-sm text-amber-200">
                      <p>{t('Missing build tools: {tools}', { tools: missingTools.join(', ') })}</p>
                      <code class="mt-1 block font-mono text-xs text-amber-300/90">sudo apt install {missingTools.join(' ')}</code>
                    </div>
                  {:else}
                    <div class="mt-3 grid gap-3 sm:grid-cols-2">
                      <label class="block">
                        <span class="text-xs text-slate-500">{t('Clone into')}</span>
                        <input
                          bind:value={repoPath}
                          oninput={() => (repoTouched = true)}
                          spellcheck="false"
                          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
                        />
                      </label>
                      <label class="block">
                        <span class="text-xs text-slate-500">{t('Compute backend')}</span>
                        <select bind:value={backendId} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-xs">
                          <option value="auto">{t('auto')}{autoLabel ? ` — ${autoLabel}` : ''}</option>
                          {#each buildableBackends as b (b.id)}
                            <option value={b.id} disabled={!b.available}>
                              {b.label}{b.available ? '' : ` — ${t('not installed')}`}
                            </option>
                          {/each}
                        </select>
                      </label>
                      <label class="block">
                        <span class="text-xs text-slate-500">{t('Parallel jobs (default {n})', { n: st.toolchain.make_jobs })}</span>
                        <input
                          bind:value={jobsOverride}
                          placeholder={String(st.toolchain.make_jobs)}
                          spellcheck="false"
                          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
                        />
                      </label>
                    </div>
                    {#if backendId !== 'auto'}
                      {@const b = st.backends.find((x) => x.id === backendId)}
                      {#if b && !b.available}
                        <p class="mt-2 text-xs text-amber-300/90">{b.detail}</p>
                      {/if}
                    {/if}
                    <button
                      disabled={busy !== null}
                      onclick={installFromSource}
                      class="mt-3 rounded bg-cyan-700/40 border border-cyan-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-cyan-700/60 disabled:opacity-50"
                    >{busy === 'build' ? t('Starting…') : t('Clone and build')}</button>
                  {/if}

                  <div class="mt-4">
                    <button
                      onclick={() => (showManual = !showManual)}
                      class="text-sm text-slate-400 underline decoration-dotted hover:text-slate-200"
                    >{t('I already have llama-server somewhere else')}</button>
                    {#if showManual}
                      <div class="mt-2 flex gap-2">
                        <input
                          bind:value={manualBin}
                          placeholder="/usr/local/bin/llama-server"
                          spellcheck="false"
                          class="flex-1 rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
                        />
                        <button
                          disabled={busy !== null || !manualBin.trim()}
                          onclick={() => useBinary(manualBin)}
                          class="rounded border border-cyan-700 bg-cyan-800/30 px-3 py-1.5 text-xs text-slate-100 hover:bg-cyan-700/40 disabled:opacity-50"
                        >{t('Use')}</button>
                      </div>
                      <p class="mt-1 text-xs text-slate-500">
                        {t('Prebuilt binaries are on the llama.cpp releases page; on macOS, brew install llama.cpp.')}
                      </p>
                    {/if}
                  </div>
                {/if}

                {#if building || buildFailed || log.length > 0}
                  <div class="mt-4">
                    <div class="flex items-center gap-2 text-xs font-mono text-slate-500">
                      <span>{t('build log')}</span>
                      {#if st.build_active}
                        <span class={buildFailed ? 'text-rose-400' : 'text-cyan-400'}>{st.build_active.status} · {st.build_active.current_step}</span>
                      {/if}
                    </div>
                    <div
                      bind:this={logContainer}
                      class="mt-1 h-56 overflow-auto rounded border border-slate-800 bg-slate-950/70 p-2 font-mono text-[11px] leading-relaxed text-slate-400"
                    >
                      {#each log as line, i (i)}<div class="whitespace-pre-wrap break-all">{line}</div>{/each}
                      {#if log.length === 0}<div class="text-slate-600">{t('waiting for output…')}</div>{/if}
                    </div>
                  </div>
                {/if}

              <!-- ---------- step 2: models folder ---------- -->
              {:else if s.id === 'models_dir'}
                {#if state === 'done'}
                  <p class="mt-1 text-xs text-slate-500 font-mono break-all">{st.models.root}</p>
                {:else if state === 'current'}
                  <p class="mt-1 text-sm text-slate-400">
                    {t('Where your GGUF files live. LlamaDeck indexes this folder and downloads into it. Created if it does not exist.')}
                  </p>
                  <div class="mt-2 flex gap-2">
                    <input
                      bind:value={modelsRoot}
                      oninput={() => (modelsRootTouched = true)}
                      placeholder="~/models"
                      spellcheck="false"
                      class="flex-1 rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
                    />
                    <button
                      disabled={busy !== null || !modelsRoot.trim()}
                      onclick={saveModelsRoot}
                      class="rounded bg-cyan-700/40 border border-cyan-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-cyan-700/60 disabled:opacity-50"
                    >{busy === 'models' ? '…' : t('Use this folder')}</button>
                  </div>
                {/if}

              <!-- ---------- step 3: first model ---------- -->
              {:else if s.id === 'model'}
                {#if state === 'done'}
                  <p class="mt-1 text-sm text-slate-400">{t('{n} models indexed', { n: st.models.count })}</p>
                {:else if state === 'current'}
                  <p class="mt-1 text-sm text-slate-400">
                    {t('Two ways in: download one from HuggingFace, or copy GGUF files you already have into your models folder and rescan.')}
                  </p>
                  <p class="mt-2 text-xs font-mono text-slate-500 break-all">{st.models.root}</p>
                  <div class="mt-3 flex flex-wrap gap-3">
                    <a href="/download" class="rounded bg-cyan-700/40 border border-cyan-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-cyan-700/60">{t('Find a model')}</a>
                    <button
                      disabled={busy !== null}
                      onclick={rescan}
                      class="rounded border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700/70 disabled:opacity-50"
                    >{busy === 'rescan' ? t('Scanning…') : t('I copied files in — rescan')}</button>
                  </div>
                {/if}

              <!-- ---------- step 4: first preset ---------- -->
              {:else if s.id === 'preset'}
                {#if state === 'done'}
                  <p class="mt-1 text-sm text-slate-400">{t('{n} presets', { n: st.presets.count })}</p>
                {:else if state === 'current'}
                  <p class="mt-1 text-sm text-slate-400">
                    {t('Pick a model, say what you want it for, and LlamaDeck fills in the flags. You can change every one of them afterwards.')}
                  </p>
                  <a href="/presets?new=1" class="mt-3 inline-block rounded bg-cyan-700/40 border border-cyan-600 px-3 py-1.5 text-sm text-slate-100 hover:bg-cyan-700/60">{t('Start the wizard')}</a>
                {/if}
              {/if}
            </div>
          </div>
        </li>
      {/each}
    </ol>
  {/if}
</div>
