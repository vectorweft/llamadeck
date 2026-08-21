<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type SettingsCheck, type Settings, type LlmEndpoint, type RpcServer } from '$lib/api';
  import { t, i18n, setLocale } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';
  import Skeleton from '$lib/components/Skeleton.svelte';

  let s = $state<Settings | null>(null);
  let saved = $state<string>('');      // JSON snapshot of the last saved state
  let error = $state<string | null>(null);
  let busy = $state(false);
  let check = $state<SettingsCheck | null>(null);
  let checkTimer: ReturnType<typeof setTimeout> | null = null;

  // What the configured OpenAI-compatible endpoint actually serves. The model
  // id is read from the server rather than typed: a name that does not match
  // what is loaded is the single easiest way to break What's New summaries.
  // RPC offload servers. Loaded separately from `s` because their *status* is
  // live process state, not a setting — the row shows whether each is actually
  // listening, which is what decides if its devices exist.
  let rpcServers = $state<RpcServer[]>([]);
  let rpcBusy = $state<string | null>(null);

  async function loadRpc() {
    try { rpcServers = (await api.rpcServers()).servers; } catch { rpcServers = []; }
  }

  async function toggleRpc(srv: RpcServer) {
    rpcBusy = srv.name;
    try {
      await (srv.running ? api.rpcStop(srv.name) : api.rpcStart(srv.name));
      await loadRpc();
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'error');
    } finally { rpcBusy = null; }
  }

  function addRpcServer() {
    if (!s) return;
    const n = (s.rpc_servers ?? []).length;
    s.rpc_servers = [...(s.rpc_servers ?? []), {
      name: `rpc${n}`, binary: '', host: '127.0.0.1',
      port: 50052 + n, devices: [], autostart: false,
    }];
  }

  let llm = $state<LlmEndpoint | null>(null);
  let llmProbing = $state(false);
  let llmTimer: ReturnType<typeof setTimeout> | null = null;
  let llmManual = $state(false);
  // Local llama-servers LlamaDeck is already running — one click beats
  // remembering which preset sits on which port.
  let localServers = $state<{ name: string; url: string }[]>([]);

  async function probeLlm() {
    if (!s || (s.llm_provider ?? 'claude') !== 'openai' || !s.llm_base_url) {
      llm = null;
      return;
    }
    llmProbing = true;
    try {
      llm = await api.llmEndpointProbe(s.llm_base_url, s.llm_api_key);
      // A model that is no longer served would fail at summary time; drop it
      // back to auto so the endpoint decides.
      if (s.llm_model && llm.models.length && !llm.models.includes(s.llm_model)) {
        s.llm_model = '';
      }
    } catch (e) {
      llm = {
        base_url: s.llm_base_url, reachable: false, models: [], n_ctx: null,
        native: false, resolved: null,
        detail: e instanceof Error ? e.message : String(e)
      };
    } finally {
      llmProbing = false;
    }
  }

  const dirty = $derived(s ? JSON.stringify(s) !== saved : false);
  // Bind/port changes only take effect on the next process start.
  const needsRestart = $derived.by(() => {
    if (!s || !saved) return false;
    const old = JSON.parse(saved) as Settings;
    return (
      old.controller_bind_host !== s.controller_bind_host ||
      old.controller_bind_port !== s.controller_bind_port ||
      old.mcp_bind_host !== s.mcp_bind_host ||
      old.mcp_bind_port !== s.mcp_bind_port
    );
  });

  onMount(async () => {
    try {
      s = await api.getSettings();
      saved = JSON.stringify(s);
      probeLlm();
      loadRpc();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
    try {
      const { presets } = await api.serverStatuses();
      localServers = Object.values(presets)
        .filter((p) => p.running && p.port)
        .map((p) => ({ name: p.name, url: `http://127.0.0.1:${p.port}` }));
    } catch { localServers = []; }
  });

  // Re-probe when the endpoint the user is typing settles.
  $effect(() => {
    if (!s) return;
    const key = `${s.llm_provider}|${s.llm_base_url}|${s.llm_api_key ?? ''}`;
    if (llmTimer) clearTimeout(llmTimer);
    llmTimer = setTimeout(() => { void key; probeLlm(); }, 500);
  });

  // Live path validation (debounced) — a typo shows up before Save, not as a
  // mysterious "no models found" three pages later.
  $effect(() => {
    if (!s) return;
    const draft = JSON.stringify({
      llama_bin: s.llama_bin,
      llama_repo: s.llama_repo,
      hf_models_root: s.hf_models_root,
      scan_roots: s.scan_roots
    });
    if (checkTimer) clearTimeout(checkTimer);
    checkTimer = setTimeout(async () => {
      try { check = await api.settingsCheck(JSON.parse(draft)); } catch { check = null; }
    }, 400);
  });

  async function save() {
    if (!s) return;
    busy = true;
    error = null;
    try {
      s = await api.putSettings(s);
      saved = JSON.stringify(s);
      // Keep the UI language in sync with what was just persisted.
      if (s.ui_language && s.ui_language !== i18n.locale) setLocale(s.ui_language);
      toast(t('Settings saved'), 'success');
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function revert() {
    if (!saved) return;
    s = JSON.parse(saved);
    toast(t('Reverted to saved settings'), 'info');
  }

  async function rescan() {
    busy = true;
    try {
      const r = await api.scanModels();
      toast(t('+{a} added · {u} updated · {r} removed · {n} total', { a: r.added, u: r.updated, r: r.removed, n: r.total }), 'success');
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function addRoot() {
    if (!s) return;
    s.scan_roots = [...s.scan_roots, ''];
  }
  function removeRoot(i: number) {
    if (!s) return;
    s.scan_roots = s.scan_roots.filter((_, j) => j !== i);
  }
</script>

{#snippet probe(p: { exists: boolean; kind: string | null; ok: boolean } | undefined, want: string)}
  {#if p}
    {#if p.ok}
      <span class="text-[11px] font-mono text-emerald-400">✓ {t('found')}</span>
    {:else if p.exists}
      <span class="text-[11px] font-mono text-amber-400">
        ⚠ {want === 'file' ? t('this is a directory, not a file') : t('this is a file, not a directory')}
      </span>
    {:else}
      <span class="text-[11px] font-mono text-rose-400">✕ {t('not found')}</span>
    {/if}
  {/if}
{/snippet}

<div class="max-w-3xl space-y-6">
  <div class="flex items-center gap-3 flex-wrap">
    <h1 class="text-2xl font-semibold">{t('Settings')}</h1>
    <span class="text-xs text-slate-500 font-mono">~/.config/llamadeck/settings.json</span>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  {#if !s}
    <Skeleton class="h-32 w-full" />
    <Skeleton class="h-32 w-full" />
  {:else}
    <!-- llama.cpp paths — the first thing a fresh install must get right -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div>
        <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('llama.cpp')}</h2>
        <p class="text-xs text-slate-500 mt-1">{t('Where your llama-server binary and models live. LlamaDeck never downloads or builds these behind your back.')}</p>
      </div>

      <label class="block">
        <span class="text-sm text-slate-400">llama_bin</span>
        <input bind:value={s.llama_bin} spellcheck="false" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" placeholder="~/llama.cpp/build/bin/llama-server" />
        <div class="mt-1 flex items-center gap-2">
          {@render probe(check?.llama_bin, 'file')}
          <span class="text-[11px] text-slate-600">{t('the llama-server executable LlamaDeck starts')}</span>
        </div>
      </label>

      <label class="block">
        <span class="text-sm text-slate-400">llama_repo</span>
        <input bind:value={s.llama_repo} spellcheck="false" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" placeholder="~/llama.cpp" />
        <div class="mt-1 flex items-center gap-2">
          {@render probe(check?.llama_repo, 'dir')}
          <span class="text-[11px] text-slate-600">{t('source checkout — only needed for the Build page')}</span>
        </div>
      </label>

      <label class="block">
        <span class="text-sm text-slate-400">hf_models_root</span>
        <input bind:value={s.hf_models_root} spellcheck="false" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" placeholder="~/llama.cpp/models" />
        <div class="mt-1 flex items-center gap-2">
          {@render probe(check?.hf_models_root, 'dir')}
          <span class="text-[11px] text-slate-600">{t('where HuggingFace downloads land')}</span>
        </div>
      </label>
    </section>

    <!-- RPC offload servers -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div class="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('RPC offload servers')}</h2>
          <p class="text-xs text-slate-500 mt-1">
            {t('Runs a second llama.cpp backend that the main binary cannot host — ROCm alongside CUDA, or a GPU in another machine. Its devices appear as RPC0 in a preset\'s GPU list once the server is running.')}
          </p>
        </div>
        <button onclick={addRpcServer}
          class="rounded border border-slate-700 px-2.5 py-1 text-xs font-mono text-slate-300 hover:bg-slate-800">
          + {t('add')}
        </button>
      </div>

      {#each (s.rpc_servers ?? []) as srv, i (i)}
        {@const live = rpcServers.find(r => r.name === srv.name)}
        <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-2">
          <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <label class="block"><span class="text-[11px] text-slate-500">{t('name')}</span>
              <input bind:value={srv.name} class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" /></label>
            <label class="block"><span class="text-[11px] text-slate-500">{t('host')}</span>
              <input bind:value={srv.host} class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" /></label>
            <label class="block"><span class="text-[11px] text-slate-500">{t('port')}</span>
              <input type="number" bind:value={srv.port} class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" /></label>
            <label class="block"><span class="text-[11px] text-slate-500">{t('devices to export')}</span>
              <input value={(srv.devices ?? []).join(',')}
                onchange={(e) => srv.devices = (e.currentTarget as HTMLInputElement).value.split(',').map(v => v.trim()).filter(Boolean)}
                placeholder="ROCm0"
                class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" /></label>
          </div>
          <label class="block"><span class="text-[11px] text-slate-500">{t('binary (blank = auto-detect)')}</span>
            <input bind:value={srv.binary} spellcheck="false" placeholder={live?.binary || '…/build-hip/bin/ggml-rpc-server'}
              class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" /></label>
          <div class="flex flex-wrap items-center gap-3 text-xs font-mono">
            <label class="flex items-center gap-1.5 text-slate-400">
              <input type="checkbox" bind:checked={srv.autostart} /> {t('autostart')}
            </label>
            <span class={live?.running ? 'text-emerald-400' : 'text-slate-500'}>
              {live?.running ? t('listening') : t('stopped')}{live?.running && !live?.owned ? ` · ${t('started outside LlamaDeck')}` : ''}
            </span>
            <button onclick={() => live && toggleRpc(live)} disabled={!live || rpcBusy === srv.name}
              class="rounded border border-slate-700 px-2 py-0.5 text-slate-300 hover:bg-slate-800 disabled:opacity-40">
              {live?.running ? t('stop') : t('start')}
            </button>
            <button onclick={() => s && (s.rpc_servers = (s.rpc_servers ?? []).filter((_, j) => j !== i))}
              class="rounded border border-rose-900 px-2 py-0.5 text-rose-300 hover:bg-rose-950/40">{t('remove')}</button>
          </div>
          {#if live?.last_error}
            <pre class="whitespace-pre-wrap rounded bg-rose-950/30 px-2 py-1 text-[11px] text-rose-200">{live.last_error}</pre>
          {/if}
        </div>
      {:else}
        <p class="text-xs text-slate-600 font-mono">{t('none configured — most machines need none.')}</p>
      {/each}
    </section>

    <!-- Model scan roots -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-3">
      <div class="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('Model scan roots')}</h2>
          <p class="text-xs text-slate-500 mt-1">{t('Directories scanned for .gguf files (recursively). Add every disk that holds models.')}</p>
        </div>
        <button onclick={addRoot} class="shrink-0 rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60">+ {t('Add root')}</button>
      </div>
      <div class="space-y-2">
        {#each s.scan_roots as _, i}
          <div class="flex items-center gap-2">
            <input bind:value={s.scan_roots[i]} spellcheck="false" class="flex-1 rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" placeholder="/path/to/models" />
            <span class="w-28 shrink-0">{@render probe(check?.scan_roots?.[i], 'dir')}</span>
            <button onclick={() => removeRoot(i)} aria-label={t('Remove')} class="shrink-0 rounded border border-slate-700 px-2 py-1 text-xs text-slate-500 hover:text-rose-300 hover:border-rose-800">✕</button>
          </div>
        {:else}
          <div class="text-xs text-slate-500 italic">{t('No scan roots — the Models page will stay empty.')}</div>
        {/each}
      </div>
      <div class="pt-1">
        <button onclick={rescan} disabled={busy} class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1 text-xs hover:bg-emerald-700/60 disabled:opacity-40">{t('Save, then rescan models')}</button>
        <span class="ml-2 text-[11px] text-slate-600">{t('run this after changing the roots')}</span>
      </div>
    </section>

    <!-- Interface -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('Interface')}</h2>
      <label class="block max-w-xs">
        <span class="text-sm text-slate-400">{t('Language')}</span>
        <select
          value={s.ui_language ?? 'en'}
          onchange={(e) => { if (s) s.ui_language = (e.currentTarget as HTMLSelectElement).value as 'en' | 'tr'; }}
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm"
        >
          <option value="en">English</option>
          <option value="tr">Türkçe</option>
        </select>
        <span class="mt-1 block text-[11px] text-slate-600">{t('Also sets the language of generated What\'s New cards and the build guide.')}</span>
      </label>
    </section>

    <!-- AI provider — which LLM writes What's New cards and the build guide -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div>
        <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('AI provider')}</h2>
        <p class="text-xs text-slate-500 mt-1">{t('Which LLM writes the What\'s New summary cards and the build guide. Keys are stored in plain text in the local settings file.')}</p>
      </div>
      <label class="block max-w-md">
        <span class="text-sm text-slate-400">{t('Provider')}</span>
        <select
          value={s.llm_provider ?? 'claude'}
          onchange={(e) => { if (s) s.llm_provider = (e.currentTarget as HTMLSelectElement).value as 'claude' | 'openai'; }}
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm"
        >
          <option value="claude">{t('Claude (subscription or API key)')}</option>
          <option value="openai">{t('OpenAI-compatible API (OpenRouter, local model, …)')}</option>
        </select>
      </label>

      {#if (s.llm_provider ?? 'claude') === 'claude'}
        <label class="block">
          <span class="text-sm text-slate-400">anthropic_api_key</span>
          <input
            type="password"
            value={s.anthropic_api_key ?? ''}
            oninput={(e) => { if (s) s.anthropic_api_key = (e.currentTarget as HTMLInputElement).value || null; }}
            spellcheck="false"
            class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
            placeholder="sk-ant-…"
          />
          <span class="mt-1 block text-[11px] text-slate-600">{t('optional: What\'s New summaries also work via a local Claude Code session')}</span>
        </label>
      {:else}
        <label class="block">
          <span class="text-sm text-slate-400">llm_base_url</span>
          <input
            bind:value={s.llm_base_url}
            spellcheck="false"
            class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
            placeholder="https://openrouter.ai/api/v1"
          />
          <div class="mt-1.5 flex items-center gap-1.5 flex-wrap">
            <span class="text-[11px] text-slate-600">{t('quick fill:')}</span>
            {#each [
              ...localServers.map((sv) => ({ label: `▶ ${sv.name}`, url: sv.url })),
              { label: 'OpenRouter', url: 'https://openrouter.ai/api/v1' },
              { label: 'OpenAI', url: 'https://api.openai.com/v1' },
              { label: 'Groq', url: 'https://api.groq.com/openai/v1' }
            ] as preset}
              <button
                onclick={() => { if (s) s.llm_base_url = preset.url; }}
                class="rounded border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-slate-700/60"
              >{preset.label}</button>
            {/each}
          </div>
        </label>
        <div class="block">
          <div class="flex items-center justify-between gap-2">
            <span class="text-sm text-slate-400">llm_model</span>
            <button
              onclick={probeLlm}
              disabled={llmProbing || !s.llm_base_url}
              class="rounded border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-slate-700/60 disabled:opacity-40"
            >{llmProbing ? t('checking…') : t('refresh')}</button>
          </div>

          {#if llmManual || (llm && llm.reachable && llm.models.length === 0)}
            <input
              bind:value={s.llm_model}
              spellcheck="false"
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
              placeholder="anthropic/claude-sonnet-4.5"
            />
          {:else}
            <select
              value={s.llm_model ?? ''}
              onchange={(e) => { if (s) s.llm_model = (e.currentTarget as HTMLSelectElement).value; }}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
            >
              <option value="">{t('Automatic — whatever the server has loaded')}</option>
              {#each llm?.models ?? [] as m}
                <option value={m}>{m.split('/').pop()}</option>
              {/each}
              {#if s.llm_model && !(llm?.models ?? []).includes(s.llm_model)}
                <option value={s.llm_model}>{s.llm_model}</option>
              {/if}
            </select>
          {/if}

          <div class="mt-1.5 text-[11px]">
            {#if llmProbing && !llm}
              <span class="text-slate-600">{t('reading the endpoint…')}</span>
            {:else if llm && !llm.reachable}
              <span class="text-rose-400">{llm.detail ?? t('endpoint unreachable')}</span>
            {:else if llm?.detail}
              <span class="text-amber-400">{llm.detail}</span>
            {:else if llm?.resolved}
              <span class="text-emerald-400">
                {t('will use')} <span class="font-mono">{llm.resolved.split('/').pop()}</span>
              </span>
              {#if llm.n_ctx}
                <span class="text-slate-600"> · context {llm.n_ctx.toLocaleString()}</span>
              {/if}
            {:else}
              <span class="text-slate-600">{t('leave on automatic for a local server — the model id is read from it')}</span>
            {/if}
          </div>

          <button
            onclick={() => { llmManual = !llmManual; }}
            class="mt-1 text-[11px] text-slate-600 underline hover:text-slate-400"
          >{llmManual ? t('pick from the list') : t('enter the id by hand')}</button>
        </div>
        <label class="block">
          <span class="text-sm text-slate-400">llm_api_key</span>
          <input
            type="password"
            value={s.llm_api_key ?? ''}
            oninput={(e) => { if (s) s.llm_api_key = (e.currentTarget as HTMLInputElement).value || null; }}
            spellcheck="false"
            class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
            placeholder="sk-or-…"
          />
          <span class="mt-1 block text-[11px] text-slate-600">{t('sent as a Bearer token — leave empty for a local server')}</span>
        </label>
      {/if}
    </section>

    <!-- Credentials -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div>
        <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('Credentials')}</h2>
        <p class="text-xs text-slate-500 mt-1">{t('Stored in plain text in the local settings file. Optional.')}</p>
      </div>
      <label class="block">
        <span class="text-sm text-slate-400">hf_token</span>
        <input
          type="password"
          value={s.hf_token ?? ''}
          oninput={(e) => { if (s) s.hf_token = (e.currentTarget as HTMLInputElement).value || null; }}
          spellcheck="false"
          class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
          placeholder="hf_…"
        />
        <span class="mt-1 block text-[11px] text-slate-600">{t('needed only for gated or private repos')}</span>
      </label>
    </section>

    <!-- Network -->
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div>
        <h2 class="text-sm uppercase tracking-wider text-slate-400">{t('Network')}</h2>
        <p class="text-xs text-slate-500 mt-1">{t('LlamaDeck has no authentication — keep it on localhost unless your network is trusted.')}</p>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <label class="block">
          <span class="text-sm text-slate-400">controller_bind_host</span>
          <input bind:value={s.controller_bind_host} spellcheck="false" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" />
        </label>
        <label class="block">
          <span class="text-sm text-slate-400">controller_bind_port</span>
          <input type="number" bind:value={s.controller_bind_port} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" />
        </label>
        <label class="block">
          <span class="text-sm text-slate-400">mcp_bind_host</span>
          <input bind:value={s.mcp_bind_host} spellcheck="false" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" />
        </label>
        <label class="block">
          <span class="text-sm text-slate-400">mcp_bind_port</span>
          <input type="number" bind:value={s.mcp_bind_port} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs" />
        </label>
        <label class="block col-span-2">
          <span class="text-sm text-slate-400">lan_token</span>
          <input
            value={s.lan_token ?? ''}
            oninput={(e) => { if (s) s.lan_token = (e.currentTarget as HTMLInputElement).value || null; }}
            spellcheck="false"
            class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1.5 font-mono text-xs"
          />
          <span class="mt-1 block text-[11px] text-slate-600">{t('required header token when binding beyond localhost')}</span>
        </label>
      </div>
      {#if needsRestart}
        <div class="rounded border border-amber-800 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
          {t('Bind host/port changes take effect after a backend restart (Dashboard → Restart backend).')}
        </div>
      {/if}
    </section>

    <!-- Sticky save bar -->
    <div class="sticky bottom-4 flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-900/95 px-4 py-3 shadow-xl backdrop-blur">
      <span class="text-xs font-mono {dirty ? 'text-amber-300' : 'text-slate-500'}">
        {dirty ? t('unsaved changes') : t('all changes saved')}
      </span>
      <div class="ml-auto flex gap-2">
        <button
          onclick={revert}
          disabled={!dirty || busy}
          class="rounded bg-slate-700/40 border border-slate-600 px-4 py-1.5 text-sm hover:bg-slate-700/60 disabled:opacity-40"
        >{t('Revert')}</button>
        <button
          onclick={save}
          disabled={!dirty || busy}
          class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
        >{t('Save')}</button>
      </div>
    </div>
  {/if}
</div>
