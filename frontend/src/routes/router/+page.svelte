<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { api, type LlamaConfig, type RouterActive, type RouterModel } from '$lib/api';
  import { confirmDialog, alertDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';

  let active = $state<RouterActive | null>(null);
  let models = $state<RouterModel[]>([]);
  let presets = $state<LlamaConfig[]>([]);
  let routerPresets = $derived(presets.filter((p) => p.mode === 'router'));
  let iniPreview = $state<string>('');
  let iniBusy = $state<string | null>(null);
  let busy = $state<Record<string, string>>({});
  let error = $state<string | null>(null);
  let poll: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      const [a, ps] = await Promise.all([api.routerActive(), api.listPresets()]);
      active = a;
      presets = ps;
      if (a.running) {
        try {
          const r = await api.routerModels();
          models = r.data ?? [];
        } catch (e) {
          models = [];
          error = e instanceof Error ? e.message : String(e);
        }
      } else {
        models = [];
      }
      if (active.running || iniPreview === '') {
        try {
          const p = await api.routerIniPreview();
          iniPreview = p.ini;
        } catch {
          // INI preview only works if router is running or models_dir provided
        }
      }
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function startRouter(name: string) {
    busy = { ...busy, [name]: 'starting' };
    try {
      await api.serverStart(name);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = { ...busy, [name]: '' };
    }
  }

  async function stopRouter(name: string) {
    const ok = await confirmDialog(
      t('All loaded models will be evicted from memory.'),
      { title: t("Stop the router preset '{name}'?", { name }), danger: true, confirmLabel: t('Stop') }
    );
    if (!ok) return;
    busy = { ...busy, [name]: 'stopping' };
    try {
      await api.serverStop(name);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = { ...busy, [name]: '' };
    }
  }

  async function loadModel(id: string) {
    busy = { ...busy, [id]: 'loading' };
    try {
      await api.routerLoad(id);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = { ...busy, [id]: '' };
    }
  }

  async function unloadModel(id: string) {
    busy = { ...busy, [id]: 'unloading' };
    try {
      await api.routerUnload(id);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = { ...busy, [id]: '' };
    }
  }

  async function rewriteIni() {
    iniBusy = 'writing';
    try {
      const r = await api.routerIniWrite();
      iniPreview = r.ini;
      await alertDialog(t('Wrote {bytes} bytes to {path}.\nRestart the router for changes to take effect.', { bytes: r.bytes, path: r.path }), { title: t('INI written') });
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      iniBusy = null;
    }
  }

  function statusColor(v: string | undefined): string {
    if (v === 'loaded') return 'bg-emerald-500/20 text-emerald-300';
    if (v === 'loading') return 'bg-amber-500/20 text-amber-300';
    if (v === 'sleeping') return 'bg-cyan-500/20 text-cyan-300';
    return 'bg-slate-700/40 text-slate-400';
  }

  onMount(() => {
    refresh();
    poll = setInterval(refresh, 4000);
  });

  onDestroy(() => {
    if (poll) clearInterval(poll);
  });
</script>

<div class="max-w-5xl space-y-6">
  <header class="flex items-baseline justify-between">
    <h1 class="text-xl font-mono text-slate-100">Router</h1>
    <p class="text-xs text-slate-500">
      <code class="text-slate-400">/models/{`{load,unload}`}</code> — {t('restart-free model switching without dropping the process.')}
    </p>
  </header>

  <p class="rounded border border-slate-800 bg-slate-900/40 px-3 py-2 text-xs text-slate-400">
    {t('This page manages the models inside a running router process.')}
    {t('To start or stop the process itself, use the')}
    <a href="/server" class="text-emerald-400 hover:underline">Server</a>
    {t('page.')}
  </p>

  {#if error}
    <div class="rounded border border-rose-700 bg-rose-900/30 px-3 py-2 text-sm text-rose-200">{error}</div>
  {/if}

  <!-- Router preset selector / controls -->
  <section class="space-y-2">
    <h2 class="text-sm font-mono uppercase tracking-wider text-slate-500">Router presets</h2>
    {#if routerPresets.length === 0}
      <div class="rounded border border-slate-800 bg-slate-900/40 px-3 py-2 text-sm text-slate-400">
        {t('No router-mode preset defined. On the')} <a href="/presets" class="text-emerald-400 underline">Presets</a> {t('page, create one with')} <code>mode = "router"</code>.
      </div>
    {:else}
      <div class="grid gap-2">
        {#each routerPresets as p (p.name)}
          {@const isActive = active?.preset === p.name}
          {@const running = isActive && active?.running}
          <div class="rounded border border-slate-800 bg-slate-900/40 px-3 py-3 flex items-center gap-4">
            <div class="flex-1">
              <div class="font-mono text-sm text-slate-200">
                {p.name}
                <span class="ml-2 text-xs text-slate-500">:{p.port}</span>
                {#if running}
                  <span class="ml-2 px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[11px] uppercase">running</span>
                {/if}
              </div>
              <div class="text-xs text-slate-500 mt-0.5">
                models_dir: <code class="text-slate-400">{p.models_dir ?? '—'}</code>
                · max {p.models_max ?? 1}
                · autoload {p.models_autoload ? 'on' : 'off'}
                · ctx {p.ctx_size}
                · ngl {p.n_gpu_layers}
              </div>
            </div>
            {#if running}
              <button
                onclick={() => stopRouter(p.name)}
                disabled={!!busy[p.name]}
                class="px-3 py-1.5 rounded bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-sm">
                {busy[p.name] === 'stopping' ? t('Stopping…') : t('Stop')}
              </button>
            {:else}
              <button
                onclick={() => startRouter(p.name)}
                disabled={!!busy[p.name]}
                class="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-sm">
                {busy[p.name] === 'starting' ? t('Starting…') : t('Start')}
              </button>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </section>

  <!-- Loaded / loadable models -->
  {#if active?.running}
    <section class="space-y-2">
      <h2 class="text-sm font-mono uppercase tracking-wider text-slate-500">
        Models on <code class="text-slate-400">{active.preset}</code>
      </h2>
      {#if models.length === 0}
        <div class="text-sm text-slate-500">{t("No models found. Check the router's models_dir setting.")}</div>
      {:else}
        <div class="rounded border border-slate-800 overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="bg-slate-900/60 text-xs text-slate-500 uppercase">
              <tr>
                <th class="text-left px-3 py-2">Model id</th>
                <th class="text-left px-3 py-2">Status</th>
                <th class="text-left px-3 py-2">Source</th>
                <th class="text-right px-3 py-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {#each models as m (m.id)}
                {@const v = m.status?.value}
                <tr class="border-t border-slate-800/60">
                  <td class="px-3 py-2 font-mono text-slate-200">{m.id}</td>
                  <td class="px-3 py-2">
                    <span class="px-1.5 py-0.5 rounded text-[11px] uppercase {statusColor(v)}">{v ?? 'unknown'}</span>
                    {#if m.status?.failed}
                      <span class="ml-1 text-xs text-rose-300">{t('failed (exit {code})', { code: m.status.exit_code ?? '?' })}</span>
                    {/if}
                  </td>
                  <td class="px-3 py-2 text-xs text-slate-500">
                    {#if m.in_cache}<span>cache</span>{:else if m.path}<code class="text-slate-400">{m.path}</code>{:else}<span>—</span>{/if}
                  </td>
                  <td class="px-3 py-2 text-right">
                    {#if v === 'loaded' || v === 'sleeping' || v === 'loading'}
                      <button
                        onclick={() => unloadModel(m.id)}
                        disabled={!!busy[m.id]}
                        class="px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs">
                        {busy[m.id] === 'unloading' ? t('Unloading…') : t('Unload')}
                      </button>
                    {:else}
                      <button
                        onclick={() => loadModel(m.id)}
                        disabled={!!busy[m.id]}
                        class="px-2 py-1 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-xs">
                        {busy[m.id] === 'loading' ? t('Loading…') : t('Load')}
                      </button>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>
  {/if}

  <!-- INI preview / regenerate -->
  <section class="space-y-2">
    <div class="flex items-center justify-between">
      <h2 class="text-sm font-mono uppercase tracking-wider text-slate-500">Per-model INI overrides</h2>
      <button
        onclick={rewriteIni}
        disabled={iniBusy !== null || !active?.running}
        class="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-xs">
        {iniBusy ? t('Writing…') : t('Regenerate router-models.ini')}
      </button>
    </div>
    <!-- eslint-disable-next-line svelte/no-at-html-tags — sözlükten gelen kendi HTML'imiz -->
    <p class="text-xs text-slate-500">{@html t("<strong>The INI is the single source of truth for per-model settings.</strong> The <code>[*]</code> global section comes from the router preset (ctx/ngl/parallel defaults). Each <code>[&lt;model_id&gt;]</code> section is generated from a sibling single-mode preset whose <code>model_path</code> lives under <code>models_dir</code>; section keys override <code>[*]</code>. The router's CLI is deliberately kept minimal (only bind + control flags) so per-model INI settings actually win — llama-server precedence is CLI &gt; per-model &gt; <code>[*]</code>. After writing, restart the router for changes to take effect.")}</p>
    <pre class="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs font-mono text-slate-300 overflow-x-auto whitespace-pre">{iniPreview || t('(start a router preset to see a preview)')}</pre>
  </section>
</div>
