<script lang="ts">
  /**
   * Logs page: the llama-server output for one preset, live.
   *
   * The Server page tells you a preset is running; this is where you find out
   * what it said while doing it — the load line, the device split, the reason
   * it exited. Two sources, deliberately: a tail of the file on disk (so a
   * preset that died an hour ago still has its last words) and, for a running
   * one, the same SSE stream the supervisor feeds. Tail first, then stream, so
   * the view opens full rather than empty and filling.
   */
  import { onDestroy, onMount, tick } from 'svelte';
  import { api, type PresetStatus } from '$lib/api';
  import { t } from '$lib/i18n.svelte';

  let statuses = $state<Record<string, PresetStatus>>({});
  let selected = $state<string>('');
  let lines = $state<string[]>([]);
  let filter = $state('');
  let error = $state<string | null>(null);
  let loading = $state(false);
  let autoScroll = $state(true);

  let source: EventSource | null = null;
  let logContainer: HTMLDivElement | null = $state(null);
  let poll: ReturnType<typeof setInterval> | null = null;

  const presetNames = $derived(Object.keys(statuses).sort());
  const current = $derived(selected ? statuses[selected] : undefined);
  const shown = $derived.by(() => {
    const f = filter.trim().toLowerCase();
    if (!f) return lines;
    return lines.filter((l) => l.toLowerCase().includes(f));
  });

  async function refreshStatuses() {
    try {
      const r = await api.serverStatuses();
      statuses = r.presets;
      // First load: land on something worth reading — a running preset if there
      // is one, otherwise just the first, rather than an empty page.
      if (!selected) {
        const names = Object.keys(r.presets).sort();
        const running = names.find((n) => r.presets[n].running);
        const pick = running ?? names[0];
        if (pick) await select(pick);
      }
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function select(name: string) {
    selected = name;
    source?.close();
    source = null;
    lines = [];
    // Exactly one source per preset, because both read the same in-memory ring:
    // the stream replays its scrollback before going live, so tailing as well
    // printed the overlap twice. A running preset gets the stream (scrollback
    // then live); a stopped one has nothing more to say, so it gets the tail
    // and no connection left hanging open.
    if (statuses[name]?.running) {
      openStream(name);
      return;
    }
    loading = true;
    try {
      const r = await api.logsTail(name, 1000);
      lines = r.lines;
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
    await scrollToEnd();
  }

  function openStream(name: string) {
    source?.close();
    source = new EventSource(`/api/server/logs/stream/${encodeURIComponent(name)}`);
    source.onmessage = async (ev) => {
      try {
        const data = JSON.parse(ev.data);
        lines = [...lines, data.line];
        if (lines.length > 5000) lines = lines.slice(-5000);
        if (autoScroll) await scrollToEnd();
      } catch { /* ignore malformed frame */ }
    };
  }

  async function scrollToEnd() {
    await tick();
    logContainer?.scrollTo({ top: logContainer.scrollHeight });
  }

  onMount(() => {
    refreshStatuses();
    // Cheap: statuses is the same endpoint the dashboard polls. Picking up a
    // preset that started since the page opened matters more than the request.
    poll = setInterval(refreshStatuses, 5000);
  });
  onDestroy(() => {
    source?.close();
    if (poll) clearInterval(poll);
  });
</script>

<div class="max-w-6xl space-y-6">
  <div class="flex items-center gap-3 flex-wrap">
    <h1 class="text-2xl font-semibold">{t('Logs')}</h1>
    <span class="text-xs text-slate-500 font-mono">{t('llama-server output, per preset')}</span>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  {#if presetNames.length === 0}
    <p class="text-sm text-slate-500">{t('No presets yet — create one on the Presets page.')}</p>
  {:else}
    <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div class="flex items-center gap-2 flex-wrap">
        {#each presetNames as name (name)}
          <button
            onclick={() => select(name)}
            class="rounded border px-3 py-1 text-xs font-mono {selected === name
              ? 'border-emerald-600 bg-emerald-700/30 text-emerald-100'
              : 'border-slate-700 bg-slate-800/60 text-slate-300 hover:bg-slate-800'}"
          >
            {name}
            {#if statuses[name].running}
              <span class="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 align-middle"></span>
            {/if}
          </button>
        {/each}
      </div>

      <div class="flex items-center justify-between flex-wrap gap-3 pt-2 border-t border-slate-800">
        <div class="text-xs font-mono text-slate-500">
          {#if current}
            {current.running
              ? t('running · pid {pid} · :{port}', { pid: current.pid ?? '—', port: current.port })
              : t('stopped · last output still in memory; the full log is the file below')}
            {#if current.log_file}
              <span class="text-slate-600"> · {current.log_file}</span>
            {/if}
          {/if}
        </div>
        <div class="flex items-center gap-3">
          <input
            bind:value={filter}
            placeholder={t('filter…')}
            class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs font-mono w-44"
          />
          <label class="flex items-center gap-1.5 text-xs text-slate-500">
            <input type="checkbox" bind:checked={autoScroll} class="accent-emerald-600" />
            {t('follow')}
          </label>
        </div>
      </div>

      <div
        bind:this={logContainer}
        class="h-[32rem] overflow-y-auto rounded bg-slate-950 border border-slate-800 p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap"
      >{#if loading}<span class="text-slate-600">{t('loading…')}</span>{:else}{#each shown as line, i (i)}{line}{'\n'}{:else}<span class="text-slate-600">{filter.trim() ? t('nothing matches the filter') : t('this log is empty')}</span>{/each}{/if}</div>

      {#if filter.trim()}
        <p class="text-xs font-mono text-slate-500">
          {t('{shown} of {total} lines', { shown: shown.length, total: lines.length })}
        </p>
      {/if}
    </section>
  {/if}
</div>
