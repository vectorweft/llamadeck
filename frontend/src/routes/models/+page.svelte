<script lang="ts">
  import { t } from '$lib/i18n.svelte';
  import Skeleton from '$lib/components/Skeleton.svelte';
  import { onMount } from 'svelte';
  import { api, type ModelEntry, type ModelInfoBundle, type VerifyJob } from '$lib/api';

  let models = $state<ModelEntry[]>([]);
  let families = $state<string[]>([]);
  let filterFamily = $state<string>('');
  let filterText = $state<string>('');
  let error = $state<string | null>(null);
  let scanning = $state(false);
  let scanMessage = $state<string | null>(null);

  let openPath = $state<string | null>(null);
  let infoCache = $state<Record<string, ModelInfoBundle>>({});
  let infoLoading = $state<string | null>(null);
  let infoError = $state<Record<string, string>>({});

  // Checksum verification. A corrupted model keeps its exact size, so nothing
  // short of hashing tells the difference between a good copy and one that
  // will produce garbage — see verify.py.
  let verify = $state<Record<string, VerifyJob>>({});
  let verifyError = $state<Record<string, string>>({});

  async function runVerify(path: string) {
    verifyError = { ...verifyError, [path]: '' };
    try {
      verify = { ...verify, [path]: await api.startVerify(path) };
      // Hashing 100 GB takes minutes; poll until the job reports it is done.
      while (verify[path]?.state === 'running') {
        await new Promise(r => setTimeout(r, 1000));
        verify = { ...verify, [path]: await api.verifyStatus(path) };
      }
    } catch (e) {
      verifyError = { ...verifyError, [path]: e instanceof Error ? e.message : String(e) };
    }
  }

  function verdictClass(v: string): string {
    if (v === 'ok') return 'text-emerald-300';
    if (v === 'corrupt' || v === 'incomplete') return 'text-rose-300';
    return 'text-amber-300';
  }

  function verdictText(v: string): string {
    if (v === 'ok') return t('Verified — every part matches its recorded checksum.');
    if (v === 'corrupt') return t('Damaged — the contents do not match. Re-copy or re-download the parts marked below.');
    if (v === 'incomplete') return t('Incomplete — a part of this model is not on disk. Download the parts marked missing.');
    if (v === 'running') return t('Hashing…');
    return t('No recorded checksum to compare against, so this cannot be confirmed either way.');
  }

  // Sortable columns: click a header to sort, click again to flip direction.
  type SortKey = 'family' | 'quant' | 'size_bytes' | 'path' | 'last_used';
  let sortKey = $state<SortKey>('family');
  let sortDir = $state<1 | -1>(1);
  function setSort(k: SortKey) {
    if (sortKey === k) sortDir = sortDir === 1 ? -1 : 1;
    else { sortKey = k; sortDir = k === 'size_bytes' || k === 'last_used' ? -1 : 1; }
  }
  function sortArrow(k: SortKey): string {
    return sortKey === k ? (sortDir === 1 ? ' ▲' : ' ▼') : '';
  }

  const filtered = $derived.by(() => {
    const needle = filterText.trim().toLowerCase();
    const out = models.filter(m => {
      if (filterFamily && m.family !== filterFamily) return false;
      if (!needle) return true;
      return m.path.toLowerCase().includes(needle) ||
             (m.family ?? '').toLowerCase().includes(needle) ||
             (m.quant ?? '').toLowerCase().includes(needle);
    });
    out.sort((a, b) => {
      const av = a[sortKey] ?? (typeof b[sortKey] === 'number' ? 0 : '');
      const bv = b[sortKey] ?? (typeof a[sortKey] === 'number' ? 0 : '');
      const c = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av).localeCompare(String(bv));
      return c * sortDir;
    });
    return out;
  });

  const totalSize = $derived(filtered.reduce((s, m) => s + m.size_bytes, 0) / 1024**3);

  async function load() {
    try {
      [models, families] = await Promise.all([api.listModels(), api.modelFamilies()]);
      error = null;
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
  }

  async function rescan() {
    scanning = true;
    scanMessage = null;
    try {
      const r = await api.scanModels();
      scanMessage = t('+{a} added · {u} updated · {r} removed · {n} total', { a: r.added, u: r.updated, r: r.removed, n: r.total });
      await load();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      scanning = false;
    }
  }

  async function toggleInfo(path: string) {
    if (openPath === path) { openPath = null; return; }
    openPath = path;
    if (infoCache[path] || infoLoading === path) return;
    infoLoading = path;
    try {
      const bundle = await api.modelInfo(path);
      infoCache = { ...infoCache, [path]: bundle };
      delete infoError[path];
      infoError = { ...infoError };
    } catch (e) {
      infoError = { ...infoError, [path]: e instanceof Error ? e.message : String(e) };
    } finally {
      infoLoading = null;
    }
  }

  function timeAgo(ts: number | null): string {
    if (!ts) return '—';
    const d = Date.now() / 1000 - ts;
    if (d < 60) return `${Math.floor(d)}s`;
    if (d < 3600) return `${Math.floor(d/60)}m`;
    if (d < 86400) return `${Math.floor(d/3600)}h`;
    return `${Math.floor(d/86400)}d`;
  }

  function formatSampling(s: { [k: string]: number } | undefined): string {
    if (!s) return '—';
    const order = ['temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty', 'typical_p'];
    const parts: string[] = [];
    for (const k of order) {
      if (k in s) parts.push(`${k}=${s[k]}`);
    }
    return parts.length ? parts.join(' · ') : '—';
  }

  onMount(load);
</script>

<div class="max-w-6xl space-y-5">
  <div class="flex items-center justify-between">
    <h1 class="text-2xl font-semibold">Models</h1>
    <button
      disabled={scanning}
      onclick={rescan}
      class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
    >{scanning ? t('Scanning…') : t('Rescan')}</button>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}
  {#if scanMessage}
    <div class="rounded border border-emerald-900 bg-emerald-950/30 px-4 py-2 text-sm text-emerald-200 font-mono">{scanMessage}</div>
  {/if}

  <div class="flex items-center gap-3 flex-wrap">
    <input
      bind:value={filterText}
      placeholder={t('Search: path / quant…')}
      class="flex-1 min-w-[12rem] rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm font-mono"
    />
    <select bind:value={filterFamily} class="rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm font-mono">
      <option value="">{t('All families')}</option>
      {#each families as f}
        <option value={f}>{f}</option>
      {/each}
    </select>
    <span class="text-xs text-slate-500 font-mono">
      {filtered.length} model · {totalSize.toFixed(1)} GB
    </span>
  </div>

  <div class="rounded-lg border border-slate-800 bg-slate-900/40 overflow-x-auto">
    <table class="w-full text-sm">
      <thead class="bg-slate-900/60 border-b border-slate-800 text-left text-xs uppercase tracking-wider text-slate-500">
        <tr>
          <th class="px-3 py-2 w-6"></th>
          <th class="px-3 py-2 whitespace-nowrap"><button class="uppercase tracking-wider hover:text-slate-300" onclick={() => setSort('family')}>{t('Family')}{sortArrow('family')}</button></th>
          <th class="px-2 py-2 whitespace-nowrap"><button class="uppercase tracking-wider hover:text-slate-300" onclick={() => setSort('quant')}>Quant{sortArrow('quant')}</button></th>
          <th class="px-2 py-2 text-right whitespace-nowrap"><button class="uppercase tracking-wider hover:text-slate-300" onclick={() => setSort('size_bytes')}>{t('Size')}{sortArrow('size_bytes')}</button></th>
          <th class="px-3 py-2 hidden xl:table-cell">mmproj</th>
          <th class="px-3 py-2"><button class="uppercase tracking-wider hover:text-slate-300" onclick={() => setSort('path')}>{t('Path')}{sortArrow('path')}</button></th>
          <th class="px-2 py-2 text-right whitespace-nowrap hidden xl:table-cell"><button class="uppercase tracking-wider hover:text-slate-300" onclick={() => setSort('last_used')}>{t('Last used')}{sortArrow('last_used')}</button></th>
          <th class="px-3 py-2 hidden xl:table-cell"></th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-800 font-mono text-xs">
        {#each filtered as m}
          {@const isOpen = openPath === m.path}
          {@const bundle = infoCache[m.path]}
          <tr class="group hover:bg-slate-900/60 cursor-pointer" onclick={() => toggleInfo(m.path)}>
            <td class="px-3 py-1.5 text-slate-500 select-none">{isOpen ? '▾' : '▸'}</td>
            <td class="px-3 py-1.5 text-emerald-400 whitespace-nowrap">{m.family ?? '—'}</td>
            <td class="px-2 py-1.5 text-amber-400 whitespace-nowrap">{m.quant ?? '—'}</td>
            <td class="px-2 py-1.5 text-right text-slate-300 whitespace-nowrap tabular-nums">{m.size_gb.toFixed(1)} GB</td>
            <td class="px-3 py-1.5 hidden xl:table-cell">{m.has_mmproj ? '✓' : ''}</td>
            <!-- `break-all` used to shatter a long path into ~40 lines once the
                 column got squeezed, blowing rows up to 600px tall. Ellipsize
                 instead; the full path is in the tooltip and the expanded row. -->
            <td class="px-3 py-1.5 text-slate-500 w-full max-w-0">
              <div class="truncate" title={m.path}>{m.path}</div>
            </td>
            <td class="px-2 py-1.5 text-right text-slate-500 whitespace-nowrap hidden xl:table-cell">{timeAgo(m.last_used)}</td>
            <td class="px-3 py-1.5 text-right whitespace-nowrap hidden xl:table-cell opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
              <a
                href="/presets?new_from={encodeURIComponent(m.path)}"
                onclick={(e) => e.stopPropagation()}
                class="rounded bg-emerald-800/40 border border-emerald-700 px-2 py-0.5 text-[11px] hover:bg-emerald-700/50 text-emerald-300"
                title={t('Create a ready-tuned preset from this model (pick a purpose, the rest is auto-filled)')}
              >{t('Create preset')}</a>
              <a
                href="/bench?model={encodeURIComponent(m.path)}"
                onclick={(e) => e.stopPropagation()}
                class="ml-1 rounded bg-slate-700/40 border border-slate-600 px-2 py-0.5 text-[11px] hover:bg-slate-700/60 text-slate-300"
                title={t('Benchmark this model with llama-bench')}
              >Bench</a>
              <button
                type="button"
                onclick={(e) => { e.stopPropagation(); if (!isOpen) toggleInfo(m.path); runVerify(m.path); }}
                disabled={verify[m.path]?.state === 'running'}
                class="ml-1 rounded bg-slate-700/40 border border-slate-600 px-2 py-0.5 text-[11px] hover:bg-slate-700/60 text-slate-300 disabled:opacity-50"
                title={t('Hash every part and compare against the checksum recorded at download. Catches a corrupted copy that still has the right file size.')}
              >{verify[m.path]?.state === 'running' ? t('Hashing…') : t('Verify')}</button>
            </td>
          </tr>
          {#if isOpen}
            <tr class="bg-slate-950/60">
              <td colspan="8" class="px-6 py-4">
                {#if infoLoading === m.path}
                  <div class="space-y-2"><Skeleton class="h-4 w-2/3" /><Skeleton class="h-4 w-1/2" /><Skeleton class="h-4 w-3/5" /></div>
                {:else if infoError[m.path]}
                  <div class="text-rose-300">{infoError[m.path]}</div>
                {:else if bundle}
                  <div class="space-y-4 text-[13px] font-sans">
                    {#if verifyError[m.path]}
                      <div class="rounded border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">{verifyError[m.path]}</div>
                    {:else if verify[m.path]}
                      {@const v = verify[m.path]}
                      <div class="rounded border border-slate-800 bg-slate-900/60 px-3 py-2 space-y-2">
                        <div class="flex items-baseline justify-between gap-3">
                          <span class="text-xs uppercase tracking-wider text-slate-500 font-mono">{t('Integrity')}</span>
                          <span class="text-xs {verdictClass(v.verdict)}">{verdictText(v.verdict)}</span>
                        </div>
                        {#if v.state === 'running'}
                          <div class="h-1.5 w-full rounded bg-slate-800 overflow-hidden">
                            <div class="h-full bg-cyan-500 transition-all" style="width: {v.percent}%"></div>
                          </div>
                          <div class="text-[11px] font-mono text-slate-500">{v.percent.toFixed(1)}%</div>
                        {/if}
                        <div class="space-y-0.5 font-mono text-[11px]">
                          {#each v.shards as sh (sh.path)}
                            <div class="flex justify-between gap-3">
                              <span class="text-slate-400 truncate">{sh.name}</span>
                              <span class="shrink-0 {sh.status === 'ok' ? 'text-emerald-400' : sh.status === 'corrupt' || sh.status === 'missing' ? 'text-rose-400' : 'text-slate-500'}">{sh.status}</span>
                            </div>
                          {/each}
                        </div>
                      </div>
                    {/if}
                    <!-- Row actions repeated here for touch devices (no hover)
                         and for narrow windows, where the hover column is hidden -->
                    <div class="flex flex-wrap gap-2">
                      <a
                        href="/presets?new_from={encodeURIComponent(m.path)}"
                        class="rounded bg-emerald-800/40 border border-emerald-700 px-2.5 py-1 text-xs hover:bg-emerald-700/50 text-emerald-300"
                      >{t('Create preset')}</a>
                      <a
                        href="/bench?model={encodeURIComponent(m.path)}"
                        class="rounded bg-slate-700/40 border border-slate-600 px-2.5 py-1 text-xs hover:bg-slate-700/60 text-slate-300"
                      >Bench</a>
                      <button
                        type="button"
                        onclick={() => runVerify(m.path)}
                        disabled={verify[m.path]?.state === 'running'}
                        class="rounded bg-slate-700/40 border border-slate-600 px-2.5 py-1 text-xs hover:bg-slate-700/60 text-slate-300 disabled:opacity-50"
                        title={t('Hash every part and compare against the checksum recorded at download. Catches a corrupted copy that still has the right file size.')}
                      >{verify[m.path]?.state === 'running' ? t('Hashing…') : t('Verify')}</button>
                    </div>
                    {#if bundle.info}
                      <div>
                        <h3 class="text-base font-semibold text-emerald-300">{bundle.info.family}</h3>
                        <p class="mt-1 text-slate-300">{bundle.info.summary}</p>
                      </div>
                    {/if}

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div class="rounded border border-slate-800 bg-slate-900/40 p-3">
                        <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">{t('Architecture')}</div>
                        <div class="font-mono text-xs text-slate-200">
                          {bundle.defaults.architecture ?? '—'}
                          {#if bundle.defaults.context_length}
                            · ctx {bundle.defaults.context_length.toLocaleString()}
                          {/if}
                        </div>
                        {#if bundle.defaults.quantized_by}
                          <div class="font-mono text-xs text-slate-400 mt-1">{t('quantized by:')} {bundle.defaults.quantized_by}</div>
                        {/if}
                      </div>

                      <div class="rounded border border-slate-800 bg-slate-900/40 p-3">
                        <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
                          {t('Sampling source:')} <span class="text-emerald-400">{bundle.recommended.source}</span>
                          {#if bundle.recommended.fallback_family}
                            {t('· matched:')} <span class="text-amber-400">{bundle.recommended.fallback_family}</span>
                          {/if}
                        </div>
                        {#if bundle.recommended.source === 'family-variants'}
                          <div class="space-y-1">
                            <div class="font-mono text-xs"><span class="text-slate-500">thinking:</span> <span class="text-slate-200">{formatSampling(bundle.recommended.thinking)}</span></div>
                            <div class="font-mono text-xs"><span class="text-slate-500">non-thinking:</span> <span class="text-slate-200">{formatSampling(bundle.recommended.non_thinking)}</span></div>
                          </div>
                        {:else}
                          <div class="font-mono text-xs text-slate-200">{formatSampling(bundle.recommended.thinking)}</div>
                        {/if}
                      </div>
                    </div>

                    {#if bundle.info}
                      <div>
                        <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1">{t('Prompt format')}</div>
                        <div class="prose prose-sm prose-invert max-w-none text-slate-300 whitespace-pre-line">{bundle.info.prompt_format}</div>
                      </div>

                      {#if bundle.info.behavior.length}
                        <div>
                          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1">{t('Behavior')}</div>
                          <ul class="list-disc list-inside text-slate-300 space-y-0.5">
                            {#each bundle.info.behavior as b}<li>{b}</li>{/each}
                          </ul>
                        </div>
                      {/if}

                      {#if bundle.info.deployment.length}
                        <div>
                          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1">{t('Deployment notes')}</div>
                          <ul class="list-disc list-inside text-slate-300 space-y-0.5">
                            {#each bundle.info.deployment as d}<li>{d}</li>{/each}
                          </ul>
                        </div>
                      {/if}

                      {#if bundle.info.caveats.length}
                        <div>
                          <div class="text-[11px] uppercase tracking-wider text-amber-500 mb-1">{t('Warnings')}</div>
                          <ul class="list-disc list-inside text-amber-200/80 space-y-0.5">
                            {#each bundle.info.caveats as c}<li>{c}</li>{/each}
                          </ul>
                        </div>
                      {/if}

                      {#if bundle.info.references.length}
                        <div>
                          <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1">{t('References')}</div>
                          <ul class="space-y-0.5">
                            {#each bundle.info.references as r}
                              <li><a href={r.url} target="_blank" rel="noopener" class="text-sky-400 hover:underline text-xs">{r.title}</a></li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                    {:else}
                      <div class="text-slate-500 italic text-xs">{t('No curated documentation for this family yet — sampling defaults still come from GGUF / family fallback.')}</div>
                    {/if}

                    {#if bundle.defaults.chat_template_preview}
                      <details class="text-xs">
                        <summary class="cursor-pointer text-slate-500 hover:text-slate-300">{t('Chat template preview')}</summary>
                        <pre class="mt-2 p-2 rounded bg-slate-950 border border-slate-800 text-slate-400 overflow-x-auto whitespace-pre-wrap break-all">{bundle.defaults.chat_template_preview}</pre>
                      </details>
                    {/if}
                  </div>
                {/if}
              </td>
            </tr>
          {/if}
        {:else}
          <tr><td colspan="8" class="px-3 py-8 text-center">
            <div class="space-y-3">
              <div class="text-sm text-slate-500">{t('No models found. Check the scan roots in settings.')}</div>
              <a href="/download" class="inline-block rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60">{t('Download a model →')}</a>
            </div>
          </td></tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
