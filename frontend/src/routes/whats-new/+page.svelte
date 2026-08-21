<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { marked } from 'marked';
  import {
    api,
    type FeatureAbRun,
    type FeatureCard,
    type FeatureScan,
    type Guide,
    type LlamaConfig,
    type LlamaVersion,
    type ModelEntry,
    type Settings,
    type SummaryAuthStatus
  } from '$lib/api';
  import { confirmDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';

  let cards = $state<FeatureCard[]>([]);
  let scans = $state<FeatureScan[]>([]);
  let abRuns = $state<FeatureAbRun[]>([]);
  let presets = $state<LlamaConfig[]>([]);
  let models = $state<ModelEntry[]>([]);
  let settings = $state<Settings | null>(null);
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let keyInput = $state('');
  let poll: ReturnType<typeof setInterval> | null = null;
  // Summarization backend: provider claude (api_key | env | claude_cli |
  // profile) or an OpenAI-compatible endpoint picked in Settings.
  let auth = $state<SummaryAuthStatus | null>(null);
  const authMode = $derived(auth?.mode ?? null);

  // Per-card selected preset/model (id → name/path)
  let tryPreset = $state<Record<number, string>>({});
  let abModel = $state<Record<number, string>>({});

  // Guide tab
  let view = $state<'feed' | 'guide'>('feed');
  let guide = $state<Guide | null>(null);
  let version = $state<LlamaVersion | null>(null);
  const guideStale = $derived(
    guide?.status === 'success' && version?.commit && guide.commit_sha && guide.commit_sha !== version.commit
  );
  const guideHtml = $derived(
    guide?.status === 'success' && guide.content_md ? (marked.parse(guide.content_md) as string) : ''
  );

  async function startGuide() {
    busy = 'guide';
    error = null;
    try {
      await api.featuresGuideStart();
      guide = await api.featuresGuide();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  const needsKey = $derived(authMode === 'none');
  const pendingScans = $derived(scans.filter((s) => s.status === 'pending' || s.status === 'failed'));
  const singlePresets = $derived(presets.filter((p) => (p.mode ?? 'single') === 'single'));

  // Pending scan phase: pending + no error → summary is being generated NOW;
  // pending + error → waiting for credentials/credits; failed → the call failed.
  function scanPhase(s: FeatureScan): 'generating' | 'blocked' | 'failed' {
    if (s.status === 'failed') return 'failed';
    if (s.error) return 'blocked';
    return 'generating';
  }

  // Group cards by build/scan (newest build on top; the backend already
  // returns created_at DESC). Commit range + generation time live in the
  // group header; not repeated on cards.
  type CardGroup = {
    key: string;
    from_commit: string | null;
    to_commit: string | null;
    build_number: number | null;
    created_at: number;
    cards: FeatureCard[];
  };
  const cardGroups = $derived.by(() => {
    const groups: CardGroup[] = [];
    const idx = new Map<string, CardGroup>();
    for (const c of cards) {
      const key = `${c.from_commit ?? '?'}→${c.to_commit ?? '?'}·${c.build_number ?? ''}`;
      let g = idx.get(key);
      if (!g) {
        g = {
          key,
          from_commit: c.from_commit,
          to_commit: c.to_commit,
          build_number: c.build_number,
          created_at: c.created_at,
          cards: []
        };
        idx.set(key, g);
        groups.push(g);
      }
      g.cards.push(c);
      if (c.created_at > g.created_at) g.created_at = c.created_at;
    }
    return groups;
  });

  async function refresh() {
    try {
      const [c, s, ab, g] = await Promise.all([
        api.featuresList({ limit: 100 }),
        api.featureScans(20),
        api.featureAbRuns(null, 30),
        api.featuresGuide()
      ]);
      cards = c;
      scans = s;
      abRuns = ab;
      guide = g;
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(async () => {
    refresh();
    try {
      const [st, pr, mo, au, v] = await Promise.all([
        api.getSettings(),
        api.listPresets(),
        api.listModels(),
        api.featuresAuthStatus(),
        api.buildVersion()
      ]);
      settings = st;
      presets = pr;
      models = mo;
      auth = au;
      version = v;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
    poll = setInterval(refresh, 10000);
  });
  onDestroy(() => {
    if (poll) clearInterval(poll);
  });

  async function saveKey() {
    if (!settings || !keyInput.trim()) return;
    busy = 'key';
    try {
      settings = await api.putSettings({ ...settings, anthropic_api_key: keyInput.trim() });
      keyInput = '';
      await recheckAuth();
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function recheckAuth() {
    try {
      auth = await api.featuresAuthStatus();
    } catch { /* ignore */ }
  }

  async function scanNow() {
    busy = 'scan';
    error = null;
    try {
      await api.featuresScanNow();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function retryScan(id: number) {
    busy = `retry-${id}`;
    error = null;
    try {
      await api.featureScanRetry(id);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function markSeen(id: number) {
    try {
      await api.featureSeen(id);
      cards = cards.map((c) => (c.id === id ? { ...c, seen: 1 } : c));
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function seenAll() {
    try {
      await api.featuresSeenAll();
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function tryIt(card: FeatureCard, start: boolean) {
    const preset = tryPreset[card.id];
    if (!preset) return;
    if (start) {
      const ok = await confirmDialog(
        t('Preset "{p}" will be cloned with flags "{f}" and STARTED. Mind your VRAM usage.', { p: preset, f: card.flags.join(' ') }),
        { title: t('Start feature trial?'), confirmLabel: t('Clone & start') }
      );
      if (!ok) return;
    }
    busy = `try-${card.id}`;
    error = null;
    try {
      const r = await api.featureTry(card.id, preset, start);
      await markSeen(card.id);
      await confirmDialog(
        start
          ? t('"{p}" was created and started. Watch it on the Server page.', { p: r.preset })
          : t('"{p}" was created (not started). Edit and start it on the Presets page.', { p: r.preset }),
        { title: t('Trial preset ready'), confirmLabel: t('OK'), cancelLabel: t('Close') }
      );
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  async function runAb(card: FeatureCard) {
    const model = abModel[card.id];
    if (!model) return;
    const ok = await confirmDialog(
      t('Two sequential llama-bench runs with the selected model (flags off → on). The GPU should be idle; results are unreliable while a preset is running.'),
      { title: t('Start A/B bench?'), confirmLabel: t('Start') }
    );
    if (!ok) return;
    busy = `ab-${card.id}`;
    error = null;
    try {
      await api.featureAbRun(card.id, model, { n_prompts: 512, n_gens: 128, repetitions: 2 });
      await markSeen(card.id);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = null;
    }
  }

  function cardAbRuns(id: number): FeatureAbRun[] {
    return abRuns.filter((r) => r.feature_id === id);
  }

  /** Pair off/on results by test type and produce delta rows */
  function abRows(run: FeatureAbRun): { test: string; off: number | null; on: number | null; delta: number | null }[] {
    const offMap = new Map((run.off?.results ?? []).map((r) => [String(r.test ?? ''), r.avg_ts ?? null]));
    const onMap = new Map((run.on?.results ?? []).map((r) => [String(r.test ?? ''), r.avg_ts ?? null]));
    const tests = [...new Set([...offMap.keys(), ...onMap.keys()])].filter(Boolean);
    return tests.map((t) => {
      const off = offMap.get(t) ?? null;
      const on = onMap.get(t) ?? null;
      const delta = off != null && on != null && off > 0 ? ((on - off) / off) * 100 : null;
      return { test: t, off, on, delta };
    });
  }

  /** Which of the user's local models match the card's architecture list */
  function matchedFamilies(card: FeatureCard): string[] {
    if (card.architectures.length === 0) return [];
    const fams = [...new Set(models.map((m) => m.family).filter((f): f is string => !!f))];
    return fams.filter((f) => {
      const fl = f.toLowerCase().replace(/[^a-z0-9]/g, '');
      return card.architectures.some((a) => {
        const al = a.replace(/[^a-z0-9]/g, '');
        return fl.includes(al) || al.includes(fl) || fl.startsWith(al.replace(/\d+$/, ''));
      });
    });
  }

  function fmtTs(ts: number | null): string {
    if (!ts) return '—';
    return new Date(ts * 1000).toLocaleString();
  }

  const confidenceStyle: Record<string, string> = {
    high: 'border-emerald-800 bg-emerald-950/50 text-emerald-300',
    medium: 'border-amber-800 bg-amber-950/50 text-amber-300',
    low: 'border-slate-700 bg-slate-800/50 text-slate-400'
  };
  /** The model often wraps commands in a markdown code fence. We render the
   * card as plain text, so the ``` lines would show up literally. Strip the
   * fences (keeping their content) — this fixes already-stored cards too,
   * without re-generating them. */
  function stripFences(text: string): string {
    return text
      .split('\n')
      .filter((line) => !/^\s*```/.test(line))
      .join('\n')
      .trim();
  }

  const confidenceLabel: Record<string, string> = {
    high: 'evidence: strong',
    medium: 'evidence: medium',
    low: 'evidence: weak'  // t() render'da uygulanır
  };
</script>

<div class="max-w-5xl space-y-6">
  <div class="flex items-center gap-3 flex-wrap">
    <h1 class="text-2xl font-semibold">{t("What's New")}</h1>
    <div class="flex rounded border border-slate-700 overflow-hidden text-sm">
      <button
        onclick={() => view = 'feed'}
        class="px-3 py-1 {view === 'feed' ? 'bg-emerald-800/40 text-emerald-300' : 'bg-slate-800/40 text-slate-400 hover:text-slate-200'}"
      >{t('Feed')}</button>
      <button
        onclick={() => view = 'guide'}
        class="px-3 py-1 border-l border-slate-700 {view === 'guide' ? 'bg-emerald-800/40 text-emerald-300' : 'bg-slate-800/40 text-slate-400 hover:text-slate-200'}"
      >{t('Guide')}{#if guideStale}<span class="ml-1 text-amber-400" title={t('Build changed — guide is stale')}>●</span>{/if}</button>
    </div>
    <span class="text-xs text-slate-500 font-mono">{view === 'feed' ? t("what's new in llama.cpp? what is it for? how to use it?") : t('what you can do with the current build')}</span>
    <div class="ml-auto flex gap-2">
      {#if view === 'feed'}
        {#if cards.some((c) => !c.seen)}
          <button
            onclick={seenAll}
            class="rounded bg-slate-700/40 border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-700/60"
          >{t('Mark all read')}</button>
        {/if}
        <button
          onclick={scanNow}
          disabled={busy != null}
          class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
        >{busy === 'scan' ? t('Scanning…') : t('Scan now')}</button>
      {:else}
        <button
          onclick={startGuide}
          disabled={busy != null || guide?.status === 'running'}
          class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
        >{guide?.status === 'running' ? t('Generating…') : guide?.status === 'success' ? t('Update guide') : t('Generate guide')}</button>
      {/if}
    </div>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  <!-- No usable backend: cards cannot be generated; offer the available options -->
  {#if needsKey && auth?.provider === 'openai'}
    <section class="rounded-lg border border-amber-800 bg-amber-950/20 p-4 space-y-2">
      <div class="text-sm text-amber-200">
        {t('An OpenAI-compatible provider is selected but its base URL / model are not set. Without them scans still run, but only the raw flag list is shown.')}
      </div>
      <a href="/settings" class="inline-block rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:bg-slate-700/60">
        {t('Configure in Settings → AI provider')}
      </a>
    </section>
  {:else if needsKey}
    <section class="rounded-lg border border-amber-800 bg-amber-950/20 p-4 space-y-3">
      <div class="text-sm text-amber-200">
        {t('An AI provider is needed for summary cards. Without one scans still run, but only the raw flag list is shown.')}
      </div>
      <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-1.5">
        <div class="text-xs uppercase tracking-wider text-slate-500">{t('Option 1 — Claude subscription (Pro/Max, recommended)')}</div>
        <div class="text-sm text-slate-300">
          {t('If Claude Code is installed on this machine and you are signed in, summaries are generated through it — no key or API credits needed.')}
        </div>
        <button
          onclick={recheckAuth}
          class="rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60"
        >{t('Re-check session')}</button>
      </div>
      <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-1.5">
        <div class="text-xs uppercase tracking-wider text-slate-500">{t('Option 2 — Anthropic API key')}</div>
        <div class="flex gap-2 items-center flex-wrap">
          <input
            type="password"
            bind:value={keyInput}
            placeholder="sk-ant-…"
            class="flex-1 min-w-64 rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono"
          />
          <button
            onclick={saveKey}
            disabled={busy === 'key' || !keyInput.trim()}
            class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40"
          >{busy === 'key' ? t('Saving…') : t('Save')}</button>
        </div>
        <div class="text-xs text-slate-500">{t('The key is stored in the local settings file and used only for summary generation.')}</div>
      </div>
      <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-1.5">
        <div class="text-xs uppercase tracking-wider text-slate-500">{t('Option 3 — another provider (OpenRouter, OpenAI, a local model, …)')}</div>
        <div class="text-sm text-slate-300">
          {t('Any OpenAI-compatible endpoint works — including a llama-server running on this machine.')}
        </div>
        <a href="/settings" class="inline-block rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:bg-slate-700/60">
          {t('Configure in Settings → AI provider')}
        </a>
      </div>
    </section>
  {:else if authMode === 'openai'}
    <div class="text-xs text-slate-500">
      <span class="text-emerald-400">✓</span>
      {t('Summaries are generated via {model} at {url}.', { model: auth?.model ?? '?', url: auth?.base_url ?? '?' })}
    </div>
  {:else if authMode === 'claude_cli'}
    <div class="text-xs text-slate-500">
      <span class="text-emerald-400">✓</span> {t('Summaries are generated via your Claude Code session (Pro subscription) — no key/credits needed.')}
    </div>
  {:else if authMode === 'profile'}
    <div class="text-xs text-slate-500">
      <span class="text-amber-400">!</span> {t('`ant auth login` profile found — this path needs API credits; without them summarization fails.')}
    </div>
  {/if}

  {#if view === 'guide'}
    <!-- Guide: what the current build can do (generated from live system data) -->
    {#if guideStale}
      <div class="rounded border border-amber-800 bg-amber-950/20 px-4 py-3 text-sm text-amber-200">
        {t('This guide was generated for')} <span class="font-mono">{guide?.commit_sha}</span>{t('; the current build is')}
        <span class="font-mono">{version?.commit}</span>. {t('Refresh it with "Update guide".')}
      </div>
    {/if}
    {#if !guide || guide.status === 'none'}
      <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-500 space-y-2">
        <p>{t('No guide yet. "Generate guide" collects your build\'s flags, MTP-capable architectures, model inventory and presets, and produces a guide tailored to your system (via the configured AI provider, ~1-2 minutes).')}</p>
      </section>
    {:else if guide.status === 'running'}
      <section class="rounded-lg border border-amber-800 bg-amber-950/20 p-6 text-sm text-amber-200 flex items-center gap-3">
        <span class="h-2 w-2 rounded-full bg-amber-400 animate-pulse shrink-0"></span>
        {t('Generating guide — system data collected, the model is writing… (page refreshes itself)')}
      </section>
    {:else if guide.status === 'failed'}
      <section class="rounded-lg border border-rose-800 bg-rose-950/20 p-4 text-sm text-rose-200 space-y-2">
        <div>{t('Guide generation failed:')}</div>
        <div class="font-mono text-xs">{guide.error}</div>
      </section>
    {:else}
      <div class="text-xs font-mono text-slate-500">
        build {guide.build_number} · {guide.commit_sha} · {fmtTs(guide.created_at ?? null)}
      </div>
      <article class="guide-md rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <!-- eslint-disable-next-line svelte/no-at-html-tags — content produced by our own backend -->
        {@html guideHtml}
      </article>
    {/if}
  {:else}

  <!-- Scans awaiting summary / failed -->
  {#each pendingScans as scan (scan.id)}
    {@const phase = scanPhase(scan)}
    <section class="rounded-lg border p-4 space-y-2 {phase === 'failed' ? 'border-rose-800 bg-rose-950/20' : phase === 'blocked' ? 'border-amber-800 bg-amber-950/20' : 'border-sky-800 bg-sky-950/20'}">
      <div class="flex items-center gap-2 flex-wrap text-xs font-mono {phase === 'failed' ? 'text-rose-300' : phase === 'blocked' ? 'text-amber-300' : 'text-sky-300'}">
        {#if scan.build_number}<span class="rounded bg-slate-800/60 border border-slate-700 px-2 py-0.5">build {scan.build_number}</span>{/if}
        <span>{scan.from_commit ?? '?'} → {scan.to_commit ?? '?'}</span>
        <span class="opacity-50">·</span>
        <span>{scan.new_flags.length} yeni bayrak, {scan.commits.length} commit</span>
        <span class="opacity-50">·</span>
        <span>{fmtTs(scan.created_at)}</span>
      </div>

      {#if phase === 'generating'}
        <div class="flex items-center gap-2 text-sm text-sky-200">
          <span class="h-2 w-2 rounded-full bg-sky-400 animate-pulse shrink-0"></span>
          <span><strong>{t('Summary in progress')}</strong>{t(" — the model is writing this update's cards… (may take ~2-3 min, page refreshes automatically)")}</span>
          <button
            onclick={() => retryScan(scan.id)}
            disabled={busy != null}
            class="ml-auto rounded border border-slate-700 px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 disabled:opacity-40"
            title={t('Restart if generation is stuck')}
          >{busy === `retry-${scan.id}` ? t('Summarizing…') : t('stuck? retry')}</button>
        </div>
      {:else}
        <div class="flex items-center gap-3 flex-wrap text-sm {phase === 'failed' ? 'text-rose-200' : 'text-amber-200'}">
          <span>{phase === 'failed' ? t('Summarization failed.') : t('Waiting for an AI provider to summarize.')}</span>
          <button
            onclick={() => retryScan(scan.id)}
            disabled={busy != null}
            class="ml-auto rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60 disabled:opacity-40"
          >{busy === `retry-${scan.id}` ? t('Summarizing…') : t('Retry summary')}</button>
        </div>
        {#if scan.error}
          <div class="text-xs font-mono {phase === 'failed' ? 'text-rose-300' : 'text-amber-300'}">{scan.error}</div>
        {/if}
      {/if}

      {#if scan.new_flags.length > 0}
        <details>
          <summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-200">ham bayrak listesi ({scan.new_flags.length})</summary>
          <ul class="mt-1 space-y-0.5 text-[11px] font-mono text-slate-400 max-h-48 overflow-y-auto">
            {#each scan.new_flags as f}
              <li><span class="text-amber-400">{f.flag}</span> — {f.usage}</li>
            {/each}
          </ul>
        </details>
      {/if}
    </section>
  {/each}

  <!-- Feature cards — grouped by build/scan -->
  {#each cardGroups as group (group.key)}
    <div class="flex items-center gap-3 pt-2">
      <div class="h-px flex-1 bg-slate-800"></div>
      <div class="flex items-center gap-2 flex-wrap justify-center text-xs font-mono text-slate-400">
        {#if group.build_number}<span class="rounded bg-slate-800 border border-slate-700 px-2 py-0.5 text-slate-300">build {group.build_number}</span>{/if}
        <span>{group.from_commit ?? '?'} → {group.to_commit ?? '?'}</span>
        <span class="text-slate-600">·</span>
        <span class="text-slate-300" title={t('summary generation time')}>{fmtTs(group.created_at)}</span>
        <span class="text-slate-600">·</span>
        <span>{t('{n} updates', { n: group.cards.length })}</span>
      </div>
      <div class="h-px flex-1 bg-slate-800"></div>
    </div>
    {#each group.cards as card (card.id)}
    <section class="rounded-lg border {card.seen ? 'border-slate-800' : 'border-emerald-800/60'} bg-slate-900/40 p-5 space-y-4">
      <div class="flex items-start gap-3 flex-wrap">
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2 flex-wrap">
            {#if !card.seen}
              <span class="h-2 w-2 rounded-full bg-emerald-400 shrink-0" title={t('unread')}></span>
            {/if}
            <h2 class="text-lg font-semibold text-slate-100">{card.title_tr}</h2>
          </div>
        </div>
        <span class="inline-flex rounded-full border px-2 py-0.5 text-[11px] {confidenceStyle[card.confidence]}">{t(confidenceLabel[card.confidence])}</span>
      </div>

      {#if card.flags.length > 0 || card.architectures.length > 0}
        <div class="flex gap-1.5 flex-wrap">
          {#each card.flags as f}
            <span class="inline-flex rounded-full border border-amber-800 bg-amber-950/50 px-2 py-0.5 text-[11px] font-mono text-amber-300">{f}</span>
          {/each}
          {#each card.architectures as a}
            <span class="inline-flex rounded-full border border-violet-800 bg-violet-950/50 px-2 py-0.5 text-[11px] font-mono text-violet-300">{a}</span>
          {/each}
        </div>
      {/if}

      <div class="space-y-3 text-sm text-slate-300">
        <div>
          <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">{t('What landed')}</div>
          <p>{card.what_tr}</p>
        </div>
        <div>
          <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">{t('How to use')}</div>
          <p class="whitespace-pre-wrap">{stripFences(card.how_tr)}</p>
        </div>
        <div>
          <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">{t('Why use it')}</div>
          <p>{card.why_tr}</p>
        </div>
      </div>

      {#if matchedFamilies(card).length > 0}
        <div class="text-xs text-slate-400">
          <span class="text-emerald-400">{t('You have:')}</span>
          {t('this update matches these local models:')}
          <span class="text-slate-200">{matchedFamilies(card).join(', ')}</span>
          ailelerini ilgilendiriyor.
        </div>
      {/if}

      {#if card.source_urls.length > 0}
        <div class="flex gap-3 flex-wrap text-xs">
          {#each card.source_urls as u, i}
            <a href={u} target="_blank" rel="noreferrer" class="text-sky-400 hover:underline font-mono">{t('source {n} ↗', { n: i + 1 })}</a>
          {/each}
        </div>
      {/if}

      <!-- Eylemler -->
      <div class="border-t border-slate-800 pt-3 flex items-center gap-2 flex-wrap text-sm">
        {#if card.flags.length > 0}
          <select bind:value={tryPreset[card.id]} class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs max-w-56">
            <option value={undefined} disabled selected>{t('pick a preset…')}</option>
            {#each singlePresets as p}
              <option value={p.name}>{p.name}</option>
            {/each}
          </select>
          <button
            onclick={() => tryIt(card, false)}
            disabled={busy != null || !tryPreset[card.id]}
            class="rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60 disabled:opacity-40"
            title={t('Clones the preset and adds the flags; does not start it')}
          >{t('Clone')}</button>
          <button
            onclick={() => tryIt(card, true)}
            disabled={busy != null || !tryPreset[card.id]}
            class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1 text-xs hover:bg-emerald-700/60 disabled:opacity-40"
            title={t('Clones the preset, adds the flags and starts it immediately')}
          >{busy === `try-${card.id}` ? t('Trying…') : t('Clone & start')}</button>

          <span class="mx-1 text-slate-700">|</span>

          <select bind:value={abModel[card.id]} class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs max-w-64">
            <option value={undefined} disabled selected>{t('pick a model…')}</option>
            {#each models as m}
              <option value={m.path}>{m.path.split('/').pop()} ({m.size_gb.toFixed(1)} GB)</option>
            {/each}
          </select>
          <button
            onclick={() => runAb(card)}
            disabled={busy != null || !abModel[card.id]}
            class="rounded bg-violet-700/40 border border-violet-600 px-3 py-1 text-xs hover:bg-violet-700/60 disabled:opacity-40"
            title={t('Two bench runs with flags off and on — measures the speed delta')}
          >{busy === `ab-${card.id}` ? t('Starting…') : 'A/B bench'}</button>
        {:else}
          <span class="text-xs text-slate-500 italic">{t('This update has no flags to try (informational card).')}</span>
        {/if}
        {#if !card.seen}
          <button
            onclick={() => markSeen(card.id)}
            class="ml-auto rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60"
          >{t('Mark read')}</button>
        {/if}
      </div>

      <!-- A/B results -->
      {#each cardAbRuns(card.id) as run (run.id)}
        <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-2">
          <div class="flex items-center gap-2 text-xs font-mono flex-wrap">
            <span class="text-slate-500">A/B #{run.id}</span>
            <span class="text-slate-400 truncate max-w-[300px]" title={run.model_path}>{run.model_path.split('/').pop()}</span>
            <span class={run.status === 'success' ? 'text-emerald-400' : run.status === 'failed' ? 'text-rose-400' : 'text-amber-400 animate-pulse'}>
              {run.status === 'running' ? t('running…') : run.status === 'success' ? t('done') : t('failed')}
            </span>
            {#if run.error}<span class="text-rose-300">{run.error}</span>{/if}
          </div>
          {#if run.status === 'success' && abRows(run).length > 0}
            <div class="overflow-x-auto">
              <table class="w-full text-xs font-mono">
                <thead class="text-slate-500 uppercase tracking-wider">
                  <tr>
                    <th class="py-1 text-left">test</th>
                    <th class="py-1 text-right">{t('off (t/s)')}</th>
                    <th class="py-1 text-right">{t('on (t/s)')}</th>
                    <th class="py-1 text-right">Δ%</th>
                  </tr>
                </thead>
                <tbody>
                  {#each abRows(run) as row}
                    <tr class="border-t border-slate-800/60">
                      <td class="py-1 text-slate-300">{row.test}</td>
                      <td class="py-1 text-right text-slate-300">{row.off?.toFixed(1) ?? '—'}</td>
                      <td class="py-1 text-right text-slate-300">{row.on?.toFixed(1) ?? '—'}</td>
                      <td class="py-1 text-right {row.delta == null ? 'text-slate-500' : row.delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
                        {row.delta == null ? '—' : `${row.delta >= 0 ? '+' : ''}${row.delta.toFixed(1)}%`}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      {/each}
    </section>
    {/each}
  {:else}
    {#if pendingScans.length === 0}
      <section class="rounded-lg border border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-500">
        {t('No feature cards yet. The first cards appear automatically after the next llama.cpp build; you can also trigger a scan manually with "Scan now".')}
      </section>
    {/if}
  {/each}

  {/if}
</div>

<style>
  /* Guide markdown content — matches the page's slate theme */
  .guide-md :global(h2) {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgb(52 211 153); /* emerald-400 */
    font-family: ui-monospace, monospace;
    margin: 1.6rem 0 0.6rem;
    padding-top: 1rem;
    border-top: 1px solid rgb(30 41 59); /* slate-800 */
  }
  .guide-md :global(h2:first-child) { margin-top: 0; padding-top: 0; border-top: none; }
  .guide-md :global(h3) {
    font-size: 0.9rem;
    color: rgb(203 213 225);
    font-family: ui-monospace, monospace;
    margin: 1rem 0 0.4rem;
  }
  .guide-md :global(p) { color: rgb(203 213 225); font-size: 0.9rem; line-height: 1.65; margin: 0 0 0.8rem; }
  .guide-md :global(li) { color: rgb(203 213 225); font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.35rem; }
  .guide-md :global(ul), .guide-md :global(ol) { padding-left: 1.3rem; margin: 0 0 0.9rem; }
  .guide-md :global(code) {
    font-family: ui-monospace, monospace;
    font-size: 0.8em;
    color: rgb(252 211 77); /* amber-300 */
    background: rgb(30 41 59 / 0.6);
    border: 1px solid rgb(51 65 85);
    border-radius: 3px;
    padding: 0.05em 0.35em;
  }
  .guide-md :global(strong) { color: rgb(241 245 249); }
  .guide-md :global(table) {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    font-family: ui-monospace, monospace;
    margin: 0 0 1rem;
  }
  .guide-md :global(th) {
    text-align: left;
    color: rgb(100 116 139);
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgb(30 41 59);
  }
  .guide-md :global(td) {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgb(30 41 59 / 0.6);
    color: rgb(203 213 225);
    vertical-align: top;
  }
  .guide-md :global(a) { color: rgb(56 189 248); }
  .guide-md :global(blockquote) {
    border-left: 3px solid rgb(51 65 85);
    margin: 0 0 0.9rem;
    padding: 0.2rem 0 0.2rem 0.9rem;
    color: rgb(148 163 184);
  }
</style>
