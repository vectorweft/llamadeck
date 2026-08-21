<script lang="ts">
  import '../app.css';
  import { onDestroy, onMount } from 'svelte';
  import { page } from '$app/stores';
  import ConfirmDialog from '$lib/components/ConfirmDialog.svelte';
  import Toasts from '$lib/components/Toasts.svelte';
  import CommandPalette from '$lib/components/CommandPalette.svelte';
  import { i18n, setLocale, t, type Locale } from '$lib/i18n.svelte';
  import {
    Download, Gauge, Hammer, LayoutDashboard, Boxes, Route, ScrollText,
    Server, Settings as SettingsIcon, SlidersHorizontal, Sparkles, Wand2
  } from '@lucide/svelte';

  let { children } = $props();

  // Grouped navigation: Operate = live processes, Library = artifacts on disk,
  // Toolbox = occasional maintenance tools. Group labels + item labels are
  // translated at render time via t().
  const navGroups = [
    {
      label: 'Operate',
      items: [
        { href: '/', label: 'Dashboard', icon: LayoutDashboard },
        { href: '/server', label: 'Server', icon: Server },
        { href: '/router', label: 'Router', icon: Route }
      ]
    },
    {
      label: 'Library',
      items: [
        { href: '/presets', label: 'Presets', icon: SlidersHorizontal },
        { href: '/models', label: 'Models', icon: Boxes },
        { href: '/download', label: 'Download', icon: Download }
      ]
    },
    {
      label: 'Toolbox',
      items: [
        { href: '/bench', label: 'Bench', icon: Gauge },
        { href: '/build', label: 'Build', icon: Hammer },
        { href: '/logs', label: 'Logs', icon: ScrollText }
      ]
    }
  ];
  const whatsNew = { href: '/whats-new', label: "What's New", icon: Sparkles };
  const settingsLink = { href: '/settings', label: 'Settings', icon: SettingsIcon };
  // Only shown while the install is incomplete. A permanent nav entry for a
  // one-time flow is clutter; without any entry, a user who dismissed the
  // dashboard card has no way back to /setup.
  const setupLink = { href: '/setup', label: 'Setup', icon: Wand2 };
  let setupPending = $state(false);

  function isActive(href: string, pathname: string): boolean {
    return href === '/' ? pathname === '/' : pathname.startsWith(href);
  }

  // Backend health dot: /health is polled independently of page-level
  // pollers so connectivity is visible on every route.
  let online = $state<boolean | null>(null);
  let healthTimer: ReturnType<typeof setInterval> | null = null;
  // What's New badge: count of unseen feature cards/scans. Refreshed on every
  // 6th health ping (~30 s) — no separate timer needed.
  let unseenFeatures = $state(0);
  let pingCount = 0;
  async function ping() {
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      online = r.ok;
    } catch {
      online = false;
    }
    if (pingCount++ % 6 === 0 && online) {
      try {
        const r = await fetch('/api/features/unseen-count', { cache: 'no-store' });
        if (r.ok) unseenFeatures = (await r.json()).count ?? 0;
      } catch { /* badge is non-critical */ }
      try {
        const r = await fetch('/api/setup/state', { cache: 'no-store' });
        // Only the prerequisites keep this visible. Someone who supplies their
        // own GGUFs never finishes the optional steps, and a nav entry that
        // never goes away is a permanent accusation of being half-installed.
        if (r.ok) setupPending = (await r.json()).required_done === false;
      } catch { /* nav entry is non-critical */ }
    }
  }

  type Theme = 'slate' | 'graphite';
  let theme = $state<Theme>('slate');
  function toggleTheme() {
    theme = theme === 'slate' ? 'graphite' : 'slate';
    if (theme === 'graphite') document.documentElement.dataset.theme = 'graphite';
    else delete document.documentElement.dataset.theme;
    try { localStorage.setItem('llamadeck-theme', theme); } catch { /* ignore */ }
  }

  // UI zoom for readability. We use the CSS `zoom` property (not transform:
  // scale) so sticky sidebar + fixed overlays keep positioning correctly, and
  // it scales both rem- and px-based text uniformly. Persisted per browser.
  const ZOOM_MIN = 0.8, ZOOM_MAX = 1.6, ZOOM_STEP = 0.1;
  let zoom = $state(1);
  function applyZoom() {
    // The parent document's zoom already scales the <iframe> box, so a pane
    // that zoomed itself as well would come out at zoom².
    if (paneMode) return;
    // `zoom` isn't in the typed CSSStyleDeclaration everywhere; set via style.
    document.documentElement.style.setProperty('zoom', String(zoom));
    // Mirror it into --zoom so `--vh` (see app.css) can divide it back out.
    document.documentElement.style.setProperty('--zoom', String(zoom));
    measureLayout();
  }
  function setZoom(z: number) {
    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(z * 100) / 100));
    applyZoom();
    try { localStorage.setItem('llamadeck-zoom', String(zoom)); } catch { /* ignore */ }
  }
  // Tailwind's `lg:` breakpoint measures the raw viewport, but the CSS `zoom`
  // shrinks how many CSS pixels the layout actually gets: at zoom 1.6 a 1280px
  // screen still counts as "large" while only ~800px are usable, and the 208px
  // rail eats 39% of them. <body> sits inside the zoom, so its client width is
  // already the space we really have.
  //
  // Three signals feed it, because none covers everything: `resize` misses
  // zoom changes (the viewport did not move, only what fits inside it),
  // applyZoom misses window drags, and a ResizeObserver — which would cover
  // both — only delivers on a rendered frame, so it stays silent while the
  // tab is not compositing.
  let layoutW = $state(1280);
  const wideRail = $derived(layoutW >= 1024);
  let railObserver: ResizeObserver | undefined;
  function measureLayout() { layoutW = document.body.clientWidth; }

  // ---- Split view -------------------------------------------------------
  // A wide monitor leaves the left-aligned content with a screenful of dead
  // space beside it. Centring the column would only move the gap, so the gap
  // holds a second page instead. That page is this same app loaded with
  // ?pane=1: it drops the rail and reads theme / language / zoom from the same
  // localStorage keys, so opening one cannot leave the two halves on different
  // settings. `storage` events keep them in step while it is open.
  const SPLIT_MIN_W = 1400;
  let splitHref = $state<string | null>(null);
  let splitRatio = $state(0.5);
  const paneMode = $derived($page.url.searchParams.has('pane'));
  const canSplit = $derived(!paneMode && layoutW >= SPLIT_MIN_W);
  const splitOn = $derived(canSplit && !!splitHref);
  const splitPages = [...navGroups.flatMap(g => g.items), whatsNew, settingsLink];
  const splitLabel = $derived(splitPages.find(p => p.href === splitHref)?.label ?? '');
  const paneSrc = $derived(splitHref ? `${splitHref}${splitHref.includes('?') ? '&' : '?'}pane=1` : '');

  function setSplit(href: string) {
    splitHref = href || null;
    try { localStorage.setItem('llamadeck-split', href); } catch { /* ignore */ }
  }

  let mainEl: HTMLElement | undefined;
  let shellEl: HTMLElement | undefined;
  function startResize(e: PointerEvent) {
    e.preventDefault();
    const move = (ev: PointerEvent) => {
      if (!mainEl || !shellEl) return;
      const left = mainEl.getBoundingClientRect().left;
      const right = shellEl.getBoundingClientRect().right;
      if (right <= left) return;
      splitRatio = Math.min(0.75, Math.max(0.25, (ev.clientX - left) / (right - left)));
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      try { localStorage.setItem('llamadeck-split-ratio', String(splitRatio)); } catch { /* ignore */ }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  // Fires only in the *other* document, so this cannot loop back on itself.
  function onStorage(e: StorageEvent) {
    if (e.key === 'llamadeck-theme') {
      theme = e.newValue === 'graphite' ? 'graphite' : 'slate';
      if (theme === 'graphite') document.documentElement.dataset.theme = 'graphite';
      else delete document.documentElement.dataset.theme;
    } else if (e.key === 'llamadeck-lang') {
      const l: Locale = e.newValue === 'tr' ? 'tr' : 'en';
      i18n.locale = l;
      document.documentElement.lang = l;
    } else if (e.key === 'llamadeck-zoom') {
      const z = parseFloat(e.newValue ?? '1');
      if (Number.isFinite(z)) { zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z)); applyZoom(); }
    }
  }

  const zoomIn = () => setZoom(zoom + ZOOM_STEP);
  const zoomOut = () => setZoom(zoom - ZOOM_STEP);
  const zoomReset = () => setZoom(1);

  function onKeydown(e: KeyboardEvent) {
    // Override the browser's native zoom with our in-app zoom so px-based
    // sizes scale too. Ctrl/Cmd + '=' / '+', '-', and '0'.
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    if (e.key === '=' || e.key === '+') { e.preventDefault(); zoomIn(); }
    else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomOut(); }
    else if (e.key === '0') { e.preventDefault(); zoomReset(); }
  }

  onMount(() => {
    theme = document.documentElement.dataset.theme === 'graphite' ? 'graphite' : 'slate';
    try {
      const z = parseFloat(localStorage.getItem('llamadeck-zoom') ?? '1');
      if (Number.isFinite(z)) zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
    } catch { /* ignore */ }
    applyZoom();
    try {
      splitHref = localStorage.getItem('llamadeck-split') || null;
      const r = parseFloat(localStorage.getItem('llamadeck-split-ratio') ?? '0.5');
      if (Number.isFinite(r)) splitRatio = Math.min(0.75, Math.max(0.25, r));
    } catch { /* ignore */ }
    measureLayout();
    railObserver = new ResizeObserver(measureLayout);
    railObserver.observe(document.body);
    window.addEventListener('resize', measureLayout);
    window.addEventListener('keydown', onKeydown);
    window.addEventListener('storage', onStorage);
    ping();
    healthTimer = setInterval(ping, 5000);
  });
  onDestroy(() => {
    if (healthTimer) clearInterval(healthTimer);
    railObserver?.disconnect();
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', onKeydown);
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('resize', measureLayout);
    }
  });
</script>

{#snippet navLink(item: { href: string; label: string; icon: typeof Server }, badge: number = 0)}
  {@const active = isActive(item.href, $page.url.pathname)}
  {@const Icon = item.icon}
  <a
    href={item.href}
    aria-current={active ? 'page' : undefined}
    title={t(item.label)}
    class="flex items-center gap-2.5 rounded px-2.5 py-1.5 text-sm transition-colors
           {active
             ? 'bg-slate-800/70 text-emerald-400 font-medium'
             : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'}"
  >
    <Icon size={16} strokeWidth={2} class="shrink-0" />
    <span class="{wideRail ? 'inline' : 'hidden'} truncate">{t(item.label)}</span>
    {#if badge > 0}
      <span
        class="ml-auto {wideRail ? 'inline-flex' : 'hidden'} items-center rounded-full border border-amber-800 bg-amber-950/50 px-1.5 text-[10px] font-mono text-amber-300"
        title={t('{n} unseen updates', { n: badge })}
      >{badge}</span>
    {/if}
  </a>
{/snippet}

<div class="min-h-full flex" bind:this={shellEl}>
  {#if !paneMode}
  <!-- The CSS `zoom` on <html> multiplies lengths, so a plain 100vh rail ends up
       `zoom` times taller than the screen and its footer falls off the bottom.
       Divide it back out, then let the nav scroll so the controls stay pinned. -->
  <aside
    class="app-rail sticky top-0 h-[calc(100*var(--vh))] {wideRail ? 'w-52' : 'w-14'} shrink-0 flex flex-col border-r border-slate-800 bg-slate-900/50 px-2 py-3 overflow-hidden"
  >
    <a href="/" class="shrink-0 flex items-center gap-2 px-2 pb-3 font-mono text-sm tracking-wider text-slate-400" title="LlamaDeck">
      <span class="text-base leading-none">🦙</span>
      <span class="{wideRail ? 'inline' : 'hidden'} text-emerald-400 font-semibold">LlamaDeck</span>
    </a>

    <nav class="flex-1 min-h-0 overflow-y-auto space-y-4">
      {#each navGroups as group}
        <div class="space-y-0.5">
          <div class="{wideRail ? 'block' : 'hidden'} px-2.5 pb-1 text-[11px] font-mono uppercase tracking-wider text-slate-600">{t(group.label)}</div>
          {#each group.items as item}
            {@render navLink(item)}
          {/each}
        </div>
      {/each}
      <div class="space-y-0.5 border-t border-slate-800 pt-3">
        {#if setupPending}
          {@render navLink(setupLink)}
        {/if}
        {@render navLink(whatsNew, unseenFeatures)}
        {@render navLink(settingsLink)}
      </div>
    </nav>

    <div class="shrink-0 mt-3 space-y-1.5 border-t border-slate-800 pt-3">
      <div class="{wideRail ? 'flex' : 'hidden'} items-center gap-1.5 px-2.5 text-[10px] font-mono text-slate-600" title={t('Command palette')}>
        <kbd class="rounded border border-slate-700 bg-slate-800/60 px-1">Ctrl</kbd><kbd class="rounded border border-slate-700 bg-slate-800/60 px-1">K</kbd>
        <span>{t('commands')}</span>
      </div>
      <div
        class="flex items-center gap-1.5 px-2.5 text-[11px] font-mono text-slate-500"
        title={online === null ? t('Checking backend status…') : online ? t('Backend reachable') : t('Backend unreachable')}
      >
        <span class="h-2 w-2 shrink-0 rounded-full {online === null ? 'bg-slate-600' : online ? 'bg-emerald-500' : 'bg-rose-500 animate-pulse'}"></span>
        <span class="{wideRail ? 'inline' : 'hidden'}">backend</span>
      </div>
      <button
        onclick={() => setLocale(i18n.locale === 'en' ? 'tr' : 'en')}
        title={t('Switch language (English ↔ Türkçe)')}
        aria-label={t('Switch language (English ↔ Türkçe)')}
        class="w-full rounded border border-slate-700 bg-slate-800/60 hover:bg-slate-700/70 px-2 py-1 text-[11px] font-mono text-slate-300 uppercase text-left"
      >{#if wideRail}{t('language:')} {i18n.locale}{:else}{i18n.locale}{/if}</button>
      <button
        onclick={toggleTheme}
        title={t('Switch theme (Slate ↔ Graphite)')}
        aria-label={t('Switch theme (Slate ↔ Graphite)')}
        class="w-full rounded border border-slate-700 bg-slate-800/60 hover:bg-slate-700/70 px-2 py-1 text-[11px] font-mono text-slate-300 text-left"
      >{#if wideRail}{t('theme:')} {theme}{:else}{theme === 'slate' ? '◐' : '◑'}{/if}</button>
      {#if canSplit}
        <select
          value={splitHref ?? ''}
          onchange={(e) => setSplit(e.currentTarget.value)}
          title={t('Open a second page beside this one')}
          class="w-full rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 px-2 py-1 text-[11px] font-mono text-slate-300"
        >
          <option value="">{t('split: off')}</option>
          {#each splitPages as p}
            <option value={p.href}>{t('split:')} {t(p.label)}</option>
          {/each}
        </select>
      {/if}
      <div
        class="flex items-stretch rounded border border-slate-700 bg-slate-800/60 overflow-hidden text-[11px] font-mono text-slate-300"
        title={t('Zoom the interface (Ctrl +/− , Ctrl 0 to reset)')}
      >
        <button
          onclick={zoomOut}
          disabled={zoom <= ZOOM_MIN}
          aria-label={t('Zoom out')}
          class="px-2 py-1 hover:bg-slate-700/70 disabled:opacity-40 disabled:hover:bg-transparent"
        >A−</button>
        <button
          onclick={zoomReset}
          aria-label={t('Reset zoom')}
          class="flex-1 min-w-0 border-x border-slate-700 px-1 py-1 text-center hover:bg-slate-700/70 tabular-nums"
        >{#if wideRail}{t('zoom:')} {Math.round(zoom * 100)}%{:else}{Math.round(zoom * 100)}{/if}</button>
        <button
          onclick={zoomIn}
          disabled={zoom >= ZOOM_MAX}
          aria-label={t('Zoom in')}
          class="px-2 py-1 hover:bg-slate-700/70 disabled:opacity-40 disabled:hover:bg-transparent"
        >A+</button>
      </div>
    </div>
  </aside>
  {/if}

  <!-- In split mode each half scrolls on its own, so the shell stops being one
       tall page and becomes two viewport-height columns. -->
  <main
    bind:this={mainEl}
    class="min-w-0 px-6 py-6 {splitOn ? 'h-[calc(100*var(--vh))] overflow-y-auto' : 'flex-1'}"
    style={splitOn ? `flex: 0 0 ${(splitRatio * 100).toFixed(2)}%` : ''}
  >
    {@render children()}
  </main>

  {#if splitOn}
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={t('Resize the split')}
      title={t('Resize the split')}
      onpointerdown={startResize}
      class="w-1.5 shrink-0 cursor-col-resize bg-slate-800 hover:bg-emerald-600/70 transition-colors"
    ></div>
    <section class="flex-1 min-w-0 h-[calc(100*var(--vh))] flex flex-col border-l border-slate-800">
      <div class="flex shrink-0 items-center gap-2 border-b border-slate-800 bg-slate-900/50 px-3 py-1.5 text-[11px] font-mono text-slate-400">
        <span class="truncate">{t(splitLabel)}</span>
        <a
          href={splitHref}
          class="ml-auto rounded border border-slate-700 px-1.5 hover:bg-slate-800 hover:text-slate-200"
          title={t('Open in the main pane')}
        >⤢</a>
        <button
          onclick={() => setSplit('')}
          class="rounded border border-slate-700 px-1.5 hover:bg-slate-800 hover:text-slate-200"
          title={t('Close the second pane')}
          aria-label={t('Close the second pane')}
        >✕</button>
      </div>
      <iframe src={paneSrc} title={t(splitLabel)} class="flex-1 w-full border-0"></iframe>
    </section>
  {/if}
</div>

<ConfirmDialog />
<Toasts />
{#if !paneMode}<CommandPalette />{/if}
