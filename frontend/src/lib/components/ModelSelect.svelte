<script lang="ts">
  import type { ModelEntry } from '$lib/api';
  import { t } from '$lib/i18n.svelte';

  let {
    models,
    value = $bindable(''),
    disabled = false,
    placeholder = undefined,
  }: {
    models: ModelEntry[];
    value?: string;
    disabled?: boolean;
    placeholder?: string;
  } = $props();

  let open = $state(false);
  let query = $state('');
  let highlight = $state(0);
  let root: HTMLDivElement | null = $state(null);
  let inputEl: HTMLInputElement | null = $state(null);

  const selected = $derived(models.find(m => m.path === value) ?? null);

  function label(m: ModelEntry): string {
    const file = m.path.split('/').pop() ?? m.path;
    return `${m.family ?? '—'} · ${m.quant ?? '—'} · ${m.size_gb.toFixed(1)} GB · ${file}`;
  }

  // Filter by any of family / quant / filename / full path (case-insensitive,
  // all whitespace-separated terms must match somewhere — lets you type
  // "qwen q4 4b" and narrow across fields).
  const filtered = $derived.by(() => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return models;
    return models.filter(m => {
      const hay = `${m.family ?? ''} ${m.quant ?? ''} ${m.path}`.toLowerCase();
      return terms.every(term => hay.includes(term));
    });
  });

  function openMenu() {
    if (disabled) return;
    open = true;
    query = '';
    highlight = Math.max(0, filtered.findIndex(m => m.path === value));
    queueMicrotask(() => inputEl?.focus());
  }
  function closeMenu() {
    open = false;
    query = '';
  }
  function choose(m: ModelEntry) {
    value = m.path;
    closeMenu();
  }

  function onKeydown(e: KeyboardEvent) {
    if (!open) {
      if (e.key === 'Enter' || e.key === 'ArrowDown' || e.key === ' ') { e.preventDefault(); openMenu(); }
      return;
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); highlight = Math.min(filtered.length - 1, highlight + 1); scrollToHighlight(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); highlight = Math.max(0, highlight - 1); scrollToHighlight(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (filtered[highlight]) choose(filtered[highlight]); }
    else if (e.key === 'Escape') { e.preventDefault(); closeMenu(); }
  }

  let listEl: HTMLDivElement | null = $state(null);
  function scrollToHighlight() {
    queueMicrotask(() => {
      const node = listEl?.children[highlight] as HTMLElement | undefined;
      node?.scrollIntoView({ block: 'nearest' });
    });
  }

  // Reset highlight to the top whenever the query changes the result set.
  $effect(() => { query; highlight = 0; });

  function onDocClick(e: MouseEvent) {
    if (open && root && !root.contains(e.target as Node)) closeMenu();
  }
  $effect(() => {
    if (open) {
      document.addEventListener('mousedown', onDocClick);
      return () => document.removeEventListener('mousedown', onDocClick);
    }
  });
</script>

<div bind:this={root} class="relative">
  <button
    type="button"
    {disabled}
    onclick={() => (open ? closeMenu() : openMenu())}
    onkeydown={onKeydown}
    class="w-full flex items-center justify-between gap-2 rounded bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm font-mono text-left disabled:opacity-50 {open ? 'border-emerald-700' : 'hover:border-slate-600'}"
    aria-haspopup="listbox"
    aria-expanded={open}
  >
    <span class="truncate {selected ? 'text-slate-200' : 'text-slate-500'}">{selected ? label(selected) : (placeholder ?? t('— pick a model —'))}</span>
    <span class="shrink-0 text-slate-500 text-xs">▾</span>
  </button>

  {#if open}
    <div class="absolute z-30 mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 shadow-2xl">
      <div class="p-2 border-b border-slate-800">
        <input
          bind:this={inputEl}
          bind:value={query}
          onkeydown={onKeydown}
          placeholder={t('Filter models…')}
          class="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm font-mono focus:border-emerald-700 focus:outline-none"
        />
      </div>
      <div bind:this={listEl} class="max-h-72 overflow-y-auto py-1" role="listbox" tabindex="-1">
        {#each filtered as m, i (m.path)}
          <button
            type="button"
            role="option"
            aria-selected={m.path === value}
            onclick={() => choose(m)}
            onmousemove={() => (highlight = i)}
            class="w-full flex items-center gap-2 px-3 py-1.5 text-left text-xs font-mono
                   {i === highlight ? 'bg-slate-800' : ''}
                   {m.path === value ? 'text-emerald-300' : 'text-slate-300'}"
          >
            <span class="shrink-0 w-3 text-emerald-400">{m.path === value ? '✓' : ''}</span>
            <span class="min-w-0 flex-1 truncate">
              <span class="text-slate-200">{m.family ?? '—'}</span>
              <span class="text-slate-500"> · {m.quant ?? '—'} · {(m.path.split('/').pop() ?? '')}</span>
            </span>
            <span class="shrink-0 text-amber-500/80">{m.size_gb.toFixed(1)} GB</span>
          </button>
        {:else}
          <div class="px-3 py-2 text-xs font-mono text-slate-500">{t('No models match.')}</div>
        {/each}
      </div>
    </div>
  {/if}
</div>
