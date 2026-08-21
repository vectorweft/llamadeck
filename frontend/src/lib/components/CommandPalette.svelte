<script lang="ts">
  import { goto } from '$app/navigation';
  import { api, type PresetStatus } from '$lib/api';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';

  interface Command {
    id: string;
    label: string;      // already-translated display text
    hint: string;       // small right-aligned kind hint
    run: () => void | Promise<void>;
  }

  let open = $state(false);
  let query = $state('');
  let selected = $state(0);
  let presetCmds = $state<Command[]>([]);
  let inputEl = $state<HTMLInputElement | null>(null);

  const PAGES = [
    { href: '/', label: 'Dashboard' },
    { href: '/server', label: 'Server' },
    { href: '/router', label: 'Router' },
    { href: '/presets', label: 'Presets' },
    { href: '/models', label: 'Models' },
    { href: '/download', label: 'Download' },
    { href: '/bench', label: 'Bench' },
    { href: '/build', label: 'Build' },
    { href: '/logs', label: 'Logs' },
    { href: '/whats-new', label: "What's New" },
    { href: '/settings', label: 'Settings' }
  ];

  function baseCommands(): Command[] {
    return PAGES.map((p) => ({
      id: `go:${p.href}`,
      label: t('Go to {page}', { page: t(p.label) }),
      hint: t('page'),
      run: () => goto(p.href)
    }));
  }

  async function loadPresetCommands() {
    try {
      const r = await api.serverStatuses();
      const out: Command[] = [];
      for (const s of Object.values(r.presets) as PresetStatus[]) {
        if (s.running) {
          out.push({
            id: `stop:${s.name}`,
            label: t('Stop {name}', { name: s.name }),
            hint: 'preset',
            run: async () => {
              await api.serverStop(s.name);
              toast(t('Stopped {name}', { name: s.name }), 'success');
            }
          });
        } else {
          out.push({
            id: `start:${s.name}`,
            label: t('Start {name}', { name: s.name }),
            hint: 'preset',
            run: async () => {
              await api.serverStart(s.name);
              toast(t('Started {name}', { name: s.name }), 'success');
            }
          });
        }
      }
      presetCmds = out;
    } catch {
      presetCmds = [];
    }
  }

  const commands = $derived.by(() => {
    const all = [...baseCommands(), ...presetCmds];
    const needle = query.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((c) => c.label.toLowerCase().includes(needle));
  });

  function show() {
    open = true;
    query = '';
    selected = 0;
    loadPresetCommands();
    // Focus after the element renders.
    setTimeout(() => inputEl?.focus(), 0);
  }

  function hide() {
    open = false;
  }

  async function runSelected() {
    const cmd = commands[selected];
    if (!cmd) return;
    hide();
    try {
      await cmd.run();
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'error');
    }
  }

  function onWindowKeydown(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (open) hide(); else show();
      return;
    }
    if (!open) return;
    if (e.key === 'Escape') { e.preventDefault(); hide(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); selected = Math.min(selected + 1, commands.length - 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); selected = Math.max(selected - 1, 0); }
    else if (e.key === 'Enter') { e.preventDefault(); runSelected(); }
  }
</script>

<svelte:window onkeydown={onWindowKeydown} />

{#if open}
  <div
    class="fixed inset-0 z-[55] bg-black/60 flex items-start justify-center pt-[calc(15*var(--vh))] px-4"
    role="presentation"
    onclick={(e) => { if (e.target === e.currentTarget) hide(); }}
  >
    <div class="w-full max-w-lg rounded-lg border border-slate-700 bg-slate-900 shadow-2xl overflow-hidden" role="dialog" aria-modal="true" aria-label={t('Command palette')}>
      <input
        bind:this={inputEl}
        bind:value={query}
        oninput={() => selected = 0}
        placeholder={t('Type a command or page name…')}
        class="w-full bg-transparent border-b border-slate-800 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none"
      />
      <ul class="max-h-72 overflow-y-auto py-1">
        {#each commands as cmd, i (cmd.id)}
          <li>
            <button
              onclick={runSelected}
              onmouseenter={() => selected = i}
              class="flex w-full items-center gap-3 px-4 py-2 text-left text-sm
                     {i === selected ? 'bg-slate-800/80 text-emerald-300' : 'text-slate-300'}"
            >
              <span class="min-w-0 flex-1 truncate">{cmd.label}</span>
              <span class="shrink-0 text-[10px] font-mono uppercase text-slate-600">{cmd.hint}</span>
            </button>
          </li>
        {:else}
          <li class="px-4 py-3 text-sm text-slate-500">{t('No matching command.')}</li>
        {/each}
      </ul>
      <div class="border-t border-slate-800 px-4 py-1.5 text-[10px] font-mono text-slate-600 flex gap-3">
        <span>↑↓ {t('navigate')}</span><span>↵ {t('run')}</span><span>esc {t('close')}</span>
      </div>
    </div>
  </div>
{/if}
