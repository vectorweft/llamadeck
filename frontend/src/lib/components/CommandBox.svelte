<script lang="ts">
  /**
   * The last word on a preset: the exact `llama-server …` command line.
   *
   * Three states, one idea — what you read here is what runs:
   *
   *   preview  the command rendered from the fields, read-only
   *   editing  you typed in it; Apply folds it back INTO the fields
   *   locked   `argv_override` is set, so this text IS the process and every
   *            field above is just LlamaDeck's reading of it
   *
   * Both directions go through `lld.argv` on the backend (the same module the
   * supervisor executes), so the box can never drift from the real command.
   */
  import { api, type CommandParse, type CommandPreview, type FlagSpec, type LlamaConfig } from '$lib/api';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';

  let {
    config,
    onapply,
  }: {
    /** The preset currently in the editor. */
    config: LlamaConfig;
    /** Replace the edited preset wholesale (fields parsed out of a command). */
    onapply: (cfg: LlamaConfig) => void;
  } = $props();

  const locked = $derived(!!(config.argv_override ?? '').trim());

  let preview = $state<CommandPreview | null>(null);
  let previewError = $state<string | null>(null);
  let draft = $state('');          // what the textarea holds while editing
  let editing = $state(false);
  let parsed = $state<CommandParse | null>(null);
  let parseError = $state<string | null>(null);
  let busy = $state(false);

  let previewTimer: ReturnType<typeof setTimeout> | null = null;
  let previewAbort: AbortController | null = null;
  let previewSeq = 0;

  // Re-render whenever anything in the preset changes. Debounced: the editor
  // fires on every keystroke in every field.
  $effect(() => {
    const snapshot = JSON.stringify(config);
    if (previewTimer) clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      previewAbort?.abort();
      previewAbort = new AbortController();
      const mySeq = ++previewSeq;
      try {
        const r = await api.presetCommand(JSON.parse(snapshot), previewAbort.signal);
        if (mySeq !== previewSeq) return;
        preview = r;
        previewError = null;
      } catch (e) {
        if (mySeq !== previewSeq) return;
        previewError = e instanceof Error ? e.message : String(e);
      }
    }, 250);
  });

  /** The text on screen: the raw override when locked, else the render. */
  const shown = $derived(locked ? (config.argv_override ?? '') : (preview?.command ?? ''));

  let parseTimer: ReturnType<typeof setTimeout> | null = null;

  /** Parse whatever is in the textarea, so the diff/warnings stay live. */
  function scheduleParse(text: string) {
    if (parseTimer) clearTimeout(parseTimer);
    parseTimer = setTimeout(async () => {
      try {
        parsed = await api.presetCommandParse(text, { ...config, argv_override: null });
        parseError = null;
      } catch (e) {
        parsed = null;
        parseError = e instanceof Error ? e.message : String(e);
      }
    }, 300);
  }

  // A locked preset is not "being edited", but its flags still deserve the
  // unknown-flag check — otherwise the warning only appears once you type.
  $effect(() => {
    if (locked && !editing && parsed === null) scheduleParse(config.argv_override ?? '');
  });

  function startEditing() {
    draft = shown;
    parsed = null;
    parseError = null;
    editing = true;
    scheduleParse(draft);
  }

  function onDraftInput(e: Event) {
    draft = (e.currentTarget as HTMLTextAreaElement).value;
    if (locked) {
      // Locked: the text IS the preset. Persist every keystroke; the fields
      // are re-read from it on save.
      onapply({ ...config, argv_override: draft });
    }
    scheduleParse(draft);
  }

  /** Fold the typed command back into the form and go on as normal. */
  async function applyToFields() {
    busy = true;
    try {
      const r = await api.presetCommandParse(draft, { ...config, argv_override: null });
      onapply({ ...r.config, argv_override: null });
      editing = false;
      parsed = null;
      toast(
        r.diff.length
          ? t('{n} field(s) updated from the command.', { n: r.diff.length })
          : t('The command already matched the form — nothing changed.'),
        'success'
      );
    } catch (e) {
      parseError = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  /** Run this text verbatim from now on. */
  function lock() {
    onapply({ ...config, argv_override: draft || shown });
    editing = false;
    toast(t('This command now runs as written.'), 'success');
  }

  /** Stop overriding: fold the command into the fields and let them render again. */
  async function unlock() {
    busy = true;
    try {
      const r = await api.presetCommandParse(config.argv_override ?? '', { ...config, argv_override: null });
      onapply({ ...r.config, argv_override: null });
      editing = false;
      toast(t('Back to the form — the command is now rendered from the fields.'), 'success');
    } catch (e) {
      parseError = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(
        (locked ? config.argv_override : preview?.command) ?? ''
      );
      toast(t('Command copied.'), 'success');
    } catch {
      toast(t('Could not reach the clipboard — select the text and copy it.'), 'error');
    }
  }

  // --- flag reference -------------------------------------------------------
  // The binary's own --help, searchable, so looking up a flag does not mean
  // leaving the app for a terminal. Loaded on first open: probing the binary
  // costs a subprocess, and most edits never need it.
  let flagList = $state<FlagSpec[] | null>(null);
  let flagsAvailable = $state(true);
  let flagsLoading = $state(false);
  let flagFilter = $state('');

  async function loadFlags() {
    if (flagList !== null || flagsLoading) return;
    flagsLoading = true;
    try {
      const r = await api.serverFlags();
      flagList = r.flags;
      flagsAvailable = r.available;
    } catch {
      flagList = [];
      flagsAvailable = false;
    } finally {
      flagsLoading = false;
    }
  }

  const flagMatches = $derived.by<FlagSpec[]>(() => {
    const all = flagList ?? [];
    const q = flagFilter.trim().toLowerCase();
    const hits = q
      ? all.filter(f => f.names.some(n => n.includes(q)) || f.help.toLowerCase().includes(q))
      : all;
    return hits.slice(0, FLAG_ROWS);
  });
  /** How many rows of the flag list are rendered at once. The binary
   *  advertises ~250; showing a silent prefix of them made the list read as
   *  the complete set, and a flag sitting at #194 (--tools) looked like one
   *  this build does not have. The count below the list says otherwise. */
  const FLAG_ROWS = 60;
  const flagTotal = $derived.by<number>(() => {
    const all = flagList ?? [];
    const q = flagFilter.trim().toLowerCase();
    return q
      ? all.filter(f => f.names.some(n => n.includes(q)) || f.help.toLowerCase().includes(q)).length
      : all.length;
  });

  const unknown = $derived(
    editing || locked ? (parsed?.unknown_flags ?? []) : (preview?.unknown_flags ?? [])
  );
  // A flag with no value is fatal in both modes: llama-server rejects the
  // whole command line, so the preset dies at spawn with nothing on screen.
  const missingValues = $derived(
    editing || locked ? (parsed?.missing_values ?? []) : (preview?.missing_values ?? [])
  );
  // Only meaningful for the field-rendered command: in raw mode the user IS
  // the one repeating a flag, and they may well mean to.
  const shadowed = $derived(locked ? [] : (preview?.shadowed ?? []));
  // Conflicts DO apply in raw mode: two flags cancelling out is exactly the
  // kind of thing a hand-written command line gets wrong.
  const conflicts = $derived(
    editing || locked ? (parsed?.conflicts ?? []) : (preview?.conflicts ?? [])
  );

  /** Collapse duplicate flags by parsing the rendered command back in: the
   *  winning value lands in its field and extra_flags keeps only what has no
   *  field. Same operation as Apply, run against the rendered command. */
  async function fold() {
    busy = true;
    try {
      const r = await api.presetCommandParse(preview?.command ?? '', { ...config, argv_override: null });
      onapply({ ...r.config, argv_override: null });
      toast(t('Folded into the fields — the duplicates are gone.'), 'success');
    } catch (e) {
      parseError = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
  const fmt = (v: unknown) =>
    v === null || v === undefined ? '—' : Array.isArray(v) ? (v.join(' ') || '—') : String(v);
</script>

<div class="space-y-3">
  <div class="flex items-start justify-between gap-3 flex-wrap">
    <div>
      <div class="text-sm text-slate-200">{t('The command that runs')}</div>
      <div class="text-[11px] text-slate-500 mt-0.5">
        {locked
          ? t('This text is executed verbatim. Every field in the other tabs is now just a reading of it — LlamaDeck re-parses the command on save so it still knows the port, the model and the context size.')
          : t('Rendered from the fields in the other tabs, exactly as the supervisor will run it. Edit it here to work the other way round: type flags, apply, and the fields follow.')}
      </div>
    </div>
    <div class="flex items-center gap-2 shrink-0">
      <button type="button" onclick={copy}
        class="rounded bg-slate-800 border border-slate-700 px-2.5 py-1 text-[11px] hover:bg-slate-700">{t('Copy')}</button>
      {#if locked}
        <button type="button" onclick={unlock} disabled={busy}
          class="rounded bg-amber-700/40 border border-amber-600 px-2.5 py-1 text-[11px] hover:bg-amber-700/60 disabled:opacity-40">{t('Unlock → back to fields')}</button>
      {:else if editing}
        <button type="button" onclick={() => { editing = false; parsed = null; parseError = null; }}
          class="rounded bg-slate-800 border border-slate-700 px-2.5 py-1 text-[11px] hover:bg-slate-700">{t('Cancel')}</button>
      {:else}
        <button type="button" onclick={startEditing}
          class="rounded bg-slate-800 border border-slate-700 px-2.5 py-1 text-[11px] hover:bg-slate-700">{t('Edit')}</button>
      {/if}
    </div>
  </div>

  {#if locked}
    <div class="rounded border border-amber-800 bg-amber-950/20 px-3 py-2 text-[11px] text-amber-200">
      {t('Raw command mode. Flags typed here win over every setting in the editor — including ones LlamaDeck has no field for.')}
      {#if config.mode === 'router'}
        <div class="mt-1 text-amber-300">
          {t('Router mode: this applies to the router process itself. Per-model settings still come from the generated INI, not from this command.')}
        </div>
      {/if}
    </div>
  {/if}

  {#if editing || locked}
    <textarea
      value={editing ? draft : (config.argv_override ?? '')}
      oninput={onDraftInput}
      spellcheck="false"
      rows="16"
      class="w-full rounded bg-slate-950 border {locked ? 'border-amber-800' : 'border-slate-700'} px-3 py-2 font-mono text-[12px] leading-relaxed text-slate-200 resize-y"
    ></textarea>
    <div class="text-[11px] text-slate-500">
      {t('Line breaks, backslash continuations and # comments are fine. The program name is replaced with the binary from Settings, so a rebuild never leaves a preset pointing at the old one.')}
    </div>
  {:else if previewError}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200 font-mono">{previewError}</div>
  {:else if preview}
    <pre class="w-full overflow-x-auto rounded bg-slate-950 border border-slate-800 px-3 py-2 font-mono text-[12px] leading-relaxed text-slate-300 whitespace-pre">{preview.command}</pre>
  {:else}
    <div class="text-xs font-mono text-slate-500">{t('Rendering…')}</div>
  {/if}

  {#if conflicts.length > 0}
    <div class="rounded border border-rose-900 bg-rose-950/25 px-3 py-2 space-y-1">
      {#each conflicts as c (c.id)}
        <div class="text-[11px] text-rose-200">
          <span class="font-mono text-rose-300">{c.flags.join(' + ')}</span> — {t(c.message)}
        </div>
      {/each}
    </div>
  {/if}

  {#if shadowed.length > 0}
    <div class="rounded border border-amber-800 bg-amber-950/20 px-3 py-2 space-y-1.5">
      <div class="text-[11px] text-amber-200">
        {t('Passed twice — llama.cpp keeps the last one, so the value in the form never reaches the model:')}
      </div>
      {#each shadowed as sh (sh.flag)}
        <div class="text-[11px] font-mono">
          <span class="text-amber-300">{sh.flag}</span>
          <span class="text-slate-500 line-through ml-2">{sh.shadowed.join(', ')}</span>
          <span class="text-slate-600 mx-1">→</span>
          <span class="text-emerald-300">{sh.wins}</span>
        </div>
      {/each}
      <button type="button" onclick={fold} disabled={busy}
        class="rounded bg-emerald-700/40 border border-emerald-600 px-2.5 py-1 text-[11px] hover:bg-emerald-700/60 disabled:opacity-40">
        {t('Fold into the fields')}
      </button>
      <div class="text-[11px] text-slate-400">
        {t('Folding keeps the winning value and empties extra_flags of anything that has a field — the command stays identical, the editor stops lying about it.')}
      </div>
    </div>
  {/if}

  {#if parseError}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200 font-mono">{parseError}</div>
  {/if}

  {#if missingValues.length > 0}
    <div class="rounded border border-rose-800 bg-rose-950/25 px-3 py-2 space-y-1">
      <div class="text-[11px] text-rose-200">
        {t('These flags need a value — llama-server refuses the whole command line without one:')}
      </div>
      {#each missingValues as m (m.flag)}
        <div class="text-[11px] font-mono text-rose-300">
          {m.flag} {#if m.placeholder}<span class="text-slate-400">{m.placeholder}</span>{/if}
          {#if m.help}<div class="font-sans text-slate-400">{m.help}</div>{/if}
        </div>
      {/each}
    </div>
  {/if}

  {#if unknown.length > 0}
    <div class="rounded border border-amber-800 bg-amber-950/20 px-3 py-2 space-y-1">
      <div class="text-[11px] text-amber-200">
        {t('This llama-server build does not accept:')}
      </div>
      {#each unknown as u (u.flag)}
        <div class="text-[11px] font-mono text-amber-300">
          {u.flag}
          {#if u.suggestions.length > 0}
            <span class="text-slate-400">→ {t('did you mean')} {u.suggestions.join(', ')}?</span>
          {/if}
        </div>
      {/each}
      <div class="text-[11px] text-slate-400">
        {t('Read from the binary’s own --help, so it follows your build. Start it anyway and llama-server exits with a parse error instead.')}
      </div>
    </div>
  {/if}

  <details class="rounded border border-slate-800 bg-slate-900/40" ontoggle={loadFlags}>
    <summary class="cursor-pointer px-3 py-2 text-[11px] text-slate-400 hover:text-slate-200">
      {t('Flags this binary accepts')}
    </summary>
    <div class="border-t border-slate-800 px-3 py-2 space-y-2">
      {#if flagsLoading}
        <div class="text-[11px] font-mono text-slate-500">{t('Reading --help…')}</div>
      {:else if flagList && !flagsAvailable}
        <div class="text-[11px] text-amber-300">
          {t('The binary could not be queried, so nothing here is validated against it.')}
        </div>
      {:else if flagList}
        <input
          bind:value={flagFilter}
          placeholder={t('filter — e.g. reasoning, cache, moe')}
          class="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-[11px]"
        />
        <div class="max-h-64 overflow-y-auto space-y-1">
          {#each flagMatches as f (f.canonical)}
            <div class="text-[11px]">
              <span class="font-mono text-emerald-300">{f.names.join(', ')}</span>
              {#if f.placeholder}<span class="font-mono text-slate-500 ml-1">{f.placeholder}</span>{/if}
              <div class="text-slate-500">{f.help}</div>
            </div>
          {/each}
          {#if flagMatches.length === 0}
            <div class="text-[11px] text-slate-500">{t('No match.')}</div>
          {/if}
        </div>
        <div class="text-[11px] text-slate-500">
          {#if flagTotal > flagMatches.length}
            {t('showing {n} of {total} — type in the filter to narrow it down', { n: flagMatches.length, total: flagTotal })}
          {:else}
            {t('{n} flags', { n: flagTotal })}
          {/if}
        </div>
      {/if}
    </div>
  </details>

  {#if editing}
    {#if parsed && parsed.warnings.length > 0}
      <div class="rounded border border-amber-800 bg-amber-950/20 px-3 py-2 space-y-1">
        {#each parsed.warnings as w}
          <div class="text-[11px] text-amber-200">{w}</div>
        {/each}
      </div>
    {/if}

    {#if parsed}
      <div class="rounded border border-slate-800 bg-slate-900/60 px-3 py-2">
        <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-1">
          {parsed.diff.length > 0
            ? t('Applying changes {n} field(s)', { n: parsed.diff.length })
            : t('Identical to the form — applying changes nothing')}
        </div>
        {#each parsed.diff as d (d.field)}
          <div class="text-[11px] font-mono flex gap-2 flex-wrap">
            <span class="text-slate-400">{d.field}</span>
            <span class="text-rose-300 line-through">{fmt(d.from)}</span>
            <span class="text-slate-600">→</span>
            <span class="text-emerald-300">{fmt(d.to)}</span>
          </div>
        {/each}
      </div>
    {/if}

    <div class="flex items-center gap-2 flex-wrap">
      <button type="button" onclick={applyToFields} disabled={busy || !!parseError}
        class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-xs hover:bg-emerald-700/60 disabled:opacity-40">
        {t('Apply to fields')}
      </button>
      <button type="button" onclick={lock} disabled={busy}
        class="rounded bg-amber-700/40 border border-amber-600 px-3 py-1.5 text-xs hover:bg-amber-700/60 disabled:opacity-40">
        {t('Run this exact text')}
      </button>
      <span class="text-[11px] text-slate-500">
        {t('Apply keeps the form in charge (flags LlamaDeck has no field for land in extra_flags). Run-as-written keeps your text — order, spelling and all.')}
      </span>
    </div>
  {/if}
</div>
