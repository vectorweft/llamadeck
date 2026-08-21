<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api, type FeatureHint, type FitCheck, type FitSuggestion, type LlamaConfig, type LlamaDevice, type ModelDefaults, type ModelEntry, type RecommendedDrafter } from '$lib/api';
  import { modelLabel } from '$lib/ui';
  import MoeOffload from '$lib/components/MoeOffload.svelte';
  import ModelRecipes from '$lib/components/ModelRecipes.svelte';
  import CommandBox from '$lib/components/CommandBox.svelte';
  import { confirmDialog } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';
  import { toast } from '$lib/toast.svelte';
  import { fly } from 'svelte/transition';

  let presets = $state<LlamaConfig[]>([]);
  let editing = $state<LlamaConfig | null>(null);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let defaults = $state<ModelDefaults | null>(null);
  let defaultsLoading = $state(false);
  let defaultsForPath = $state<string>('');  // track which path current `defaults` belongs to
  let recommendedDrafter = $state<RecommendedDrafter | null>(null);
  let allModels = $state<ModelEntry[]>([]);
  // Offload targets of the configured binary. Empty when llama-server can't be
  // queried — the picker then says so instead of showing an empty list that
  // looks like "no GPUs".
  let devices = $state<LlamaDevice[]>([]);
  let devicesLoaded = $state(false);
  let defaultsAbort: AbortController | null = null;
  let defaultsSeq = 0;  // monotonically incremented; only the latest matters
  // The wizard names a new preset after the model's GGUF file. That name has
  // to keep following the model until the user types their own, otherwise
  // swapping the model in the open editor leaves a preset called
  // "lfm2-2.6b-exp-code" that actually serves Ornith.
  let nameIsAuto = $state(false);
  let wizardPurpose = $state<string>('chat');
  // Name the editor was opened on (null for a new preset). Saving under a
  // different name POSTs a *new* preset — the original stays — so the drawer
  // has to say so instead of looking like a rename.
  let originalName = $state<string | null>(null);

  async function load() {
    try {
      [presets, allModels] = await Promise.all([api.listPresets(), api.listModels()]);
      error = null;
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
    // Device list is best-effort too: a missing or unbuilt binary must not
    // stop the preset list from rendering.
    try {
      devices = (await api.serverDevices()).devices;
    } catch { devices = []; }
    devicesLoaded = true;
  }

  /** llama.cpp's own id for "offload nothing" — the CPU row's device id. */
  const CPU_DEVICE = 'none';

  /** Toggle one offload target in the editing preset's selection.
   *
   *  CPU is exclusive, in both directions: `-dev` takes a single list and
   *  `none` in it means "offload nothing", so `none,CUDA0` is not half of
   *  each — it is invalid. Picking the CPU therefore clears the GPUs, and
   *  picking a GPU clears the CPU.
   *
   *  n_gpu_layers follows, because LlamaDeck's own VRAM estimate and the fit
   *  panel both read ngl (not -dev) to decide how much of the model lands on
   *  the card. Leaving ngl at 999 next to `-dev none` would run correctly and
   *  still show a card's worth of VRAM the model does not use. */
  function toggleDevice(id: string) {
    if (!editing) return;
    const current = editing.devices ?? [];
    const on = current.includes(id);
    if (id === CPU_DEVICE) {
      editing.devices = on ? [] : [CPU_DEVICE];
      if (!on) editing.n_gpu_layers = 0;
      else if (editing.n_gpu_layers === 0) editing.n_gpu_layers = 999;
      return;
    }
    const withoutCpu = current.filter(d => d !== CPU_DEVICE && d !== id);
    editing.devices = on ? withoutCpu : [...withoutCpu, id];
    // Coming off the CPU row with ngl still pinned at 0 would keep the whole
    // model in RAM on a GPU the user just chose.
    if (current.includes(CPU_DEVICE) && editing.n_gpu_layers === 0) {
      editing.n_gpu_layers = 999;
    }
  }

  /** Whether the current selection is "keep it all in RAM". */
  const cpuPinned = $derived((editing?.devices ?? []).includes(CPU_DEVICE));

  function newPreset() {
    editorTab = 'basics';
    nameIsAuto = false;
    originalName = null;
    editing = {
      name: '',
      model_path: null, hf_repo: null, hf_file: null, mmproj_path: null,
      host: '127.0.0.1', port: 8080, api_key: null,
      ctx_size: 8192, n_gpu_layers: 999, parallel: 1,
      batch_size: 2048, ubatch_size: 512, threads: -1,
      flash_attn: 'auto', cache_type_k: 'f16', cache_type_v: 'f16', cont_batching: true,
      temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.0,
      jinja: false, metrics: true, slots: true,
      spec_type: 'none', model_path_draft: null, n_gpu_layers_draft: 999, draft_max: null, draft_min: null,
      devices: [], tensor_split: null,
      reasoning: 'auto', reasoning_effort: null, argv_override: null,
      env: {}, extra_flags: [], notes: '',
      estimated_vram_mb: null,
      mode: 'single', models_dir: null, models_max: 1, models_autoload: true,
      models_preset_path: null, sleep_idle_seconds: null
    };
  }

  async function save() {
    if (!editing) return;
    const name = editing.name.trim();
    const existing = presets.find(p => p.name === name);
    // Upsert is keyed by name, so typing the name of another preset silently
    // replaces it. Only the preset we opened may be overwritten unasked.
    if (existing && name !== originalName) {
      const ok = await confirmDialog(
        t('A preset named "{name}" already exists — its settings will be replaced by the ones in this editor.', { name }),
        { title: t('Overwrite preset?'), danger: true, confirmLabel: t('Overwrite') }
      );
      if (!ok) return;
    }
    busy = true;
    try {
      if (existing) await api.putPreset(name, { ...editing, name });
      else await api.createPreset({ ...editing, name });
      toast(t('Preset saved: {name}', { name }), 'success');
      editing = null;
      originalName = null;
      nameIsAuto = false;
      await load();
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
    finally { busy = false; }
  }

  // Presets that form the contract of external consumer projects — renaming /
  // deleting them breaks those projects (router alias contract).
  const EXTERNAL_PRESETS = new Set([
    'qwen3.6-27b', 'qwen3.6-27b-detect', 'qwen3.6-27b-chat', 'qwen3.6-27b-ocr', 'qwen3.6-27b-8x',
  ]);

  let showHidden = $state(false);
  const visiblePresets = $derived(presets.filter(p => !p.ui_hidden));
  const hiddenPresets = $derived(presets.filter(p => p.ui_hidden));

  async function toggleHidden(p: LlamaConfig) {
    try {
      await api.putPreset(p.name, { ...p, ui_hidden: !p.ui_hidden });
      await load();
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
  }

  async function del(name: string) {
    const external = EXTERNAL_PRESETS.has(name);
    const ok = await confirmDialog(
      external
        ? t('Preset "{name}" is used by external projects — deleting it BREAKS their connection.\n\nIf you just don\'t want to see it, use "Hide" instead. Delete permanently anyway?', { name })
        : t('Preset "{name}" will be permanently deleted.', { name }),
      { title: external ? t('Warning: external project contract!') : t('Delete preset?'), danger: true, confirmLabel: t('Delete') }
    );
    if (!ok) return;
    try { await api.deletePreset(name); toast(t('Preset deleted: {name}', { name }), 'success'); await load(); }
    catch (e) { error = e instanceof Error ? e.message : String(e); }
  }

  onMount(async () => {
    try {
      const w = parseInt(localStorage.getItem('llamadeck-preset-drawer-w') ?? '', 10);
      if (Number.isFinite(w)) setDrawerWidth(w);
    } catch { /* ignore */ }
    await load();
    // The Models / Download "Create preset" buttons land here with ?new_from=<path>
    // — start the wizard at step 2 with the model preselected.
    // ?new=1 (the dashboard setup card) opens the wizard at step 1.
    if ($page.url.searchParams.get('new') === '1') {
      wizard = { step: 1, model: null };
    }
    const fromPath = $page.url.searchParams.get('new_from');
    if (fromPath) {
      let m = allModels.find((x) => x.path === fromPath) ?? null;
      // A just-finished download may not be in the registry yet (the post-download
      // rescan can still be running). Rather than dropping the user at step 1 with
      // no model, scan once and retry so the flow "just works".
      if (!m) {
        try {
          await api.scanModels();
          await load();
          m = allModels.find((x) => x.path === fromPath) ?? null;
        } catch { /* fall through to step 1 */ }
      }
      wizard = { step: m ? 2 : 1, model: m };
    }
  });

  // ---- New-preset wizard (model → purpose → prefilled editor) ----
  let wizard = $state<{ step: 1 | 2; model: ModelEntry | null } | null>(null);
  // Drawer tabs — reset to Basics every time the editor opens.
  type EditorTab = 'basics' | 'performance' | 'spec' | 'advanced' | 'command';
  let editorTab = $state<EditorTab>('basics');
  const editorTabs: { id: EditorTab; label: string }[] = [
    { id: 'basics', label: 'Basics' },
    { id: 'performance', label: 'Performance' },
    { id: 'spec', label: 'Speculative' },
    { id: 'advanced', label: 'Advanced' },
    // Last tab, and deliberately last: it shows the finished command line and
    // lets a pro overrule every tab before it.
    { id: 'command', label: 'Command' },
  ];
  // Auto-apply the recommended sampling once model info loads (only for an
  // editor opened by the wizard — a manually opened editor is left alone).
  let autoApplyDefaults = false;

  interface Purpose {
    id: string;
    label: string;
    desc: string;
    apply: (e: LlamaConfig, m: ModelEntry | null) => void;
  }
  const purposes: Purpose[] = [
    { id: 'chat', label: 'Chat', desc: 'General use — 32K context, 2 parallel requests',
      apply: (e) => { e.ctx_size = 32768; e.parallel = 2; } },
    { id: 'code', label: 'Coding assistant', desc: '64K context for long files',
      apply: (e) => { e.ctx_size = 65536; e.parallel = 2; } },
    { id: 'reasoning', label: 'Deep reasoning', desc: 'Single request, 32K — thinking models (QwQ, OLMo-Think…)',
      apply: (e) => { e.ctx_size = 32768; e.parallel = 1; } },
    { id: 'vision', label: 'Vision', desc: 'Image input — mmproj is attached automatically when present',
      apply: (e, m) => { e.ctx_size = 16384; e.parallel = 1; if (m?.mmproj_path) e.mmproj_path = m.mmproj_path; } },
    { id: 'cpu', label: 'CPU (GPU busy)', desc: 'ngl=0, -dev none, 16 threads — the whole model stays in RAM',
      apply: (e) => { e.n_gpu_layers = 0; e.threads = 16; e.ctx_size = 16384; e.parallel = 1; e.cache_type_k = 'f16'; e.cache_type_v = 'f16'; e.devices = [CPU_DEVICE]; } },
    { id: 'blank', label: 'Start blank', desc: 'Open the editor with defaults, tune everything by hand',
      apply: () => {} },
  ];

  function suggestName(m: ModelEntry | null, purposeId: string): string {
    let stem = (m?.path.split('/').pop() ?? 'new').replace(/\.gguf$/i, '');
    stem = stem
      .replace(/-mtp$/i, '')
      .replace(/-(UD-)?(I?Q\d[\w]*|MXFP4(_MOE)?|F16|BF16)$/i, '')
      .toLowerCase().replace(/[^a-z0-9.]+/g, '-').replace(/^-+|-+$/g, '');
    let name = purposeId === 'blank' || purposeId === 'chat' ? stem : `${stem}-${purposeId}`;
    const names = new Set(presets.map((p) => p.name));
    let n = 2; const base = name;
    while (names.has(name)) name = `${base}-${n++}`;
    return name;
  }

  function freePort(): number {
    const used = new Set(presets.map((p) => p.port));
    for (let p = 8080; p < 8100; p++) if (!used.has(p)) return p;
    return 8100;
  }

  function createFromWizard(purpose: Purpose) {
    const m = wizard?.model ?? null;
    newPreset();
    if (!editing) return;
    editing.model_path = m?.path ?? null;
    editing.name = suggestName(m, purpose.id);
    editing.port = freePort();
    editing.flash_attn = 'on';
    editing.cache_type_k = 'q8_0';
    editing.cache_type_v = 'q8_0';
    editing.jinja = true;
    purpose.apply(editing, m);
    wizardPurpose = purpose.id;
    nameIsAuto = true;        // keeps following model_path until the user types a name
    autoApplyDefaults = true; // applied automatically once GGUF sampling suggestions arrive
    wizard = null;
  }

  // ---- Editor drawer width (drag the left edge) -------------------------
  // The drawer used to be a fixed max-w-2xl (672px) column, which is a tight
  // fit for a two-column form with a fit-check panel above it. Default wider,
  // let the user drag, and remember the choice per browser.
  const DRAWER_DEFAULT = 900, DRAWER_MIN = 520;
  let drawerWidth = $state(DRAWER_DEFAULT);
  let drawerEl: HTMLDivElement | null = $state(null);
  let resizing = $state(false);

  function drawerMax(): number {
    // clientWidth, not innerWidth: it is in the same (layout) coordinate space
    // as getBoundingClientRect/clientX, which the app-level CSS `zoom` shifts.
    return Math.max(DRAWER_MIN, document.documentElement.clientWidth - 48);
  }
  function setDrawerWidth(px: number, persist = false) {
    drawerWidth = Math.round(Math.min(drawerMax(), Math.max(DRAWER_MIN, px)));
    if (persist) { try { localStorage.setItem('llamadeck-preset-drawer-w', String(drawerWidth)); } catch { /* ignore */ } }
  }
  function startResize(e: PointerEvent) {
    if (!drawerEl) return;
    e.preventDefault();
    resizing = true;
    const handle = e.currentTarget as HTMLElement;
    handle.setPointerCapture(e.pointerId);
    // Width from the drawer's own right edge, so the maths survives the
    // app-level zoom and any scrollbar width.
    const right = drawerEl.getBoundingClientRect().right;
    const onMove = (ev: PointerEvent) => setDrawerWidth(right - ev.clientX);
    const onUp = () => {
      resizing = false;
      setDrawerWidth(drawerWidth, true);
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
    };
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  }
  function resetDrawerWidth() { setDrawerWidth(DRAWER_DEFAULT, true); }

  // Editing a saved preset and changing its name does not rename it: the PUT
  // is keyed by name, so a changed name POSTs a second preset and the original
  // stays. Surface that instead of letting it surprise people.
  const savesAsNew = $derived(
    !!editing && originalName !== null && editing.name.trim() !== originalName
  );

  const portConflict = $derived.by(() => {
    if (!editing) return null;
    const other = presets.find((p) => p.name !== editing!.name && p.port === editing!.port);
    return other ? other.name : null;
  });

  async function fetchDefaults(path: string, presetName: string | null): Promise<void> {
    if (!path || path === defaultsForPath) return;
    // Cancel any in-flight request and bump the sequence so the late
    // response of a previous click can't overwrite the current state.
    defaultsAbort?.abort();
    defaultsAbort = new AbortController();
    const mySeq = ++defaultsSeq;
    defaultsLoading = true;
    defaultsForPath = path;
    try {
      const bundle = await api.modelInfo(path, presetName);
      if (mySeq !== defaultsSeq) return;  // a newer request superseded us
      defaults = bundle.defaults;
      recommendedDrafter = bundle.info?.recommended_drafter ?? null;
      if (autoApplyDefaults) {
        autoApplyDefaults = false;
        applyDefaults();
      }
    } catch {
      if (mySeq !== defaultsSeq) return;
      defaults = null;
      recommendedDrafter = null;
    } finally {
      if (mySeq === defaultsSeq) defaultsLoading = false;
    }
  }

  // Candidate drafter list for the current target — same family, smaller, GGUF.
  const drafterCandidates = $derived.by<ModelEntry[]>(() => {
    if (!editing || !editing.model_path) return [];
    const targetPath = editing.model_path;
    const targetSize = allModels.find(m => m.path === targetPath)?.size_bytes ?? Infinity;
    const targetFamily = allModels.find(m => m.path === targetPath)?.family ?? null;

    let famRe: RegExp | null = null;
    let nameRe: RegExp | null = null;
    if (recommendedDrafter) {
      try { famRe = new RegExp(recommendedDrafter.family_pattern, 'i'); } catch {}
      try { nameRe = new RegExp(recommendedDrafter.name_pattern, 'i'); } catch {}
    }

    return allModels
      .filter(m => m.path !== targetPath && m.size_bytes < targetSize / 2)
      .filter(m => {
        if (famRe && m.family && famRe.test(m.family)) return true;
        if (nameRe && nameRe.test(m.path)) return true;
        // Fallback: same registry family
        if (!famRe && !nameRe && targetFamily && m.family === targetFamily) return true;
        return false;
      })
      .sort((a, b) => a.size_bytes - b.size_bytes);
  });

  function applyRecommendedDrafter() {
    if (!editing || !recommendedDrafter || drafterCandidates.length === 0) return;
    editing.model_path_draft = drafterCandidates[0].path;
    if (recommendedDrafter.draft_max != null) editing.draft_max = recommendedDrafter.draft_max;
    if (recommendedDrafter.draft_min != null) editing.draft_min = recommendedDrafter.draft_min;
  }

  // What "Apply to form" would actually write, field by field. Deriving it
  // instead of applying blind is what makes the button honest: it can say how
  // many values are still pending and go quiet once none are. Before this the
  // button looked identical after the click and the only confirmation was a
  // toast in the far bottom-right corner — behind the open drawer, and easy
  // to miss entirely.
  type SamplingChange = { field: string; from: unknown; to: number };

  function samplingChanges(): SamplingChange[] {
    if (!editing || !defaults) return [];
    const s = defaults.sampling;
    const out: SamplingChange[] = [];
    const want = (field: string, value: number | null | undefined) => {
      if (value == null) return;
      const current = (editing as unknown as Record<string, unknown>)[field];
      if (current !== value) out.push({ field, from: current, to: value });
    };
    want('temperature', s.temperature);
    // In GGUF, top_k=-1 means "disabled"; the argv layer never emits -1 and
    // the llama-server default (40) silently kicks in — 0 also means
    // "disabled" but IS emitted, preserving the vendor's intent.
    want('top_k', s.top_k == null ? null : (s.top_k < 0 ? 0 : Math.round(s.top_k)));
    want('top_p', s.top_p);
    want('min_p', s.min_p);
    want('repeat_penalty', s.repeat_penalty);
    // Qwen3.6-style families publish this one too; it defaults to null (flag
    // not emitted), so applying it has to be explicit.
    want('presence_penalty', s.presence_penalty);
    return out;
  }

  let pendingDefaults = $derived(samplingChanges());
  // Two-stage feedback: a "✓ applied" flash for the click itself, then the
  // resting state (nothing left to apply) carries it from there.
  let defaultsJustApplied = $state(false);
  let defaultsFlashTimer: ReturnType<typeof setTimeout> | null = null;

  function flashApplied() {
    defaultsJustApplied = true;
    if (defaultsFlashTimer) clearTimeout(defaultsFlashTimer);
    defaultsFlashTimer = setTimeout(() => (defaultsJustApplied = false), 2500);
  }

  function applyDefaults() {
    if (!editing || !defaults) return;
    const changes = samplingChanges();
    for (const c of changes) (editing as unknown as Record<string, unknown>)[c.field] = c.to;
    flashApplied();
    toast(changes.length
      ? t('{n} sampling value(s) written to the form.', { n: changes.length })
      : t('This GGUF carries no sampling recommendation.'), 'success');
  }

  const fmtSampling = (v: unknown) =>
    typeof v === 'number' ? (v < 10 ? v.toFixed(2) : String(Math.round(v)))
      : v == null ? '—' : String(v);

  // Fields the editor *derives* from the chosen model have to follow it when
  // the model changes mid-edit. Before this, picking a different GGUF in the
  // open editor left the auto-generated name (and a vision mmproj) pointing at
  // the previous file, so the saved preset was named after a model it did not
  // serve.
  // `undefined` is "the editor just opened", which is NOT the same as a preset
  // that has no model yet (null) — the wizard's "continue without a model"
  // starts at null and the first typed character has to count as a change.
  let lastModelPath: string | null | undefined;
  $effect(() => {
    if (!editing) { lastModelPath = undefined; return; }
    const path = editing.model_path ?? null;
    if (lastModelPath !== undefined && path === lastModelPath) return;
    const prev = lastModelPath;
    lastModelPath = path;
    if (prev === undefined) return;  // first pass after the editor opened — nothing changed yet
    const m = allModels.find(x => x.path === path) ?? null;
    if (nameIsAuto) editing.name = suggestName(m, wizardPurpose);
    // An mmproj that came from the *old* model (the vision purpose attaches
    // one) is wrong for the new one; a hand-typed path is left alone.
    const prevMmproj = allModels.find(x => x.path === prev)?.mmproj_path ?? null;
    if (!editing.mmproj_path || editing.mmproj_path === prevMmproj) {
      editing.mmproj_path = m?.mmproj_path ?? null;
    }
  });

  // When editing opens or model_path changes, refresh defaults.
  $effect(() => {
    if (editing && editing.model_path) {
      const existingName = presets.find(p => p.name === editing!.name) ? editing!.name : null;
      fetchDefaults(editing.model_path, existingName);
    } else {
      defaults = null;
      defaultsForPath = '';
    }
  });

  // ---- Fit-check --------------------------------------------------------
  // Compares the draft config in the editor against live GPU/RAM: does it
  // fit, should MoE experts move to RAM, is RAM enough? Debounced;
  // recomputed whenever model_path / ctx / KV / ngl / extra_flags change.
  let fit = $state<FitCheck | null>(null);
  let fitLoading = $state(false);
  let fitTimer: ReturnType<typeof setTimeout> | null = null;
  let fitAbort: AbortController | null = null;
  let fitSeq = 0;

  $effect(() => {
    if (!editing || editing.mode === 'router' || !editing.model_path) {
      fit = null;
      return;
    }
    // Read the relevant fields so $effect tracks their changes.
    const snapshot = JSON.stringify({
      model_path: editing.model_path, mmproj_path: editing.mmproj_path,
      ctx_size: editing.ctx_size, n_gpu_layers: editing.n_gpu_layers,
      parallel: editing.parallel, cache_type_k: editing.cache_type_k,
      cache_type_v: editing.cache_type_v, extra_flags: editing.extra_flags,
      // Pinning to one card changes the VRAM the plan is budgeted against,
      // so the fit panel must recompute when the selection changes.
      devices: editing.devices, mode: editing.mode,
    });
    if (fitTimer) clearTimeout(fitTimer);
    fitTimer = setTimeout(async () => {
      fitAbort?.abort();
      fitAbort = new AbortController();
      const mySeq = ++fitSeq;
      fitLoading = true;
      try {
        const r = await api.fitCheck(JSON.parse(snapshot), fitAbort.signal);
        if (mySeq !== fitSeq) return;
        fit = r.available ? r : null;
      } catch {
        if (mySeq === fitSeq) fit = null;
      } finally {
        if (mySeq === fitSeq) fitLoading = false;
      }
    }, 400);
  });

  // The suggestion rewrites fields further down the form and the panel above
  // only catches up after the debounced re-check, so without a mark on the
  // button the click reads as "nothing happened".
  let appliedSuggestion = $state<string | null>(null);
  let suggestionFlashTimer: ReturnType<typeof setTimeout> | null = null;

  function applySuggestion(s: FitSuggestion) {
    if (!editing) return;
    appliedSuggestion = s.id;
    if (suggestionFlashTimer) clearTimeout(suggestionFlashTimer);
    suggestionFlashTimer = setTimeout(() => (appliedSuggestion = null), 2500);
    // First strip the old offload flags (together with their values).
    if (s.remove_flags.length > 0) {
      const rm = new Set(s.remove_flags);
      const out: string[] = [];
      const flags = editing.extra_flags;
      for (let i = 0; i < flags.length; i++) {
        if (rm.has(flags[i])) {
          if (i + 1 < flags.length && !flags[i + 1].startsWith('-')) i++; // skip its value too
          continue;
        }
        out.push(flags[i]);
      }
      editing.extra_flags = out;
    }
    if (s.add_flags.length > 0) editing.extra_flags = [...editing.extra_flags, ...s.add_flags];
    for (const [k, v] of Object.entries(s.set)) (editing as unknown as Record<string, unknown>)[k] = v;
  }

  function fitPanelClass(level: string): string {
    if (level === 'fits') return 'border-emerald-800 bg-emerald-950/20';
    if (level === 'fits_if_alone' || level === 'needs_offload') return 'border-amber-800 bg-amber-950/20';
    return 'border-rose-900 bg-rose-950/20';
  }

  function fitIcon(level: string): string {
    if (level === 'fits') return '✅';
    if (level === 'fits_if_alone') return '⏸️';
    if (level === 'needs_offload') return '⚠️';
    return '⛔';
  }

  const gb = (mb: number) => (mb / 1024).toFixed(1);

  // ---- What's New hints (new llama.cpp features) ----

  // Which cards belong next to THIS preset is decided on the backend: it can
  // compare against the command the preset really runs and against the
  // binary's own flag list. The old client-side filter had neither, so it
  // treated "no architecture named" as "matches every model" and every field
  // flag (--model, --ctx-size) as "missing" — which is how a Qwen3-TTS card
  // ended up on a dense text preset offering to add a valueless --model.
  let featureHints = $state<FeatureHint[]>([]);
  let hintTimer: ReturnType<typeof setTimeout> | null = null;
  let hintSeq = 0;

  $effect(() => {
    const snapshot = editing ? JSON.stringify(editing) : null;
    const arch = defaults?.architecture ?? null;
    if (!snapshot) {
      featureHints = [];
      return;
    }
    if (hintTimer) clearTimeout(hintTimer);
    hintTimer = setTimeout(async () => {
      const mySeq = ++hintSeq;
      try {
        const r = await api.featureHints(JSON.parse(snapshot), arch);
        if (mySeq === hintSeq) featureHints = r.hints;
      } catch {
        if (mySeq === hintSeq) featureHints = [];
      }
    }, 500);
  });

  function applyFeatureFlags(hint: FeatureHint) {
    if (!editing) return;
    const add: string[] = [];
    for (const f of hint.add_flags) add.push(...f.split(' '));
    editing.extra_flags = [...editing.extra_flags, ...add];
  }

  function sourceBadgeClass(src: string | null | undefined): string {
    if (src === 'gguf')   return 'bg-emerald-950/50 border-emerald-800 text-emerald-300';
    if (src === 'props')  return 'bg-cyan-950/50 border-cyan-800 text-cyan-300';
    if (src === 'family') return 'bg-amber-950/50 border-amber-800 text-amber-300';
    return 'bg-slate-900 border-slate-800 text-slate-500';
  }

  function sourceLabel(src: string | null | undefined): string {
    if (src === 'gguf')   return t('from GGUF metadata');
    if (src === 'props')  return t('from live /props');
    if (src === 'family') return t('from family fallback');
    return t('unknown');
  }
</script>

<div class="max-w-5xl space-y-6">
  <div class="flex items-center justify-between">
    <h1 class="text-2xl font-semibold">Presets</h1>
    <button onclick={() => wizard = { step: 1, model: null }} class="rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1.5 text-sm hover:bg-emerald-700/60">+ {t('New preset')}</button>
  </div>

  {#if error}
    <div class="rounded border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200 font-mono">{error}</div>
  {/if}

  {#snippet presetCard(p: LlamaConfig)}
    <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <div class="flex items-start justify-between gap-4">
        <div class="min-w-0 flex-1">
          <div class="flex items-baseline gap-3 flex-wrap">
            <h3 class="text-lg font-mono text-emerald-400 break-all">{modelLabel(p, p.name)}</h3>
            <span class="text-xs text-slate-500 font-mono">preset: {p.name}</span>
            <span class="text-xs text-slate-500 font-mono">{p.host}:{p.port}</span>
            {#if (p.argv_override ?? '').trim()}
              <span
                class="inline-flex rounded-full border border-amber-800 bg-amber-950/50 text-amber-300 px-1.5 py-0.5 text-[10px] font-mono"
                title={t('This preset runs a raw command line — the summary below is read back from it, not the other way round.')}
              >⌘ {t('raw command')}</span>
            {/if}
            {#if EXTERNAL_PRESETS.has(p.name)}
              <span
                class="inline-flex rounded-full border border-amber-800 bg-amber-950/50 text-amber-300 px-1.5 py-0.5 text-[10px] font-mono"
                title={t('External projects use this preset — renaming/deleting it breaks them. Hiding is safe.')}
              >🔒 {t('external')}</span>
            {/if}
          </div>
          <div class="text-xs text-slate-500 font-mono mt-1 truncate">
            {p.model_path ? p.model_path.substring(0, p.model_path.lastIndexOf('/')) + '/' : (p.hf_repo ?? '(no model)')}
          </div>
          <div class="text-xs text-slate-500 mt-1">
            ctx {p.ctx_size} · ngl {p.n_gpu_layers} · np {p.parallel} · fa {p.flash_attn} · kv {p.cache_type_k}
            {p.jinja ? ' · jinja' : ''}{p.metrics ? ' · metrics' : ''}
            {#if p.spec_type === 'draft-mtp'}
              <span class="ml-1 inline-flex rounded-full border border-violet-800 bg-violet-950/50 text-violet-300 px-1.5 py-0.5 font-mono">spec-decode: MTP (self)</span>
            {:else if p.model_path_draft}
              <span class="ml-1 inline-flex rounded-full border border-violet-800 bg-violet-950/50 text-violet-300 px-1.5 py-0.5 font-mono">spec-decode: {(p.model_path_draft.split('/').pop() ?? '').replace(/\.gguf$/i, '')}</span>
            {/if}
          </div>
          {#if p.notes}
            <div class="text-xs text-slate-400 mt-2 italic">{p.notes}</div>
          {/if}
        </div>
        <div class="flex gap-2 shrink-0">
          <button
            onclick={() => toggleHidden(p)}
            title={p.ui_hidden ? t('Bring back to the list') : t('Move to the hidden section (the preset keeps running)')}
            class="rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60"
          >{p.ui_hidden ? t('Show') : t('Hide')}</button>
          <button onclick={() => {
            const st = p.spec_type === 'draft-model' ? 'draft-simple' : (p.spec_type ?? (p.model_path_draft ? 'draft-simple' : 'none'));
            editorTab = 'basics';
            nameIsAuto = false;      // a saved preset's name is its identity — never auto-rename it
            originalName = p.name;
            editing = { ...p, spec_type: st };
          }} class="rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60">{t('Edit')}</button>
          <button onclick={() => del(p.name)} class="rounded bg-rose-900/40 border border-rose-800 px-3 py-1 text-xs hover:bg-rose-900/60">{t('Delete')}</button>
        </div>
      </div>
    </div>
  {/snippet}

  <div class="space-y-3">
    {#each visiblePresets as p (p.name)}
      {@render presetCard(p)}
    {:else}
      <div class="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-6 text-center space-y-3">
        <div class="text-sm text-slate-400">
          {presets.length > 0 ? t('All presets are hidden — expand them below.') : t('No presets yet.')}
        </div>
        {#if presets.length === 0}
          <button
            onclick={() => wizard = { step: 1, model: null }}
            class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60"
          >{t('Create your first preset')}</button>
        {/if}
      </div>
    {/each}
  </div>

  {#if hiddenPresets.length > 0}
    <div class="space-y-3">
      <button
        onclick={() => showHidden = !showHidden}
        class="flex items-center gap-2 text-xs font-mono text-slate-500 hover:text-slate-300"
      >
        <span>{showHidden ? '▾' : '▸'}</span>
        {t('Hidden presets ({n})', { n: hiddenPresets.length })}
      </button>
      {#if showHidden}
        {#each hiddenPresets as p (p.name)}
          <div class="opacity-70">{@render presetCard(p)}</div>
        {/each}
      {/if}
    </div>
  {/if}
</div>

{#if editing}
  <div class="fixed inset-0 z-40 bg-black/60" role="presentation"></div>
  <div
    bind:this={drawerEl}
    class="fixed inset-y-0 right-0 z-50 flex flex-col border-l border-slate-800 bg-slate-900 shadow-2xl"
    style="width: {drawerWidth}px; max-width: 100vw"
    role="dialog"
    aria-modal="true"
    transition:fly={{ x: 480, duration: 200 }}
  >
    <!-- Drag the left edge to resize; double-click resets. The two-column
         form is cramped at the old fixed 42rem, and how much room it deserves
         depends on the screen, so it is the user's call and it is remembered. -->
    <div
      class="group absolute inset-y-0 left-0 z-10 w-1.5 -ml-0.5 cursor-col-resize touch-none"
      role="separator"
      aria-orientation="vertical"
      aria-label={t('Resize the editor')}
      onpointerdown={startResize}
      ondblclick={resetDrawerWidth}
    >
      <div class="h-full w-full transition-colors {resizing ? 'bg-emerald-600' : 'group-hover:bg-emerald-700/60'}"></div>
    </div>

    <!-- Drawer header -->
    <div class="flex items-center gap-3 border-b border-slate-800 px-5 py-3">
      <h2 class="text-lg font-semibold shrink-0">{originalName !== null ? t('Edit preset') : t('New preset')}</h2>
      {#if editing.name}<span class="min-w-0 truncate font-mono text-xs text-emerald-400">{editing.name}</span>{/if}
      <button
        onclick={() => editing = null}
        class="ml-auto shrink-0 rounded px-2 py-1 text-slate-500 hover:text-slate-200 hover:bg-slate-800/60"
        aria-label={t('Cancel')}
      >✕</button>
    </div>

    <!-- Fit-check strip: pinned above the tabs so performance edits show their effect immediately -->
    {#if editing.mode !== 'router' && editing.model_path && (fit || fitLoading)}
      <div class="max-h-52 overflow-y-auto border-b p-3 {fit ? fitPanelClass(fit.level) : 'border-slate-800 bg-slate-900/60'}">
        {#if fit}
          <div class="flex items-start gap-2">
            <span class="text-base leading-5">{fitIcon(fit.level)}</span>
            <div class="min-w-0 flex-1 space-y-1.5">
              <div class="text-sm text-slate-200">{fit.headline}</div>
              {#if fit.estimate && fit.hardware && fit.plan}
                <div class="text-[11px] font-mono text-slate-500 flex flex-wrap gap-x-3 gap-y-0.5">
                  <span>model {gb(fit.estimate.model_mb)} GB</span>
                  <span>KV cache {gb(fit.estimate.kv_cache_mb)} GB</span>
                  {#if fit.plan.ram_need_mb > 0}<span class="text-sky-300">{t('{g} GB to RAM', { g: gb(fit.plan.ram_need_mb) })}</span>{/if}
                  <span>{t('· GPU {total} GB (free {free})', { total: gb(fit.hardware.gpu_total_mb), free: gb(fit.hardware.gpu_free_mb) })}</span>
                  <span>{t('RAM free {g} GB', { g: gb(fit.hardware.ram_available_mb) })}</span>
                  <!-- Why the margin is the size it is. Without this the panel
                       silently asks for 2 GB on one model and 0.5 GB on the
                       next, and the difference looks like a bug. -->
                  {#if fit.plan.headroom_mb}
                    <span
                      class={fit.plan.measured ? 'text-emerald-300' : 'text-slate-500'}
                      title={fit.plan.measured
                        ? t('This model has run on this card before, so the number comes from a measurement rather than the formula — the margin can be small.')
                        : t('This model has never been measured on this card. The number comes from the formula, which can be ~2 GB out either way, so the margin covers it.')}
                    >{fit.plan.measured
                      ? t('measured · margin {g} GB', { g: gb(fit.plan.headroom_mb) })
                      : t('margin {g} GB', { g: gb(fit.plan.headroom_mb) })}</span>
                  {/if}
                  {#if fit.model?.is_moe}
                    <span class="text-violet-300">{t('MoE: {n} experts / {m} active', { n: fit.model.expert_count ?? '?', m: fit.model.expert_used_count ?? '?' })}</span>
                  {/if}
                </div>
              {/if}
              {#each fit.messages as m}
                <div class="text-[11px] {m.severity === 'error' ? 'text-rose-300' : m.severity === 'warn' ? 'text-amber-300' : 'text-slate-400'}">{m.text}</div>
              {/each}
              {#each fit.suggestions as s (s.id)}
                <div class="rounded border border-slate-700/70 bg-slate-900/50 p-2 flex items-start justify-between gap-3">
                  <div class="min-w-0">
                    <div class="text-xs text-slate-200">{s.label}</div>
                    <div class="text-[11px] text-slate-400 mt-0.5">{s.explanation}</div>
                  </div>
                  <button
                    type="button"
                    onclick={() => applySuggestion(s)}
                    class="shrink-0 rounded border px-3 py-1 text-xs transition-colors {appliedSuggestion === s.id
                      ? 'border-emerald-400 bg-emerald-600/70 text-emerald-50'
                      : 'border-emerald-600 bg-emerald-700/40 hover:bg-emerald-700/60'}"
                  >{appliedSuggestion === s.id ? t('✓ Applied') : t('Apply suggestion')}</button>
                </div>
              {/each}
            </div>
            {#if fitLoading}<span class="text-[10px] font-mono text-slate-500 shrink-0">{t('computing…')}</span>{/if}
          </div>
        {:else}
          <div class="text-xs text-slate-500 font-mono">{t('Running fit-check…')}</div>
        {/if}
      </div>
      <!-- MoE offload is the one knob that decides whether a 96 GB model runs
           at all on a 32 GB card, and as a raw --n-cpu-moe N in extra_flags it
           is unguessable: nothing tells you what a layer costs or how close to
           the edge you are. -->
      <div class="border-b border-slate-800 px-3 py-3">
        <MoeOffload
          {fit}
          flags={editing.extra_flags}
          onchange={(f) => editing!.extra_flags = f}
        />
      </div>
    {/if}

    <!-- Tab bar -->
    <div class="flex gap-1 border-b border-slate-800 px-5 pt-2" role="tablist">
      {#each editorTabs.filter(tb => tb.id !== 'spec' || editing!.mode !== 'router') as tb (tb.id)}
        <button
          role="tab"
          aria-selected={editorTab === tb.id}
          onclick={() => editorTab = tb.id}
          class="rounded-t px-3 py-1.5 text-xs uppercase tracking-wider border-b-2 -mb-px transition-colors
                 {editorTab === tb.id
                   ? 'border-emerald-500 text-emerald-300'
                   : 'border-transparent text-slate-500 hover:text-slate-200'}"
        >{t(tb.label)}</button>
      {/each}
    </div>

    <!-- Tab content -->
    <div class="flex-1 overflow-y-auto px-5 py-4">
      <!-- With a raw command stored, the fields below no longer decide what
           runs. Saying so once, everywhere, beats letting someone tune a
           slider that the launch ignores. -->
      {#if (editing.argv_override ?? '').trim() && editorTab !== 'command'}
        <button
          type="button"
          onclick={() => editorTab = 'command'}
          class="mb-4 w-full text-left rounded border border-amber-800 bg-amber-950/20 px-3 py-2 text-[11px] text-amber-200 hover:bg-amber-950/40"
        >
          {t('This preset runs a raw command — the fields below are only a reading of it. Open the Command tab to change what actually runs.')}
        </button>
      {/if}
      {#if editorTab === 'basics'}
        <div class="grid grid-cols-2 gap-4 text-sm">
          <label class="block">
            <span class="text-slate-400">name</span>
            <input
              bind:value={editing.name}
              oninput={() => (nameIsAuto = false)}
              class="mt-1 w-full rounded bg-slate-800 border {savesAsNew ? 'border-amber-600' : 'border-slate-700'} px-2 py-1 font-mono"
            />
            {#if savesAsNew}
              <span class="mt-1 block text-[11px] text-amber-400">{t('Saved under a new name — this creates a second preset and "{name}" stays in the list.', { name: originalName ?? '' })}</span>
            {:else if nameIsAuto}
              <span class="mt-1 block text-[11px] text-slate-500">{t('Named after the model file — it follows the model until you type your own name.')}</span>
            {/if}
          </label>
          <label class="block">
            <span class="text-slate-400">mode</span>
            <select bind:value={editing.mode} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono">
              <option value="single">{t('single — one model')}</option>
              <option value="router">{t('router — restart-free model switching (uses models_dir)')}</option>
            </select>
          </label>
          <label class="block">
            <span class="text-slate-400">host</span>
            <input bind:value={editing.host} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" />
            <span class="mt-1 block text-[11px] text-slate-500">
              {editing.host === '127.0.0.1' || editing.host === 'localhost'
                ? t('This machine only. Use 0.0.0.0 to serve the whole network.')
                : t('Reachable from the network — llama-server has no authentication of its own.')}
            </span>
          </label>
          <label class="block">
            <span class="text-slate-400">port</span>
            <input type="number" bind:value={editing.port} class="mt-1 w-full rounded bg-slate-800 border {portConflict ? 'border-amber-600' : 'border-slate-700'} px-2 py-1 font-mono" />
            {#if portConflict}
              <span class="text-[11px] text-amber-400 mt-0.5 block">{t('This port is also used by preset "{name}" — they cannot run at the same time.', { name: portConflict })}</span>
            {/if}
          </label>

          {#if editing.mode === 'router'}
            <label class="block col-span-2"><span class="text-slate-400">models_dir</span><input bind:value={editing.models_dir} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" placeholder="~/llama.cpp/models" /></label>
            <label class="block"><span class="text-slate-400">models_max</span><input type="number" min="1" bind:value={editing.models_max} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
            <label class="flex items-center gap-2 text-sm pt-5"><input type="checkbox" bind:checked={editing.models_autoload} /> models_autoload</label>
            <label class="block col-span-2"><span class="text-slate-400">models_preset_path (optional INI override)</span><input bind:value={editing.models_preset_path} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" placeholder="~/.config/llamadeck/router-models.ini" /></label>
            <label class="block"><span class="text-slate-400">sleep_idle_seconds (optional)</span><input type="number" min="0" value={editing.sleep_idle_seconds ?? ''} oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; editing!.sleep_idle_seconds = v === '' ? null : Number(v); }} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
            <div class="col-span-2 text-xs text-slate-500 italic px-2">
              {t('Router mode ignores the model_path / mmproj_path / sampling fields — sub-models are loaded at runtime via /models/load. The ctx_size / n_gpu_layers / parallel / batch_size below become the defaults inherited by every loaded model.')}
            </div>
          {:else}
            <label class="block col-span-2">
              <span class="text-slate-400">model_path</span>
              <div class="mt-1 flex gap-2">
                <select
                  value={allModels.find(m => m.path === editing!.model_path) ? editing!.model_path : ''}
                  onchange={(e) => {
                    const v = (e.currentTarget as HTMLSelectElement).value;
                    if (v) editing!.model_path = v;
                  }}
                  class="rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs flex-shrink-0 max-w-[18rem]"
                  title="Pick from scanned registry"
                >
                  <option value="">{t('— pick from registry —')}</option>
                  {#each allModels.filter(m => !m.path.includes('/_drafters/')).sort((a,b) => (a.family ?? '').localeCompare(b.family ?? '') || a.size_bytes - b.size_bytes) as m}
                    <option value={m.path}>[{m.family ?? '—'}] {(m.path.split('/').pop() ?? '').replace(/\.gguf$/i, '')} · {m.size_gb.toFixed(1)} GB</option>
                  {/each}
                </select>
                <input bind:value={editing.model_path} class="flex-1 rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" placeholder="~/llama.cpp/models/..." />
              </div>
            </label>
          {/if}

          <!-- Model-defaults banner (read from GGUF / /props / family) -->
          {#if editing.model_path}
            <div class="col-span-2 rounded border border-slate-800 bg-slate-900/60 p-3">
              {#if defaultsLoading}
                <div class="text-xs text-slate-500 font-mono">{t('Inspecting GGUF…')}</div>
              {:else if defaults && defaults.source !== 'none'}
                <div class="flex items-center justify-between flex-wrap gap-2 mb-2">
                  <div class="flex items-center gap-2 flex-wrap text-xs">
                    <span class="uppercase tracking-wider text-slate-500">{t('recommended defaults')}</span>
                    <span class="inline-flex rounded-full border px-2 py-0.5 font-mono {sourceBadgeClass(defaults.source)}">{sourceLabel(defaults.source)}</span>
                    {#if defaults.architecture}
                      <span class="text-slate-500 font-mono">arch: <span class="text-slate-300">{defaults.architecture}</span></span>
                    {/if}
                    {#if defaults.quantized_by}
                      <span class="text-slate-500 font-mono">quant-by: <span class="text-slate-300">{defaults.quantized_by}</span></span>
                    {/if}
                    {#if defaults.context_length}
                      <span class="text-slate-500 font-mono">ctx_len: <span class="text-slate-300">{defaults.context_length.toLocaleString()}</span></span>
                    {/if}
                  </div>
                  {#if Object.keys(defaults.sampling).length > 0}
                    <button
                      type="button"
                      onclick={applyDefaults}
                      disabled={pendingDefaults.length === 0}
                      title={pendingDefaults.length === 0
                        ? t('The form already carries these values')
                        : t('Writes the values marked below into the sampling fields')}
                      class="rounded border px-3 py-1 text-xs transition-colors {defaultsJustApplied
                        ? 'border-emerald-400 bg-emerald-600/70 text-emerald-50'
                        : pendingDefaults.length === 0
                          ? 'border-slate-700 bg-slate-800/60 text-slate-400 cursor-default'
                          : 'border-emerald-600 bg-emerald-700/40 hover:bg-emerald-700/60'}"
                    >{defaultsJustApplied
                      ? t('✓ Applied')
                      : pendingDefaults.length === 0
                        ? t('✓ In the form')
                        : t('Apply to form ({n})', { n: pendingDefaults.length })}</button>
                  {/if}
                </div>
                <!-- Marking what differs from the form turns the apply into
                     something you can watch happen: the arrows collapse into
                     plain values as the fields take them. -->
                <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono">
                  {#each Object.entries(defaults.sampling) as [k, v]}
                    {@const pend = pendingDefaults.find(c => c.field === k)}
                    <span class="text-slate-500">{k}:
                      {#if pend}
                        <span class="text-rose-300 line-through">{fmtSampling(pend.from)}</span>
                        <span class="text-slate-600">→</span>
                        <span class="text-amber-300">{fmtSampling(v)}</span>
                      {:else}
                        <span class="text-emerald-300">{fmtSampling(v)}</span>
                      {/if}
                    </span>
                  {/each}
                </div>
                {#if defaults.chat_template_preview}
                  <details class="mt-2">
                    <summary class="text-[11px] text-slate-600 cursor-pointer hover:text-slate-400 font-mono">chat_template preview</summary>
                    <pre class="mt-1 text-[11px] text-slate-500 font-mono whitespace-pre-wrap break-all max-h-32 overflow-y-auto">{defaults.chat_template_preview}</pre>
                  </details>
                {/if}
              {:else if defaults}
                <div class="text-xs text-slate-500 font-mono">{t('No recommended defaults found for this GGUF (neither embedded nor a family match).')}</div>
              {/if}
            </div>
          {/if}

          <!-- What this model itself wants: thinking on/off and the sampling
               each mode needs, read from its chat template + GGUF metadata. -->
          {#if editing.model_path && editing.mode !== 'router'}
            <div class="col-span-2 rounded border border-slate-800 bg-slate-900/60 p-3">
              <ModelRecipes
                config={editing}
                presetName={originalName}
                onapply={(c) => editing = c}
              />
            </div>
          {/if}

          <!-- Sampling. These fields existed in the config and in `to_argv` from
               the start, but had no input anywhere — so "Apply to form" wrote
               values nobody could see, and users ended up hand-writing
               `--temp 0.7` into extra_flags, which then shadowed the fields. -->
          {#if editing.mode !== 'router'}
            <div class="col-span-2">
              <div class="text-slate-400 text-sm mb-1">{t('sampling')}</div>
              <div class="grid grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] gap-2">
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">temperature</span>
                  <input type="number" step="0.05" min="0" bind:value={editing.temperature}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">top_k</span>
                  <input type="number" step="1" min="0" bind:value={editing.top_k}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">top_p</span>
                  <input type="number" step="0.01" min="0" max="1" bind:value={editing.top_p}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">min_p</span>
                  <input type="number" step="0.01" min="0" max="1" bind:value={editing.min_p}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">repeat_penalty</span>
                  <input type="number" step="0.05" min="0" bind:value={editing.repeat_penalty}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
                <label class="block">
                  <span class="text-[11px] text-slate-500 font-mono">presence_penalty</span>
                  <input
                    type="number" step="0.1"
                    value={editing.presence_penalty ?? ''}
                    oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; editing!.presence_penalty = v === '' ? null : Number(v); }}
                    placeholder={t('model default')}
                    class="mt-0.5 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs" />
                </label>
              </div>
              <div class="text-[11px] text-slate-500 mt-1">
                {t('Empty presence_penalty means "do not pass the flag" — the model’s own default applies. These are per-preset defaults; an API request that sends its own sampling still wins.')}
              </div>
            </div>
          {/if}

          <label class="block col-span-2"><span class="text-slate-400">ctx_size</span><input type="number" bind:value={editing.ctx_size} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block col-span-2"><span class="text-slate-400">notes</span><textarea bind:value={editing.notes} rows="2" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono"></textarea></label>
        </div>

      {:else if editorTab === 'performance'}
        <div class="grid grid-cols-2 gap-4 text-sm">
          <div class="col-span-2">
            <span class="text-slate-400">{t('Offload target')}</span>
            {#if !devicesLoaded}
              <div class="mt-1 text-xs font-mono text-slate-500">…</div>
            {:else if devices.length === 0}
              <div class="mt-1 rounded border border-slate-700 bg-slate-800/50 px-2 py-2 text-xs font-mono text-slate-400">
                {t('Could not read the device list from the llama-server binary — llama.cpp will choose.')}
              </div>
            {:else}
              <div class="mt-1 space-y-1">
                {#each devices as d (d.id)}
                  {@const picked = (editing.devices ?? []).includes(d.id)}
                  <label class="flex items-center gap-2 rounded border px-2 py-1.5 font-mono text-xs {d.selectable ? 'border-slate-700 bg-slate-800/50 cursor-pointer' : 'border-slate-800 bg-slate-900/40 opacity-60 cursor-not-allowed'}">
                    <input type="checkbox" checked={picked} disabled={!d.selectable}
                      onchange={() => toggleDevice(d.id)} class="accent-sky-500" />
                    <span class="w-20 shrink-0 text-slate-200">{d.label ?? d.id}</span>
                    <span class="flex-1 truncate text-slate-400" title={d.name}>{d.name}</span>
                    <span class="shrink-0 tabular-nums text-slate-500">{(d.total_mb / 1024).toFixed(0)} GB</span>
                    {#if d.duplicate_of}
                      <span class="shrink-0 text-amber-500/80">{t('same card as {id}', { id: d.duplicate_of })}</span>
                    {:else if d.may_alias}
                      <span class="shrink-0 text-amber-500/80" title={t('An RPC endpoint reports no hardware identity, so this cannot be confirmed — if it is the same card, picking both double-books it.')}>{t('may be {id}', { id: d.may_alias })}</span>
                    {:else if d.integrated}
                      <span class="shrink-0 text-slate-500">{t('integrated')}</span>
                    {:else if d.software}
                      <span class="shrink-0 text-slate-500">{t('software')}</span>
                    {:else if d.id === CPU_DEVICE}
                      <span class="shrink-0 text-slate-500">{t('system RAM')}</span>
                    {/if}
                  </label>
                {/each}
              </div>
              <div class="mt-1 text-xs font-mono text-slate-500">
                {#if (editing.devices ?? []).length === 0}
                  {t('Nothing selected — llama.cpp decides (no -dev flag).')}
                {:else}
                  -dev {(editing.devices ?? []).join(',')}
                {/if}
              </div>
              {#if cpuPinned}
                <div class="mt-1 text-xs text-slate-400">
                  {t('Nothing is offloaded: the whole model and its KV cache stay in system RAM, and n_gpu_layers is held at 0. Speed comes from CPU cores and memory bandwidth — the right choice for a small model while the cards are busy.')}
                </div>
              {/if}
            {/if}
          </div>
          {#if (editing.devices ?? []).length > 1}
            <label class="block col-span-2">
              <span class="text-slate-400">tensor_split</span>
              <input bind:value={editing.tensor_split} placeholder="1,0"
                class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" />
              <span class="mt-1 block text-xs font-mono text-slate-500">
                {t('Share of the layers per selected GPU, in order. "1,0" keeps every layer the tensor overrides do not place on the first GPU.')}
              </span>
            </label>
          {/if}
          <label class="block"><span class="text-slate-400">n_gpu_layers</span><input type="number" bind:value={editing.n_gpu_layers} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block"><span class="text-slate-400">parallel</span><input type="number" bind:value={editing.parallel} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block"><span class="text-slate-400">batch_size</span><input type="number" bind:value={editing.batch_size} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block"><span class="text-slate-400">ubatch_size</span><input type="number" bind:value={editing.ubatch_size} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block"><span class="text-slate-400">threads</span><input type="number" bind:value={editing.threads} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          <label class="block">
            <span class="text-slate-400">flash_attn</span>
            <select bind:value={editing.flash_attn} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono">
              <option value="auto">auto</option><option value="on">on</option><option value="off">off</option>
            </select>
          </label>
          <label class="block">
            <span class="text-slate-400">cache_type_k</span>
            <select bind:value={editing.cache_type_k} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono">
              <option>f16</option><option>f32</option><option>bf16</option><option>q8_0</option><option>q4_0</option><option>q4_1</option><option>q5_0</option><option>q5_1</option><option>iq4_nl</option>
            </select>
          </label>
          <label class="block">
            <span class="text-slate-400">cache_type_v</span>
            <select bind:value={editing.cache_type_v} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono">
              <option>f16</option><option>f32</option><option>bf16</option><option>q8_0</option><option>q4_0</option><option>q4_1</option><option>q5_0</option><option>q5_1</option><option>iq4_nl</option>
            </select>
          </label>
          <label class="block col-span-2">
            <span class="text-slate-400">estimated VRAM (MB)</span>
            <input
              type="number"
              min="0"
              step="100"
              value={editing.estimated_vram_mb ?? ''}
              oninput={(e) => {
                const v = (e.currentTarget as HTMLInputElement).value;
                editing!.estimated_vram_mb = v === '' ? null : Number(v);
              }}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono"
              placeholder="e.g. 22000 for a 35B Q4 with long ctx, leave empty if unknown"
            />
            <span class="text-xs text-slate-500 mt-1 block">
              {t('Used by the VRAM budget bar on the Server page. Rule of thumb: GGUF file size + (ctx × n_layers × 0.25 MB for q8_0 KV) + 1–2 GB compute buffer.')}
              {#if fit?.plan}
                <button
                  type="button"
                  onclick={() => { if (fit?.plan) editing!.estimated_vram_mb = fit.plan.gpu_need_mb; }}
                  class="ml-1 rounded border border-slate-600 bg-slate-700/40 px-2 py-0.5 text-[11px] hover:bg-slate-700/60"
                >{t('Use live estimate ({mb} MB)', { mb: fit.plan.gpu_need_mb.toLocaleString() })}</button>
              {/if}
            </span>
          </label>
          <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={editing.cont_batching} /> cont-batching</label>
        </div>

      {:else if editorTab === 'spec' && editing.mode !== 'router'}
        <div class="space-y-2 text-sm">
          <div class="flex items-center justify-between gap-2 flex-wrap">
            <span class="text-xs uppercase tracking-wider text-violet-300">{t('Speculative decoding — speed boost')} {#if (editing.spec_type ?? 'none') !== 'none'}<span class="ml-1 normal-case text-violet-400">({editing.spec_type})</span>{/if}</span>
            {#if recommendedDrafter && drafterCandidates.length > 0 && (editing.spec_type ?? 'none') === 'draft-simple'}
              <button
                type="button"
                onclick={applyRecommendedDrafter}
                class="rounded bg-violet-700/40 border border-violet-600 px-2 py-0.5 text-[11px] hover:bg-violet-700/60"
                title={recommendedDrafter.rationale}
              >{t('Apply recommendation ({label})', { label: recommendedDrafter.label })}</button>
            {/if}
          </div>

          <label class="block">
            <span class="text-slate-400 text-xs">strategy (--spec-type)</span>
            <select
              value={editing.spec_type ?? 'none'}
              onchange={(e) => {
                const v = (e.currentTarget as HTMLSelectElement).value as 'none' | 'draft-mtp' | 'draft-simple' | 'ngram-simple';
                editing!.spec_type = v;
                if (v !== 'draft-simple') { editing!.model_path_draft = null; }
                if (v === 'none' || v === 'ngram-simple') { editing!.draft_max = null; editing!.draft_min = null; }
              }}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
            >
              <option value="none">{t('— off —')}</option>
              <option value="draft-mtp">{t("draft-mtp — the model's own MTP head (most accurate)")}</option>
              <option value="draft-simple">{t('draft-simple — separate small GGUF drafter')}</option>
              <option value="ngram-simple">{t('ngram-simple — model-free, predicts from input repetition (free)')}</option>
            </select>
          </label>

          {#if editing.spec_type === 'draft-mtp'}
            <div class="text-[11px] text-violet-200/80 italic">
              {t("Uses the model's MTP (multi-token-prediction) head as the drafter. If the head is embedded in the GGUF (e.g. Qwen3.6 nextn tensors), leave the drafter field empty; if a separate head file exists (e.g. mtp-gemma-4-31B-it.gguf), pick it as the drafter. Supported architectures depend on your build — the What's New page announces new ones (currently: gemma4, qwen3.5/3.6, glm4, deepseek3.2 and others).")}
            </div>
            <label class="block">
              <span class="text-slate-400 text-xs">{t('MTP head file (leave empty if embedded)')}</span>
              <select
                value={editing.model_path_draft ?? ''}
                onchange={(e) => { const v = (e.currentTarget as HTMLSelectElement).value; editing!.model_path_draft = v || null; }}
                class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
              >
                <option value="">{t('— embedded MTP (no file) —')}</option>
                {#each allModels.filter(m => /mtp/i.test(m.path)) as m}
                  <option value={m.path}>{(m.path.split('/').pop() ?? '')} · {m.size_gb.toFixed(1)} GB</option>
                {/each}
                {#if editing.model_path_draft && !allModels.find(m => m.path === editing!.model_path_draft)}
                  <option value={editing.model_path_draft}>{(editing.model_path_draft.split('/').pop() ?? '')} (custom)</option>
                {/if}
              </select>
            </label>
          {/if}

          {#if editing.spec_type === 'ngram-simple'}
            <div class="text-[11px] text-violet-200/80 italic">
              {t('Needs no extra model; speeds things up when the output repeats the input (document editing, summarization, code refactoring). Little effect in free-form chat.')}
            </div>
          {/if}

          {#if editing.spec_type === 'draft-simple'}
            {#if recommendedDrafter}
              <div class="text-[11px] text-violet-200/80 italic">{recommendedDrafter.rationale}</div>
            {/if}
            <label class="block">
              <span class="text-slate-400 text-xs">drafter model</span>
              <select
                value={editing.model_path_draft ?? ''}
                onchange={(e) => {
                  const v = (e.currentTarget as HTMLSelectElement).value;
                  editing!.model_path_draft = v || null;
                  if (!v) { editing!.draft_max = null; editing!.draft_min = null; }
                }}
                class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
              >
                <option value="">{t('— pick a drafter GGUF —')}</option>
                {#if drafterCandidates.length === 0}
                  <option value="" disabled>{t('no candidates in registry (download a small GGUF from the same family first)')}</option>
                {/if}
                {#each drafterCandidates as m}
                  <option value={m.path}>{(m.path.split('/').pop() ?? '').replace(/\.gguf$/i, '')} · {m.size_gb.toFixed(1)} GB · {m.family ?? '—'}</option>
                {/each}
                {#if editing.model_path_draft && !drafterCandidates.find(m => m.path === editing!.model_path_draft)}
                  <option value={editing.model_path_draft}>{(editing.model_path_draft.split('/').pop() ?? '')} (custom)</option>
                {/if}
              </select>
            </label>
            <!-- A drafter picked for a *previous* target model survives a model
                 swap in the open editor, and llama-server then refuses to start
                 (a drafter must share the target's vocabulary). Say so here
                 rather than at launch. -->
            {#if editing.model_path_draft && !defaultsLoading && drafterCandidates.length > 0
                 && !drafterCandidates.find(m => m.path === editing!.model_path_draft)}
              <div class="text-[11px] text-amber-400">
                {t('This drafter does not belong to the target model — llama-server rejects a drafter whose vocabulary differs from the model it drafts for.')}
              </div>
            {/if}
          {/if}

          {#if editing.spec_type === 'draft-mtp' || editing.spec_type === 'draft-simple'}
            <div class="grid grid-cols-[repeat(auto-fit,minmax(7rem,1fr))] gap-2">
              <label class="block">
                <span class="text-slate-400 text-xs">--draft-max</span>
                <input
                  type="number"
                  min="1"
                  value={editing.draft_max ?? ''}
                  oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; editing!.draft_max = v === '' ? null : Number(v); }}
                  placeholder={editing.spec_type === 'draft-mtp' ? '3' : '16'}
                  class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
                />
              </label>
              <label class="block">
                <span class="text-slate-400 text-xs">--draft-min</span>
                <input
                  type="number"
                  min="0"
                  value={editing.draft_min ?? ''}
                  oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; editing!.draft_min = v === '' ? null : Number(v); }}
                  placeholder={editing.spec_type === 'draft-mtp' ? '0' : '4'}
                  class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
                />
              </label>
              {#if editing.spec_type === 'draft-simple'}
                <label class="block">
                  <span class="text-slate-400 text-xs">-ngld (drafter GPU layers)</span>
                  <input
                    type="number"
                    bind:value={editing.n_gpu_layers_draft}
                    class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
                  />
                </label>
              {/if}
            </div>
          {/if}
        </div>

      {:else if editorTab === 'command'}
        <CommandBox config={editing} onapply={(c) => editing = c} />

      {:else}
        <div class="grid grid-cols-2 gap-4 text-sm">
          {#if editing.mode !== 'router'}
            <label class="block col-span-2"><span class="text-slate-400">mmproj_path (optional)</span><input bind:value={editing.mmproj_path} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
            <label class="block"><span class="text-slate-400">hf_repo</span><input bind:value={editing.hf_repo} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" placeholder="bartowski/..." /></label>
            <label class="block"><span class="text-slate-400">hf_file</span><input bind:value={editing.hf_file} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono" /></label>
          {/if}
          <!-- Thinking. Also here (not only on the model card) because a router
               preset has no model of its own, and its value lands in the INI's
               [*] section as the default every loaded model inherits. -->
          <label class="block">
            <span class="text-slate-400">reasoning</span>
            <select bind:value={editing.reasoning} class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono">
              <option value="auto">{t('auto — let the chat template decide')}</option>
              <option value="on">on — --reasoning on</option>
              <option value="off">off — --reasoning off</option>
            </select>
          </label>
          <label class="block">
            <span class="text-slate-400">reasoning_effort</span>
            <select
              value={editing.reasoning_effort ?? ''}
              onchange={(e) => editing!.reasoning_effort = (e.currentTarget as HTMLSelectElement).value || null}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono"
            >
              <option value="">{t('template default')}</option>
              {#each ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as eff}
                <option value={eff}>{eff}</option>
              {/each}
            </select>
          </label>
          <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={editing.jinja} /> jinja</label>
          <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={editing.metrics} /> metrics</label>
          <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={editing.slots} /> slots</label>
          <!-- What's New hints live here, not in Basics: adding a raw flag is
               an Advanced-tab action, and it sits right next to the
               extra_flags box it writes into. Cards are ordered
               architecture-match first; anything with nothing to add is
               filtered out on the backend. -->
          <div class="col-span-2 text-slate-400 text-sm">{t('new in llama.cpp')}</div>
          {#each featureHints as hint (hint.card.id)}
            <div class="col-span-2 rounded border border-violet-800 bg-violet-950/20 p-3">
              <div class="flex items-center justify-between flex-wrap gap-2">
                <div class="flex items-center gap-2 flex-wrap text-xs">
                  <span class="inline-flex rounded-full border border-violet-800 bg-violet-950/50 px-2 py-0.5 text-violet-300">{t('New')}</span>
                  <span class="text-slate-200">{hint.card.title_tr}</span>
                  <span class="inline-flex rounded-full border border-slate-700 bg-slate-800/50 px-2 py-0.5 font-mono text-slate-400">
                    {hint.match === 'architecture' ? t('this architecture') : t('any model')}
                  </span>
                  {#each hint.add_flags as f}
                    <span class="inline-flex rounded-full border border-amber-800 bg-amber-950/50 px-2 py-0.5 font-mono text-amber-300">{f}</span>
                  {/each}
                </div>
                <div class="flex items-center gap-2">
                  <a href="/whats-new" class="text-[11px] text-sky-400 hover:underline">{t('details')}</a>
                  {#if hint.add_flags.length > 0}
                    <button
                      type="button"
                      onclick={() => applyFeatureFlags(hint)}
                      class="rounded bg-violet-700/40 border border-violet-600 px-3 py-1 text-xs hover:bg-violet-700/60"
                    >{t('Add flags')}</button>
                  {/if}
                </div>
              </div>
              <div class="mt-1 text-[11px] text-slate-400">{hint.card.why_tr}</div>
            </div>
          {/each}
          <!-- Environment. Not cosmetic: some llama.cpp behaviour has no flag
               at all (GGML_CUDA_DISABLE_GRAPHS), so without this the only way
               to reach it is wrapping the binary in a shell script — which
               then applies to every preset instead of this one. -->
          <label class="block col-span-2">
            <span class="text-slate-400">{t('environment (one KEY=VALUE per line)')}</span>
            <textarea
              rows="2"
              value={Object.entries(editing.env ?? {}).map(([k, v]) => `${k}=${v}`).join('\n')}
              oninput={(e) => {
                const out: Record<string, string> = {};
                for (const line of (e.currentTarget as HTMLTextAreaElement).value.split('\n')) {
                  const t = line.trim();
                  if (!t) continue;
                  const i = t.indexOf('=');
                  if (i > 0) out[t.slice(0, i).trim()] = t.slice(i + 1);
                }
                editing!.env = out;
              }}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono text-xs"
              placeholder="GGML_CUDA_DISABLE_GRAPHS=1"
            ></textarea>
            <span class="mt-1 block text-[11px] text-slate-500">
              {t('Applies to this preset only, and shows up at the front of the Command tab exactly as a shell would take it.')}
            </span>
          </label>
          <label class="block col-span-2">
            <span class="text-slate-400">extra_flags (space-separated)</span>
            <input
              value={editing.extra_flags.join(' ')}
              oninput={(e) => editing!.extra_flags = (e.currentTarget as HTMLInputElement).value.split(/\s+/).filter(Boolean)}
              class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 font-mono"
              placeholder="--reasoning-format deepseek --no-context-shift -n 16384"
            />
          </label>
        </div>
      {/if}
    </div>

    <!-- Drawer footer -->
    <div class="flex items-center justify-end gap-2 border-t border-slate-800 px-5 py-3">
      <button onclick={() => editing = null} class="rounded bg-slate-700/40 border border-slate-600 px-4 py-1.5 text-sm hover:bg-slate-700/60">{t('Cancel')}</button>
      <button disabled={busy || !editing.name.trim()} onclick={save} class="rounded bg-emerald-700/40 border border-emerald-600 px-4 py-1.5 text-sm hover:bg-emerald-700/60 disabled:opacity-40">{savesAsNew ? t('Save as a new preset') : t('Save')}</button>
    </div>
  </div>
{/if}

<!-- New-preset wizard: model → purpose → prefilled editor -->
{#if wizard}
  <div class="fixed inset-0 z-50 bg-black/70 flex items-start justify-center p-6 overflow-y-auto" role="dialog">
    <div class="w-full max-w-xl rounded-lg border border-slate-800 bg-slate-900 p-6 space-y-4 my-10">
      <div class="flex items-center gap-3">
        <h2 class="text-lg font-semibold">{t('New preset')}</h2>
        <span class="text-xs font-mono text-slate-500">{t('step {n}/2', { n: wizard.step })}</span>
        <button onclick={() => wizard = null} class="ml-auto rounded bg-slate-700/40 border border-slate-600 px-3 py-1 text-xs hover:bg-slate-700/60">{t('Cancel')}</button>
      </div>

      {#if wizard.step === 1}
        <div class="text-sm text-slate-400">{t('Which model?')} <span class="text-slate-500">{t('(for a router preset or manual setup, pick "Start blank" in step 2)')}</span></div>
        <div class="max-h-[calc(50*var(--vh))] overflow-y-auto space-y-1.5">
          {#each allModels.filter(m => !/mtp-/i.test(m.path.split('/').pop() ?? '')).sort((a,b) => (a.family ?? 'zz').localeCompare(b.family ?? 'zz') || b.size_bytes - a.size_bytes) as m}
            <button
              onclick={() => wizard = { step: 2, model: m }}
              class="w-full text-left rounded border border-slate-800 bg-slate-900/60 hover:border-emerald-700 hover:bg-slate-800/60 px-3 py-2 flex items-center gap-3"
            >
              <span class="inline-flex rounded-full border border-slate-700 bg-slate-800/50 px-2 py-0.5 text-[11px] font-mono text-slate-400 shrink-0">{m.family ?? '—'}</span>
              <span class="text-sm font-mono text-slate-200 truncate flex-1">{(m.path.split('/').pop() ?? '').replace(/\.gguf$/i, '')}</span>
              <span class="text-xs font-mono text-slate-500 shrink-0">{m.size_gb.toFixed(1)} GB</span>
            </button>
          {/each}
        </div>
        <button
          onclick={() => wizard = { step: 2, model: null }}
          class="w-full rounded border border-dashed border-slate-700 px-3 py-2 text-sm text-slate-400 hover:text-slate-200 hover:border-slate-500"
        >{t("Continue without a model (I'll type it in)")}</button>
      {:else}
        <div class="text-sm text-slate-400">
          {#if wizard.model}
            <span class="font-mono text-emerald-400">{(wizard.model.path.split('/').pop() ?? '').replace(/\.gguf$/i, '')}</span> — {t('what will it be used for?')}
          {:else}
            {t('What will it be used for?')}
          {/if}
          <span class="text-slate-500">{t('The purpose sets context, parallelism and GPU settings — everything can be changed later in the editor.')}</span>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {#each purposes as pu}
            <button
              onclick={() => createFromWizard(pu)}
              class="text-left rounded border border-slate-800 bg-slate-900/60 hover:border-emerald-700 hover:bg-slate-800/60 px-3 py-2.5"
            >
              <div class="text-sm text-slate-200">{t(pu.label)}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">{t(pu.desc)}</div>
            </button>
          {/each}
        </div>
        <button onclick={() => wizard = { step: 1, model: wizard?.model ?? null }} class="text-xs text-slate-500 hover:text-slate-300">{t('← back to model')}</button>
      {/if}
    </div>
  </div>
{/if}
