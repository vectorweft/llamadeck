<script lang="ts">
  /**
   * What THIS model wants, as buttons.
   *
   * Answers the question a freshly downloaded GGUF always raises: does it
   * think, what sampling does each mode want, and which flags do I need. The
   * capabilities come from the model's own chat template (so a model released
   * yesterday works today), the numbers from its GGUF metadata, live /props or
   * the family table — whichever is most authoritative, labelled either way.
   *
   * Thinking and instruct are NOT the same preset with a switch flipped: the
   * two modes want different sampling, so each recipe carries both.
   */
  import { api, type LlamaConfig, type ModelProfile, type ModelRecipe } from '$lib/api';
  import { t } from '$lib/i18n.svelte';

  let {
    config,
    presetName = null,
    onapply,
  }: {
    config: LlamaConfig;
    /** Preset name, so a running server's live /props can win over the GGUF. */
    presetName?: string | null;
    onapply: (cfg: LlamaConfig) => void;
  } = $props();

  let profile = $state<ModelProfile | null>(null);
  let loading = $state(false);
  let forPath = $state('');
  let seq = 0;

  $effect(() => {
    const path = config.model_path;
    const mmproj = config.mmproj_path;
    if (!path || config.mode === 'router') {
      profile = null;
      forPath = '';
      return;
    }
    if (path === forPath) return;
    const mySeq = ++seq;
    loading = true;
    api.modelProfile(path, presetName, mmproj)
      .then((p) => {
        if (mySeq !== seq) return;
        profile = p;
        forPath = path;
      })
      .catch(() => { if (mySeq === seq) profile = null; })
      .finally(() => { if (mySeq === seq) loading = false; });
  });

  /** How a capability chip reads, and what the reader needs to know about it.
   *
   *  `tools` is the one that has to be spelled out. It means the model can
   *  emit tool calls, which llama-server serves through --jinja — the flag
   *  this editor already sets. It is NOT llama.cpp's `--tools`, which turns on
   *  the server's own built-in agent tools (read_file, exec_shell_command…)
   *  and has nothing to do with the model. Read as a hint to pass `--tools`,
   *  the chip costs a preset that will not start: that flag needs a value, and
   *  llama-server refuses the whole command line without one. */
  function capMeta(cap: string): { label: string; hint: string } {
    if (cap === 'tools') {
      return {
        label: t('tool calling'),
        hint: t('The model can emit tool calls; llama-server serves them through --jinja, already set here. This is not the --tools flag, which enables llama.cpp\'s own built-in agent tools.'),
      };
    }
    return { label: t(cap), hint: '' };
  }

  /** Same rules as the backend's `apply_recipe`: set fields, drop flags with
   *  their values, then append. Dropping the value too is what stops
   *  `--n-cpu-moe 24` and `--n-cpu-moe 32` from both ending up in the list. */
  function apply(r: ModelRecipe) {
    const next = { ...config } as LlamaConfig & Record<string, unknown>;
    for (const [k, v] of Object.entries(r.set)) next[k] = v;
    let flags = [...(config.extra_flags ?? [])];
    if (r.remove_flags.length > 0) {
      const drop = new Set(r.remove_flags);
      const out: string[] = [];
      for (let i = 0; i < flags.length; i++) {
        if (drop.has(flags[i])) {
          if (i + 1 < flags.length && !flags[i + 1].startsWith('-')) i++;
          continue;
        }
        out.push(flags[i]);
      }
      flags = out;
    }
    for (const f of r.add_flags) if (!flags.includes(f)) flags.push(f);
    next.extra_flags = flags;
    onapply(next as LlamaConfig);
  }

  function setReasoning(v: 'auto' | 'on' | 'off') {
    onapply({ ...config, reasoning: v });
  }

  const EFFORTS = ['', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

  const sourceLabel = (src: string) => {
    if (src === 'gguf') return t('from the GGUF itself');
    if (src === 'props') return t('from the running server');
    if (src === 'family-variants') return t('from the family best-practices table');
    if (src === 'family') return t('from the family fallback table');
    return t('no recommendation found');
  };

  const num = (v: unknown) =>
    typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v);
</script>

{#if loading && !profile}
  <div class="text-xs font-mono text-slate-500">{t('Reading the model…')}</div>
{:else if profile}
  <div class="space-y-3">
    <div class="flex items-center gap-2 flex-wrap text-[11px]">
      <span class="uppercase tracking-wider text-slate-500">{t('this model')}</span>
      <span class="font-mono text-slate-300">{profile.model_id}</span>
      {#if profile.architecture}
        <span class="inline-flex rounded-full border border-slate-700 bg-slate-800/50 px-2 py-0.5 font-mono text-slate-400">{profile.architecture}</span>
      {/if}
      {#each Object.entries(profile.capabilities) as [cap, has]}
        {#if has}
          {@const meta = capMeta(cap)}
          <span class="inline-flex rounded-full border border-emerald-800 bg-emerald-950/40 px-2 py-0.5 text-emerald-300"
                title={[profile.detected_by[cap], meta.hint].filter(Boolean).join(' — ')}>{meta.label}</span>
        {/if}
      {/each}
      <span class="text-slate-500">{sourceLabel(profile.sampling_source)}</span>
    </div>

    {#if profile.capabilities.thinking}
      <div class="rounded border border-slate-800 bg-slate-900/60 p-3 space-y-2">
        <div class="flex items-center gap-3 flex-wrap">
          <span class="text-xs text-slate-400">{t('Thinking')}</span>
          <div class="flex rounded border border-slate-700 overflow-hidden">
            {#each ['auto', 'on', 'off'] as opt}
              <button type="button" onclick={() => setReasoning(opt as 'auto' | 'on' | 'off')}
                class="px-2.5 py-1 text-[11px] font-mono transition-colors
                       {(config.reasoning ?? 'auto') === opt
                          ? 'bg-emerald-700/40 text-emerald-200'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200'}">{opt}</button>
            {/each}
          </div>
          <label class="flex items-center gap-1.5 text-[11px] text-slate-400">
            {t('effort')}
            <select
              value={config.reasoning_effort ?? ''}
              onchange={(e) => onapply({ ...config, reasoning_effort: (e.currentTarget as HTMLSelectElement).value || null })}
              class="rounded bg-slate-800 border border-slate-700 px-1.5 py-0.5 font-mono text-[11px]"
            >
              {#each EFFORTS as eff}
                <option value={eff}>{eff === '' ? t('template default') : eff}</option>
              {/each}
            </select>
          </label>
        </div>
        <div class="text-[11px] text-slate-500">
          {t('“auto” leaves it to the chat template — llama-server’s own default. Forcing it emits --reasoning on|off; the sampling for each mode is what the recipes below set.')}
        </div>
        <div class="text-[11px] text-slate-600">
          {t('This is the server-side default. A client can still flip one request with chat_template_kwargs {"enable_thinking": false} or reasoning_effort "none" — the request wins.')}
        </div>
      </div>
    {/if}

    {#if profile.recipes.length > 0}
      <div class="space-y-1.5">
        {#each profile.recipes as r (r.id)}
          <div class="flex items-start justify-between gap-3 rounded border border-slate-800 bg-slate-900/40 px-3 py-2">
            <div class="min-w-0">
              <div class="text-xs text-slate-200 flex items-center gap-2 flex-wrap">
                {r.label}
                {#if r.source === 'user'}
                  <span class="inline-flex rounded-full border border-sky-800 bg-sky-950/40 px-1.5 py-0.5 text-[10px] text-sky-300">{t('yours')}</span>
                {/if}
              </div>
              {#if r.why}<div class="text-[11px] text-slate-500 mt-0.5">{r.why}</div>{/if}
              <div class="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] font-mono text-slate-500">
                {#each Object.entries(r.set) as [k, v]}
                  <span>{k}: <span class="text-slate-300">{num(v)}</span></span>
                {/each}
                {#each r.add_flags as f}
                  <span class="text-amber-300">{f}</span>
                {/each}
              </div>
            </div>
            <button type="button" onclick={() => apply(r)}
              class="shrink-0 rounded bg-emerald-700/40 border border-emerald-600 px-3 py-1 text-[11px] hover:bg-emerald-700/60">{t('Apply')}</button>
          </div>
        {/each}
      </div>
    {/if}

    {#each profile.notes as n}
      <div class="text-[11px] text-slate-500">{n}</div>
    {/each}

    <div class="text-[11px] text-slate-600">
      {t('Add your own recipes for this or any model in {path} — they show up here, tagged “yours”.', { path: profile.overlay_path })}
    </div>
  </div>
{/if}
