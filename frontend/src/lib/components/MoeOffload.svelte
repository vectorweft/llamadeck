<script lang="ts">
  import { t } from '$lib/i18n.svelte';
  import type { FitCheck } from '$lib/api';

  let {
    fit,
    flags,
    onchange,
  }: {
    /** Latest fit-check; the control only appears for MoE models. */
    fit: FitCheck | null;
    /** The preset's current extra_flags. */
    flags: string[];
    /** Called with the rewritten extra_flags. */
    onchange: (flags: string[]) => void;
  } = $props();

  const MOE_FLAGS = new Set(['--cpu-moe', '-cmoe', '--n-cpu-moe', '-ncmoe']);

  const moe = $derived(
    fit?.model?.is_moe && fit.model.n_exp_layers && fit.model.exps_mb && fit.estimate
      ? {
          nLayers: fit.model.n_exp_layers,
          perLayerMb: fit.model.exps_mb / fit.model.n_exp_layers,
          // Everything that stays on the GPU no matter how the experts are split.
          fixedMb:
            fit.estimate.model_mb - fit.model.exps_mb +
            fit.estimate.kv_cache_mb + fit.estimate.compute_mb,
        }
      : null
  );

  /** Expert layers currently parked in RAM, read back out of extra_flags. */
  const current = $derived.by<number>(() => {
    if (!moe) return 0;
    for (let i = 0; i < flags.length; i++) {
      if (flags[i] === '--cpu-moe' || flags[i] === '-cmoe') return moe.nLayers;
      if ((flags[i] === '--n-cpu-moe' || flags[i] === '-ncmoe') && i + 1 < flags.length) {
        const n = parseInt(flags[i + 1], 10);
        if (!Number.isNaN(n)) return Math.min(Math.max(n, 0), moe.nLayers);
      }
    }
    return 0;
  });

  // Same correction fit_check applies, so the slider and the panel agree.
  const calib = $derived(fit?.plan?.calibration_mb ?? 0);
  // ...and the same safety margin, which is no longer a constant: a model
  // that has actually been measured on this card gets a much smaller one.
  // Hardcoding 2048 here made the slider call a fit "over budget" while the
  // panel one line above said it fits.
  const headroomMb = $derived(fit?.plan?.headroom_mb ?? 2048);
  const gpuMb = (n: number) =>
    moe ? Math.max(0, moe.fixedMb + (moe.nLayers - n) * moe.perLayerMb - calib) : 0;
  const ramMb = (n: number) => (moe ? n * moe.perLayerMb : 0);

  const gpuFree = $derived(fit?.hardware?.gpu_free_mb ?? 0);
  const ramFree = $derived(fit?.hardware?.ram_available_mb ?? 0);

  const predGpu = $derived(gpuMb(current));
  const predRam = $derived(ramMb(current));
  const gpuPct = $derived(gpuFree > 0 ? (predGpu / gpuFree) * 100 : 0);

  /** Fewest layers in RAM that still leaves the allocator its headroom. */
  const fastest = $derived.by<number>(() => {
    if (!moe) return 0;
    for (let n = 0; n <= moe.nLayers; n++) if (gpuMb(n) + headroomMb <= gpuFree) return n;
    return moe.nLayers;
  });
  /** Two layers of extra margin — survives a browser or compositor taking VRAM. */
  const safe = $derived(moe ? Math.min(moe.nLayers, fastest + 2) : 0);

  const overBudget = $derived(predGpu + headroomMb > gpuFree);
  const ramTight = $derived(predRam > ramFree * 0.92);
  const barClass = $derived(
    overBudget ? 'bg-rose-500' : gpuPct >= 88 ? 'bg-amber-400' : 'bg-emerald-500'
  );

  const gb = (mb: number) => (mb / 1024).toFixed(1);

  function setLayers(n: number) {
    if (!moe) return;
    const out: string[] = [];
    for (let i = 0; i < flags.length; i++) {
      if (MOE_FLAGS.has(flags[i])) {
        // Drop the flag's value too, when it carries one.
        if ((flags[i] === '--n-cpu-moe' || flags[i] === '-ncmoe') && i + 1 < flags.length) i++;
        continue;
      }
      out.push(flags[i]);
    }
    if (n >= moe.nLayers) out.push('--cpu-moe');
    else if (n > 0) out.push('--n-cpu-moe', String(n));
    onchange(out);
  }
</script>

{#if moe}
  <div class="rounded border border-violet-900/60 bg-violet-950/10 p-3 space-y-2">
    <div class="flex items-baseline justify-between gap-3">
      <span class="text-xs uppercase tracking-wider text-violet-300 font-mono">{t('Expert offload')}</span>
      <span class="text-[11px] text-slate-500 font-mono">
        {t('{n} expert layers · {mb} GB each', { n: moe.nLayers, mb: (moe.perLayerMb / 1024).toFixed(2) })}
      </span>
    </div>

    <p class="text-[11px] text-slate-400 leading-relaxed">
      {t('This model is too big for the GPU alone, but only a few experts run per token — so the expert layers can sit in RAM at a small speed cost while attention stays on the GPU. Drag to choose how many.')}
    </p>

    <input
      type="range" min="0" max={moe.nLayers} step="1" value={current}
      oninput={(e) => setLayers(parseInt((e.currentTarget as HTMLInputElement).value, 10))}
      class="w-full accent-violet-400"
      aria-label={t('Expert layers in RAM')}
    />

    <div class="flex items-baseline justify-between text-xs font-mono">
      <span class="text-slate-300">
        {t('{n} of {total} expert layers in RAM', { n: current, total: moe.nLayers })}
      </span>
      <span class="text-slate-500">
        {current === 0 ? t('all on GPU') : current >= moe.nLayers ? '--cpu-moe' : `--n-cpu-moe ${current}`}
      </span>
    </div>

    <!-- Predicted VRAM against what is actually free right now. -->
    <div class="h-2 w-full rounded bg-slate-800 overflow-hidden">
      <div class="h-full {barClass} transition-all" style="width: {Math.min(100, gpuPct)}%"></div>
    </div>
    <div class="flex justify-between text-[11px] font-mono">
      <span class={overBudget ? 'text-rose-300' : 'text-slate-300'}>
        {t('GPU ~{g} GB / {f} GB free', { g: gb(predGpu), f: gb(gpuFree) })}
      </span>
      <span class={ramTight ? 'text-amber-300' : 'text-slate-400'}>
        {t('RAM ~{g} GB / {f} GB free', { g: gb(predRam), f: gb(ramFree) })}
      </span>
    </div>

    {#if overBudget}
      <div class="text-[11px] text-rose-300">
        {t('This will not fit — llama-server will fail with an out-of-memory error at startup. Move more layers to RAM.')}
      </div>
    {:else if gpuPct >= 88}
      <div class="text-[11px] text-amber-300">
        {t('Very little VRAM left over. Anything else that wants the GPU — a browser, the desktop compositor — can push this over at startup.')}
      </div>
    {/if}
    {#if ramTight && !overBudget}
      <div class="text-[11px] text-amber-300">
        {t('The RAM share is close to what is free. The model will still run, but it will page from disk and slow down sharply.')}
      </div>
    {/if}

    <div class="flex gap-2 pt-0.5">
      <button type="button" onclick={() => setLayers(fastest)}
        class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-[11px] hover:bg-slate-700">
        {t('Fastest that fits ({n})', { n: fastest })}
      </button>
      <button type="button" onclick={() => setLayers(safe)}
        class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-[11px] hover:bg-slate-700">
        {t('Leave margin ({n})', { n: safe })}
      </button>
      <button type="button" onclick={() => setLayers(moe.nLayers)}
        class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-[11px] hover:bg-slate-700">
        {t('All experts in RAM')}
      </button>
    </div>
  </div>
{/if}
