<script lang="ts">
  /** A labelled readout: big number first, segmented bar underneath.
   *
   * The bar is deliberately segmented rather than continuous. A solid fill
   * invites you to read its *length*, which is the slow way to get a number
   * you have already been given; discrete blocks read as an at-a-glance
   * level and stop competing with the digits for attention.
   */
  type Tone = 'emerald' | 'cyan' | 'violet' | 'amber' | 'rose' | 'slate';

  let {
    label,
    labelLang = 'en',
    value,
    unit = '',
    pct = 0,
    tone = 'emerald',
    sub = '',
    mark = 0,
    segments = 20,
    escalate = false,
    size = 'md',
    title = ''
  }: {
    label: string;
    /** Language of `label`, for CSS uppercasing. Nearly every meter here is
     *  headed by an English technical term (active, kv-cache, iGPU load), and
     *  under the document's lang="tr" the browser would dot their capital I.
     *  Pass `tLang('…')` alongside a translated label. */
    labelLang?: 'en' | 'tr';
    /** Pre-formatted — this component never rounds a number it was handed. */
    value: string;
    unit?: string;
    /** 0–100, drives the bar only. */
    pct?: number;
    tone?: Tone;
    /** Small right-aligned note on the label row. */
    sub?: string;
    /** Second marker on the bar (0 hides it) — e.g. the VRAM a preset is
     *  planned to need, against the VRAM it is actually holding. */
    mark?: number;
    segments?: number;
    /** Turn amber past 70 % and rose past 90 %. For pools (VRAM, RAM, ctx),
     *  where "nearly full" is the thing you want to catch. Not for rates. */
    escalate?: boolean;
    size?: 'sm' | 'md' | 'lg';
    title?: string;
  } = $props();

  const clamped = $derived(Math.min(100, Math.max(0, pct)));
  const shown = $derived(escalate ? (clamped >= 90 ? 'rose' : clamped >= 70 ? 'amber' : tone) : tone);

  const lit = {
    emerald: 'bg-emerald-400', cyan: 'bg-cyan-400', violet: 'bg-violet-400',
    amber: 'bg-amber-400', rose: 'bg-rose-400', slate: 'bg-slate-400'
  } as const;
  const txt = {
    emerald: 'text-emerald-300', cyan: 'text-cyan-300', violet: 'text-violet-300',
    amber: 'text-amber-300', rose: 'text-rose-300', slate: 'text-slate-300'
  } as const;
  const numSize = { sm: 'text-lg', md: 'text-2xl', lg: 'text-4xl' } as const;
  const barH = { sm: 'h-1.5', md: 'h-2', lg: 'h-2.5' } as const;

  // A segment lights when the reading reaches its *start*, so any non-zero
  // value lights at least one block — a card at 1 % must not look asleep.
  const cells = $derived(Array.from({ length: segments }, (_, i) => (i / segments) * 100 < clamped));
</script>

<div class="min-w-0" {title}>
  <div class="flex items-baseline justify-between gap-2">
    <span lang={labelLang} class="text-[10px] font-mono uppercase tracking-widest text-slate-500 truncate">{label}</span>
    {#if sub}<span class="text-[10px] font-mono text-slate-600 tabular-nums shrink-0">{sub}</span>{/if}
  </div>
  <div class="flex items-baseline gap-1 leading-none mt-0.5">
    <span class="{numSize[size]} font-mono font-semibold tabular-nums {txt[shown]}">{value}</span>
    {#if unit}<span class="text-[11px] font-mono text-slate-500 whitespace-nowrap">{unit}</span>{/if}
  </div>
  {#if segments > 0}
    <div class="relative mt-1.5 flex gap-[2px] {barH[size]}">
      {#each cells as on}
        <span class="flex-1 rounded-[1px] {on ? lit[shown] : 'bg-slate-800'}"></span>
      {/each}
      {#if mark > 0}
        <span class="absolute inset-y-[-2px] w-[2px] bg-amber-300/90 rounded-full"
          style="left: {Math.min(100, mark)}%"></span>
      {/if}
    </div>
  {/if}
</div>
