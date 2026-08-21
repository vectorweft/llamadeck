<script lang="ts">
  /** A radial instrument: 270° of arc with the reading as the hero.
   *
   * This is what replaced the compact dashboard's tok/s sparkline. A moving
   * line answers "what shape was the last 30 seconds"; the question people
   * actually put to a dashboard is "what is it doing *now*", and the answer
   * to that is a number, not a curve. The one piece of history worth keeping
   * — the peak — survives as a tick on the arc, which costs no reading
   * effort at all.
   */
  type Tone = 'emerald' | 'cyan' | 'violet' | 'amber' | 'rose';

  let {
    value,
    max,
    label,
    labelLang = 'en',
    unit = '',
    peak = 0,
    tone = 'emerald',
    digits = 1,
    size = 152,
    sub = '',
    title = ''
  }: {
    value: number;
    /** Full-scale deflection. */
    max: number;
    /** Caption under the dial. */
    label: string;
    /** Language of `label`, for CSS uppercasing — see Meter.svelte. English by
     *  default, which is what every technical readout here is called. */
    labelLang?: 'en' | 'tr';
    unit?: string;
    /** Marked with a tick on the arc; 0 hides it. */
    peak?: number;
    tone?: Tone;
    digits?: number;
    size?: number;
    /** Small line under the caption (peak, lifetime average…). */
    sub?: string;
    title?: string;
  } = $props();

  const R = 40;
  const SWEEP = 270;      // degrees of travel, gap centred at the bottom
  const START = 135;      // 7-o'clock, so the gap sits under the number
  const C = 2 * Math.PI * R;
  const ARC = (SWEEP / 360) * C;

  const scale = $derived(Math.max(max, 1e-6));
  const frac = $derived(Math.min(1, Math.max(0, value / scale)));
  const peakFrac = $derived(Math.min(1, Math.max(0, peak / scale)));

  // Wide readings (a four-digit prompt rate) get a smaller face rather than
  // spilling out of the dial.
  const text = $derived(
    value >= 1000 ? Math.round(value).toLocaleString()
    : value >= 100 ? value.toFixed(0)
    : value.toFixed(digits)
  );
  const face = $derived(text.length >= 6 ? 'text-xl' : text.length >= 5 ? 'text-2xl' : 'text-3xl');

  const arcTone = { emerald: 'stroke-emerald-400', cyan: 'stroke-cyan-400', violet: 'stroke-violet-400', amber: 'stroke-amber-400', rose: 'stroke-rose-400' } as const;
  const textTone = { emerald: 'text-emerald-300', cyan: 'text-cyan-300', violet: 'text-violet-300', amber: 'text-amber-300', rose: 'text-rose-300' } as const;
  const glowTone = { emerald: 'stroke-emerald-500/20', cyan: 'stroke-cyan-500/20', violet: 'stroke-violet-500/20', amber: 'stroke-amber-500/20', rose: 'stroke-rose-500/20' } as const;

  function polar(r: number, deg: number): [number, number] {
    const a = ((deg - 90) * Math.PI) / 180;
    return [50 + r * Math.cos(a), 50 + r * Math.sin(a)];
  }
  // Ticks every 10 % of scale — the graduations that make an arc readable as
  // an instrument instead of a decorative ring.
  const ticks = Array.from({ length: 11 }, (_, i) => {
    const deg = START + 90 + (i / 10) * SWEEP;
    const [x1, y1] = polar(R + 6, deg);
    const [x2, y2] = polar(R + (i % 5 === 0 ? 10 : 8.5), deg);
    return { x1, y1, x2, y2, major: i % 5 === 0 };
  });
  const peakTick = $derived.by(() => {
    const deg = START + 90 + peakFrac * SWEEP;
    const [x1, y1] = polar(R - 6, deg);
    const [x2, y2] = polar(R + 6, deg);
    return { x1, y1, x2, y2 };
  });
</script>

<div class="shrink-0" style="width: {size}px" {title}>
  <div class="relative" style="height: {size}px">
  <svg viewBox="0 0 100 100" width={size} height={size} class="block" role="img"
    aria-label="{label} {text} {unit}">
    <g transform="rotate({START} 50 50)">
      <circle cx="50" cy="50" r={R} fill="none" class="stroke-slate-800" stroke-width="9"
        stroke-dasharray="{ARC} {C}" stroke-linecap="round" />
      {#if frac > 0}
        <!-- A wide, faint copy under the arc reads as backlight on a dark
             panel; it is what keeps a 3 %-of-scale reading visible. -->
        <circle cx="50" cy="50" r={R} fill="none" class={glowTone[tone]} stroke-width="15"
          stroke-dasharray="{frac * ARC} {C}" stroke-linecap="round" />
        <circle cx="50" cy="50" r={R} fill="none" class={arcTone[tone]} stroke-width="9"
          stroke-dasharray="{frac * ARC} {C}" stroke-linecap="round" />
      {/if}
    </g>
    {#each ticks as tk}
      <line x1={tk.x1} y1={tk.y1} x2={tk.x2} y2={tk.y2} stroke-width={tk.major ? 1.6 : 0.9}
        class={tk.major ? 'stroke-slate-600' : 'stroke-slate-700'} stroke-linecap="round" />
    {/each}
    {#if peak > 0}
      <line x1={peakTick.x1} y1={peakTick.y1} x2={peakTick.x2} y2={peakTick.y2}
        class="stroke-slate-300" stroke-width="1.8" stroke-linecap="round" opacity="0.75" />
    {/if}
  </svg>
  <div class="absolute inset-0 flex flex-col items-center justify-center gap-0 pointer-events-none">
    <div class="{face} font-mono font-semibold leading-none tabular-nums {textTone[tone]}">{text}</div>
    {#if unit}
      <!-- Units are never translated, and the document may be lang="tr" —
           without this, "MiB" would uppercase with Turkish casing. -->
      <div lang="en" class="mt-1 text-[10px] font-mono uppercase tracking-widest text-slate-500">{unit}</div>
    {/if}
  </div>
  </div>
  <!-- The caption is pulled up into the arc's own gap: that 90° of nothing at
       the bottom is exactly the room a dial's label wants, and leaving it
       empty would waste a sixth of the tile. -->
  <div class="-mt-6 text-center">
    <div lang={labelLang} class="text-[11px] font-mono uppercase tracking-wider text-slate-400">{label}</div>
    {#if sub}
      <div class="text-[10px] font-mono text-slate-600 tabular-nums">{sub}</div>
    {/if}
  </div>
</div>
