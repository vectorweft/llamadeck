<script lang="ts">
  let {
    pct,
    base = 'emerald',
    sub = '',
    innerPct = 0,
    size = 112,
    label = ''
  }: {
    /** 0–100; ring turns amber ≥70 and rose ≥90. */
    pct: number;
    /** Hue used below the warning thresholds. */
    base?: 'emerald' | 'cyan' | 'violet';
    /** Small line under the percentage (e.g. "7.9/24 GB"). */
    sub?: string;
    /** Optional secondary inner ring (e.g. active-preset VRAM estimate). */
    innerPct?: number;
    size?: number;
    /** Accessible name. */
    label?: string;
  } = $props();

  const C = 2 * Math.PI * 42;
  const Ci = 2 * Math.PI * 30;
  const strokeByBase = { emerald: 'stroke-emerald-400', cyan: 'stroke-cyan-400', violet: 'stroke-violet-400' } as const;
  const textByBase = { emerald: 'text-emerald-400', cyan: 'text-cyan-400', violet: 'text-violet-400' } as const;
  const clamped = $derived(Math.min(100, Math.max(0, pct)));
  const ring = $derived(clamped >= 90 ? 'stroke-rose-500' : clamped >= 70 ? 'stroke-amber-400' : strokeByBase[base]);
  const txt = $derived(clamped >= 90 ? 'text-rose-400' : clamped >= 70 ? 'text-amber-400' : textByBase[base]);
</script>

<svg viewBox="0 0 100 100" width={size} height={size} class="shrink-0 -rotate-90" role="img" aria-label="{label} {clamped.toFixed(0)}%">
  <circle cx="50" cy="50" r="42" fill="none" class="stroke-slate-800" stroke-width="10" />
  <circle cx="50" cy="50" r="42" fill="none" class={ring} stroke-width="10"
    stroke-dasharray="{(clamped / 100) * C} {C}" stroke-linecap="round" />
  {#if innerPct > 0}
    <circle cx="50" cy="50" r="30" fill="none" class="stroke-slate-900" stroke-width="6" />
    <circle cx="50" cy="50" r="30" fill="none" class="stroke-amber-400" stroke-width="6"
      stroke-dasharray="{(Math.min(100, innerPct) / 100) * Ci} {Ci}" stroke-linecap="round" opacity="0.85" />
  {/if}
  <g transform="rotate(90 50 50)">
    <text x="50" y="48" text-anchor="middle" class="font-mono {txt}" fill="currentColor" font-size="18" font-weight="600">{clamped.toFixed(0)}%</text>
    {#if sub}
      <text x="50" y="62" text-anchor="middle" class="fill-slate-500" font-size="8" font-family="ui-monospace, monospace">{sub}</text>
    {/if}
  </g>
</svg>
