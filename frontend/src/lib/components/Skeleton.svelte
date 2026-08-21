<script lang="ts">
  /** Pulsing placeholder block shown while the first fetch is in flight.
   * Pass Tailwind sizing via `class` (e.g. "h-24 w-full"). */
  let { class: klass = 'h-4 w-full' }: { class?: string } = $props();
</script>

<div class="animate-pulse rounded bg-slate-800/60 {klass}"></div>
