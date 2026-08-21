<script lang="ts">
  let { adopted = false, pid = null }: { adopted?: boolean; pid?: number | null } = $props();
</script>

{#if adopted}
  <span class="inline-flex items-center gap-1.5 rounded-full bg-amber-950/50 border border-amber-800 px-2 py-0.5 text-xs font-mono text-amber-400">adopted{pid != null ? ` · PID ${pid}` : ''}</span>
{:else}
  <span class="inline-flex items-center gap-1.5 rounded-full bg-emerald-950/50 border border-emerald-800 px-2 py-0.5 text-xs font-mono text-emerald-400">running{pid != null ? ` · PID ${pid}` : ''}</span>
{/if}
