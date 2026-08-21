<script lang="ts">
  import { fly } from 'svelte/transition';
  import { dismissToast, toastState } from '$lib/toast.svelte';

  const styles: Record<string, string> = {
    success: 'border-emerald-700 bg-emerald-950/90 text-emerald-100',
    error: 'border-rose-800 bg-rose-950/90 text-rose-100',
    info: 'border-slate-700 bg-slate-900/95 text-slate-200'
  };
  const dots: Record<string, string> = {
    success: 'bg-emerald-400',
    error: 'bg-rose-400',
    info: 'bg-sky-400'
  };
</script>

{#if toastState.items.length > 0}
  <div class="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 max-w-sm" role="status" aria-live="polite">
    {#each toastState.items as item (item.id)}
      <div
        transition:fly={{ x: 40, duration: 180 }}
        class="flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-sm shadow-xl backdrop-blur {styles[item.kind]}"
      >
        <span class="mt-1.5 h-2 w-2 shrink-0 rounded-full {dots[item.kind]}"></span>
        <span class="min-w-0 flex-1 break-words">{item.text}</span>
        <button
          onclick={() => dismissToast(item.id)}
          class="shrink-0 text-slate-500 hover:text-slate-200 leading-none pt-0.5"
          aria-label="Dismiss"
        >✕</button>
      </div>
    {/each}
  </div>
{/if}
