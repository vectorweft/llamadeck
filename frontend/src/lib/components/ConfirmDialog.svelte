<script lang="ts">
  import { confirmState } from '$lib/confirm';
  import { t } from '$lib/i18n.svelte';

  function close(ok: boolean) {
    const req = $confirmState;
    if (!req) return;
    confirmState.set(null);
    req.resolve(ok);
  }

  function onKeydown(e: KeyboardEvent) {
    if (!$confirmState) return;
    if (e.key === 'Escape') { e.preventDefault(); close(false); }
    if (e.key === 'Enter') { e.preventDefault(); close(true); }
  }

  // Focus the confirm button when the dialog opens.
  function autofocus(node: HTMLElement) {
    node.focus();
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if $confirmState}
  <div
    class="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6"
    role="presentation"
    onclick={(e) => { if (e.target === e.currentTarget) close(false); }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-slate-700 bg-slate-900 p-5 space-y-4 shadow-xl"
      role="alertdialog"
      aria-modal="true"
      aria-label={$confirmState.title ?? t('Confirm')}
    >
      {#if $confirmState.title}
        <h2 class="text-base font-semibold text-slate-100">{$confirmState.title}</h2>
      {/if}
      <p class="text-sm text-slate-300 whitespace-pre-line">{$confirmState.message}</p>
      <div class="flex justify-end gap-2 pt-1">
        {#if $confirmState.cancelLabel !== null}
          <button
            onclick={() => close(false)}
            class="rounded bg-slate-700/40 border border-slate-600 px-4 py-1.5 text-sm hover:bg-slate-700/60"
          >{$confirmState.cancelLabel ?? t('Cancel')}</button>
        {/if}
        <button
          use:autofocus
          onclick={() => close(true)}
          class="rounded px-4 py-1.5 text-sm border
                 {$confirmState.danger
                   ? 'bg-rose-900/40 border-rose-800 text-rose-200 hover:bg-rose-900/60'
                   : 'bg-emerald-700/40 border-emerald-600 hover:bg-emerald-700/60'}"
        >{$confirmState.confirmLabel ?? t('Confirm')}</button>
      </div>
    </div>
  </div>
{/if}
