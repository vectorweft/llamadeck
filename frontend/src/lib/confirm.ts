import { writable } from 'svelte/store';
import { t } from './i18n.svelte';

export interface ConfirmRequest {
  message: string;
  title?: string;
  /** Style the confirm button as destructive (red). */
  danger?: boolean;
  confirmLabel?: string;
  /** null hides the cancel button (alert-style dialog). */
  cancelLabel?: string | null;
  resolve: (ok: boolean) => void;
}

export const confirmState = writable<ConfirmRequest | null>(null);

/** Themed replacement for window.confirm(). Resolves true on confirm. */
export function confirmDialog(
  message: string,
  opts: Partial<Omit<ConfirmRequest, 'message' | 'resolve'>> = {}
): Promise<boolean> {
  return new Promise((resolve) => confirmState.set({ message, ...opts, resolve }));
}

/** Themed replacement for window.alert(). Always resolves true. */
export function alertDialog(
  message: string,
  opts: Partial<Omit<ConfirmRequest, 'message' | 'resolve' | 'cancelLabel'>> = {}
): Promise<boolean> {
  return new Promise((resolve) =>
    confirmState.set({ message, confirmLabel: t('OK'), ...opts, cancelLabel: null, resolve })
  );
}
