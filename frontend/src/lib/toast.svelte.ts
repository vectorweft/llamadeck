/** Toast notifications. Call `toast(t('…'), 'success')` after user actions
 * that would otherwise complete silently. Errors that already have an inline
 * spot on the page should stay inline — don't double-report them here. */

export type ToastKind = 'success' | 'error' | 'info';
export interface ToastItem { id: number; kind: ToastKind; text: string }

export const toastState = $state<{ items: ToastItem[] }>({ items: [] });
let nextId = 1;

export function toast(text: string, kind: ToastKind = 'info', ttlMs = 4000): void {
  const id = nextId++;
  toastState.items.push({ id, kind, text });
  setTimeout(() => dismissToast(id), ttlMs);
}

export function dismissToast(id: number): void {
  const i = toastState.items.findIndex((x) => x.id === id);
  if (i >= 0) toastState.items.splice(i, 1);
}
