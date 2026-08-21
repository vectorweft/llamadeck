/** Tiny bilingual layer. English strings ARE the keys; `tr.ts` maps them to
 * Turkish. A missing entry falls back to English, so an untranslated new
 * string degrades gracefully instead of breaking the UI. Short technical
 * labels (used, free, slots, Kill…) are intentionally not translated —
 * they stay English in both locales (long-standing UI convention).
 *
 * One catch when you add such a label: if CSS uppercases it and it contains a
 * lowercase `i`, give the element `lang="en"`. `text-transform` follows the
 * element's language, and under the document's lang="tr" the browser dots the
 * capital — `live` came out LİVE, `active` came out ACTİVE. */
import { tr } from './tr';

export type Locale = 'en' | 'tr';

function initialLocale(): Locale {
  try {
    return localStorage.getItem('llamadeck-lang') === 'tr' ? 'tr' : 'en';
  } catch {
    return 'en';
  }
}

export const i18n = $state<{ locale: Locale }>({ locale: initialLocale() });

/** Keep <html lang> on the active locale.
 *
 *  Not a formality — CSS `text-transform: uppercase` follows the element's
 *  language, and the app uppercases most of its section headings. Under
 *  lang="en" the browser applies English casing to Turkish text and every
 *  dotted i loses its dot: SISTEM for Sistem, GÜÇ TÜKETIMI for Güç tüketimi,
 *  BOŞTAKI for Boştaki. app.html sets it before first paint; this keeps it
 *  right when the user toggles. */
function syncDocumentLang(l: Locale): void {
  if (typeof document !== 'undefined') document.documentElement.lang = l;
}

export function setLocale(l: Locale): void {
  i18n.locale = l;
  syncDocumentLang(l);
  try { localStorage.setItem('llamadeck-lang', l); } catch { /* ignore */ }
  // Persist to backend settings so card/guide generation follows the UI
  // language. Fire-and-forget: the UI must not block on this.
  fetch('/api/settings', { cache: 'no-store' })
    .then(r => r.ok ? r.json() : null)
    .then(s => {
      if (s && s.ui_language !== l) {
        return fetch('/api/settings', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...s, ui_language: l })
        });
      }
    })
    .catch(() => { /* offline is fine; localStorage already updated */ });
}

/** The language `t(text)` will actually come back in.
 *
 *  For `lang=` on an element whose text CSS uppercases. A string that has no
 *  Turkish entry falls back to English, and English text under lang="tr" gets
 *  Turkish casing — LİVE, ACTİVE, NVİDİA. The reverse is just as wrong:
 *  labelling a translated "enerji" as English yields ENERJI, not ENERJİ. Only
 *  the dictionary knows which happened. */
export function tLang(text: string): Locale {
  return i18n.locale === 'tr' && tr[text] !== undefined ? 'tr' : 'en';
}

/** Translate `text` (English key) into the active locale, then substitute
 * `{param}` placeholders. Use placeholders for any interpolated value so the
 * word order can differ between languages. */
export function t(text: string, params?: Record<string, string | number>): string {
  let s = i18n.locale === 'tr' ? (tr[text] ?? text) : text;
  if (params) {
    for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, String(v));
  }
  return s;
}
