/** @type {import('tailwindcss').Config} */

// All palette hues are routed through CSS variables (see app.css) so the
// whole UI can be re-skinned per theme (data-theme on <html>) without
// touching markup. Variables hold "R G B" triplets to keep /alpha working.
const shades = [100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const varScale = (hue) =>
  Object.fromEntries(shades.map((s) => [s, `rgb(var(--${hue}-${s}) / <alpha-value>)`]));

export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace']
      },
      colors: {
        slate: varScale('slate'),
        emerald: varScale('emerald'),
        amber: varScale('amber'),
        rose: varScale('rose'),
        cyan: varScale('cyan'),
        violet: varScale('violet'),
        sky: { 400: 'rgb(var(--sky-400) / <alpha-value>)' }
      }
    }
  },
  plugins: []
};
