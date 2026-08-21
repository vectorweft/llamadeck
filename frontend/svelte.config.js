import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: '../backend/lld/static',
      assets: '../backend/lld/static',
      fallback: 'index.html',
      precompress: false,
      strict: true
    }),
    prerender: { entries: [] }
  }
};

export default config;
