import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://Yuk-Long.github.io',
  base: '/blog/',
  markdown: {
    shikiConfig: {
      theme: 'github-light',
    },
  },
});