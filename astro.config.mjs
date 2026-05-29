import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://Yuk-Long.github.io',
  base: '/blog/',
  trailingSlash: 'always',
  markdown: {
    shikiConfig: {
      theme: 'github-light',
    },
  },
});