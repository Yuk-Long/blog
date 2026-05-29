import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sisidashemowang.github.io',
  base: '/blog',
  markdown: {
    shikiConfig: {
      theme: 'github-light',
    },
  },
});