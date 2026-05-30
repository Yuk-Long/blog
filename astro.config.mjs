import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://Yuk-Long.github.io",
  base: "/blog/",
  vite: {
    build: {
      assetsDir: "assets",
    },
  },
  markdown: {
    shikiConfig: {
      theme: "github-light",
    },
  },
});