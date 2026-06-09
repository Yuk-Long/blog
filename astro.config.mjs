import { defineConfig } from "astro/config";

export default defineConfig({
  site: "http://8.163.114.59",
  base: "/",
  build: {
    assets: "assets",
  },
  markdown: {
    shikiConfig: {
      theme: "github-light",
    },
  },
});
