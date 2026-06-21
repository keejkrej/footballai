import { defineConfig } from "vite-plus";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  root: "apps/web",
  plugins: [
    svelte({
      compilerOptions: {
        runes: ({ filename }) =>
          filename.split(/[/\\]/).includes("node_modules") ? undefined : true,
      },
    }),
  ],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
