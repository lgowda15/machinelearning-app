/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Dev-only: lets the browser client call the FastAPI backend as if
      // same-origin, without the backend needing CORS config. Production
      // gets the backend URL baked in at build time (ARCHITECTURE.md §11).
      "/api": {
        target: "http://localhost:8050",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    // svg2pdf.js's package.json has no "exports" map, so Vitest's default
    // Node-style resolution for externalized deps picks its CJS/UMD "main"
    // (which expects a global `jspdf`) instead of its ESM "module" build.
    // Forcing it through Vite's own resolver (which respects "module")
    // fixes it -- see renderReportPdf.ts, the only place that imports it.
    server: {
      deps: {
        inline: ["svg2pdf.js"],
      },
    },
  },
});
