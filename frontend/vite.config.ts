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
        target: "http://localhost:6060",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
