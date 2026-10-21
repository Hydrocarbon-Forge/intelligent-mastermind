import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The built UI is served by the Python program from humanapi/static.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../humanapi/static", emptyOutDir: true },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8080",
      "/health": "http://127.0.0.1:8080",
      "/ws": { target: "ws://127.0.0.1:8080", ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
