// Vite config for the renderer. Electron's main process doesn't touch Vite at all —
// it loads the built output (or the dev server URL) as a regular HTML page.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // React plugin provides Fast Refresh (instant UI updates without losing state)
  // and the automatic JSX runtime so we don't need to import React everywhere.
  plugins: [react()],

  // 'base: "./"' makes all built asset paths relative. This matters for the
  // production build because Electron loads index.html via the file:// protocol,
  // which breaks absolute paths like '/assets/foo.js'. Relative paths work in both
  // http://localhost (dev) and file:// (prod).
  base: "./",

  server: {
    // Bind explicitly to IPv4 127.0.0.1. Vite 6 defaults to IPv6 (::1) only,
    // but `wait-on http://localhost:5173` in package.json resolves via Node's
    // DNS which prefers IPv4 on macOS — so wait-on keeps polling a port that
    // isn't bound on IPv4, and Electron sits idle for tens of seconds.
    // Pinning to 127.0.0.1 makes wait-on's first probe succeed instantly.
    host: "127.0.0.1",
    // Fixed port so the Electron main process knows exactly where to wait.
    // strictPort: fail instead of silently picking a different port — we want a
    // hard error if something's already using 5173 so the user notices.
    port: 5173,
    strictPort: true,
  },

  build: {
    // Matches the path Electron's main process loads in production mode.
    outDir: "dist",
    emptyOutDir: true,
  },

  optimizeDeps: {
    // Declare deps explicitly so Vite skips the import-scanning pass on
    // every cold start. Without this, Vite walks the renderer's import
    // graph to discover what to pre-bundle, which is slow on a freshly
    // installed node_modules where Spotlight may also be reading files.
    include: ["react", "react-dom", "react-dom/client"],
  },
});
