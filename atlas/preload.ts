// =============================================================================
// Atlas — Preload Script (renderer ↔ main IPC bridge)
// =============================================================================
//
// Preload scripts are a special Electron concept: they run in the renderer
// PROCESS, but in a special context that has access to Node's `require` and
// to Electron's `contextBridge` / `ipcRenderer` — none of which the renderer
// itself can touch when `contextIsolation: true` (which we require for
// security). The preload's job is to expose a small, carefully-designed API
// from main onto the renderer's `window` global.
//
// Why a bridge at all? Because two overlay behaviours need the renderer to
// talk to main:
//
//   1. "Escape was pressed → please hide the window" — the renderer owns the
//      keystroke (so it doesn't fight with other apps' Escape behaviour), but
//      only main can call `win.hide()`.
//
//   2. "The window just became visible → reset local state" — this fires FROM
//      main TO the renderer. The renderer subscribes via `onShow` and clears
//      its input / selection / timers. Enforces the "reset everything on
//      re-open" spec without needing to reload the whole page.
//
// The exposed surface is DELIBERATELY tiny. Every additional API is another
// seam for a compromised renderer to exploit, so we only add what's strictly
// needed. No direct `ipcRenderer` access, no filesystem, no shell.
// =============================================================================

import { contextBridge, ipcRenderer, IpcRendererEvent } from "electron";

// Shape of the API we expose on window.atlasAPI. Mirrored by the `.d.ts`
// file in src/ so renderer code typechecks against it.
const atlasAPI = {
  // Dismiss the overlay. Called from the renderer's Escape-key handler.
  // One-way send (no response needed).
  hide(): void {
    ipcRenderer.send("atlas:hide");
  },

  // Subscribe to "overlay is now visible again" events from main. Returns
  // an unsubscribe function so React components can clean up in useEffect
  // without leaking listeners across renders / hot-reloads.
  onShow(callback: () => void): () => void {
    const handler = (_event: IpcRendererEvent) => callback();
    ipcRenderer.on("atlas:show", handler);
    return () => ipcRenderer.removeListener("atlas:show", handler);
  },

  // Tell main how tall the rendered overlay content currently is. Main
  // resizes the BrowserWindow to match, so the window never contains
  // invisible space (which would still render frosted vibrancy and look
  // wrong as a ghost rectangle below the content).
  //
  // Caller: a ResizeObserver in App.tsx, firing once per overlay layout.
  // Main clamps the value to [initial, max] internally so the renderer
  // doesn't have to worry about validating its own number.
  setContentHeight(heightPx: number): void {
    ipcRenderer.send("atlas:set-content-height", heightPx);
  },

  // Forward an interaction event to main for appending to the JSONL
  // interaction log (Subtask 9). We accept `unknown` at the IPC boundary
  // because the concrete discriminated-union type (LogEvent) lives in
  // src/logger.ts — a renderer-side module the preload doesn't import.
  // Main stamps the event with a timestamp before writing, so callers
  // here don't need to.
  logEvent(event: unknown): void {
    ipcRenderer.send("atlas:log", event);
  },
};

// `exposeInMainWorld` is the ONLY safe way to put something on window when
// contextIsolation is on — it serialises the API across the context boundary
// so the renderer can't reach back into preload internals.
contextBridge.exposeInMainWorld("atlasAPI", atlasAPI);
