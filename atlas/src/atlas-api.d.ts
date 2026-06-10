// =============================================================================
// Atlas — Ambient types for the preload-exposed `window.atlasAPI`.
// =============================================================================
//
// The preload script (../preload.ts) calls
// `contextBridge.exposeInMainWorld("atlasAPI", atlasAPI)`, which adds a new
// property to the renderer's `window` global at runtime. TypeScript does not
// know that happened, so without this declaration `window.atlasAPI.hide()`
// would be a type error in the renderer.
//
// We keep this file in src/ (not alongside preload.ts) because it describes
// the shape of `window.atlasAPI` AS SEEN BY THE RENDERER. The source of truth
// is the preload module, but the renderer is the consumer.
// =============================================================================

// `export {}` at the bottom would normally convert this into a module,
// turning the `declare global` into a targeted augmentation. But for a
// pure ambient-window augmentation, keeping the whole file script-mode
// (no imports/exports) means every .tsx file picks it up automatically.

interface AtlasAPI {
  // Ask main to hide the overlay. One-way, no return value. Invoked from
  // the renderer's Escape-key handler.
  hide(): void;

  // Subscribe to "overlay is now visible" events from main. The callback
  // runs every time the window transitions hidden → visible (via hotkey).
  // Returns an unsubscribe function — pair it with React useEffect cleanup.
  onShow(callback: () => void): () => void;

  // Report the current rendered height of the overlay content in pixels.
  // Main uses this to resize the BrowserWindow so it matches the content
  // exactly — no ghost vibrancy below the last zone. Called from a
  // ResizeObserver in App.tsx; main clamps the value internally.
  setContentHeight(heightPx: number): void;

  // Forward an interaction event to main for appending to the JSONL log
  // (Subtask 9). Typed as `unknown` at the boundary; the strongly-typed
  // `LogEvent` union lives in src/logger.ts alongside its wrapper that
  // call sites use in practice — direct calls to this API should be rare.
  logEvent(event: unknown): void;

  // Open Finder and select the file at `absolutePath`, or open the folder
  // if `absolutePath` is a directory. No-op when the path does not exist.
  revealInFinder(absolutePath: string): void;

  // Open the iMessage thread for `handle` (E.164 phone number or email).
  openInMessages(handle: string): void;

  // Open a native Finder directory picker and return the selected absolute
  // folder path, or null if the user cancels.
  pickFolder(): Promise<string | null>;
}

interface Window {
  atlasAPI: AtlasAPI;
}
