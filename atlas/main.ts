// =============================================================================
// Atlas — Electron Main Process (Subtask 2: overlay window management)
// =============================================================================
//
// Responsibilities owned by the main process:
//
//   1. Create a frameless, always-on-top "overlay" window sized like macOS
//      Spotlight (~620px wide, ~640px tall). No chrome, no title bar, no
//      traffic lights — the overlay IS the UI.
//
//   2. Hide the app from the Dock and the Cmd+Tab app switcher. Atlas should
//      feel like it belongs to the OS rather than like an app the user
//      opened. The modern way to do this on macOS is
//      `app.setActivationPolicy('accessory')` — it replaces the older
//      `app.dock.hide()` pattern and is the same policy used by menu-bar-only
//      apps (1Password mini, Alfred, Raycast, etc.).
//
//   3. Register the global hotkey `Cmd+Shift+Space` to toggle show/hide. We
//      avoid `Cmd+Space` (taken by Spotlight) and `Ctrl+Cmd+Space` (taken by
//      the emoji picker).
//
//   4. Dismiss-on-Escape is handled by the RENDERER (App.tsx) because we
//      don't want to grab the Escape key globally — other apps rely on it.
//      The renderer sends an "atlas:hide" IPC message which we receive here.
//
//   5. Dismiss-on-click-outside via the window's `blur` event. When focus
//      leaves the overlay (for any reason), we hide it.
//
//   6. Broadcast an "atlas:show" IPC event every time the window becomes
//      visible, so the renderer can reset its state (empty input, no
//      selected result, etc.) — honouring the "reset everything on re-open"
//      spec. We wire the plumbing in Subtask 2; the renderer starts using it
//      in later subtasks.
//
// Note for Subtask 2 specifically: there is intentionally no UI yet. The
// window comes up as a plain white rectangle so we can verify that the
// behaviour — appear centered / hide on Esc / hide on blur / toggle via
// hotkey / invisible in Dock & Cmd+Tab — works end to end.
// =============================================================================

import {
  app,
  BrowserWindow,
  dialog,
  globalShortcut,
  ipcMain,
  screen,
  shell,
} from "electron";
import * as fs from "fs";
import * as path from "path";

// -----------------------------------------------------------------------------
// Dev vs. prod: the renderer URL to load. In dev, main waits for Vite to be up
// on :5173 (see the dev:electron npm script) and loads that URL. In prod, main
// loads the built static index.html from dist/.
// -----------------------------------------------------------------------------
const VITE_DEV_SERVER_URL: string | undefined = process.env.VITE_DEV_SERVER_URL;

// -----------------------------------------------------------------------------
// Overlay dimensions.
//
// The overlay GROWS and SHRINKS with its content, matching real Spotlight.
// Keeping the window at a fixed tall height produced a "ghost frosted
// rectangle" below the visible content, because native macOS vibrancy
// renders for the entire window regardless of what's painted on top — the
// only way to truly hide empty space is to make the window smaller.
//
// Design:
//   WINDOW_WIDTH            — fixed, 620px (Spotlight-width aesthetic).
//   WINDOW_INITIAL_HEIGHT   — ~height of just the search bar. This is what
//                             the overlay looks like on first show and when
//                             nothing is typed. 52px matches the CSS
//                             .searchbar height (56px) minus a hairline to
//                             avoid a 1-px gap at the bottom edge.
//   WINDOW_MAX_HEIGHT       — safety cap. Prevents pathological content
//                             (e.g. a runaway preview body) from growing
//                             the overlay to screen height. The relevant
//                             zone inside would scroll instead.
//
// The renderer sends the measured content height via
// `atlas:set-content-height` (see the ipcMain handler below). Width is
// constant; only height changes, so we keep x and y fixed — the search bar
// stays anchored to the same screen position and new zones unfurl below.
// -----------------------------------------------------------------------------
const WINDOW_WIDTH = 620;
const WINDOW_INITIAL_HEIGHT = 52;
// Bumped from 640 → 720 in Subtask 7: once the expanded preview (header +
// title + body + metadata table + "Open in …" link) lands under a 4-row
// results list, the full stack lands around ~700px. 720 gives a little
// breathing room without feeling oversized on smaller displays. If content
// still exceeds this, individual zones (e.g. the preview body) handle
// their own overflow internally — see .preview__body max-height in
// styles.css.
const WINDOW_MAX_HEIGHT = 720;

// Module-scoped reference. Without this, the BrowserWindow becomes GC-eligible
// the moment createWindow() returns and the window would flicker/vanish.
let mainWindow: BrowserWindow | null = null;

// =============================================================================
// Fade animation machinery (Subtask 8)
// =============================================================================
//
// We animate the overlay's NATIVE WINDOW opacity (win.setOpacity) rather than
// CSS opacity on the renderer root. Two reasons:
//
//   1. The window's native shadow, vibrancy blur, and rounded silhouette all
//      fade together with the content when we animate at the OS level. If we
//      animated only the renderer content, the shadow + frosted frame would
//      pop in/out sharply while the inside fades — uncanny.
//
//   2. CSS opacity transitions on dismissal have to sequence a "fade, then
//      tell main to hide" message pipeline, which adds fragility. Native
//      window opacity keeps the animation and the hide call in one process.
//
// Implementation is a plain setInterval at ~60Hz stepping through an
// ease-in-out cubic. No Electron bundled animation API exists for opacity
// specifically — `win.setBounds(..., true)` animates bounds on macOS but not
// opacity. This hand-rolled loop is ~15 lines and gives us full control.
//
// Race handling: `fadeToken` invalidates any in-flight animation. If a new
// fade starts while an old one is running (e.g. user re-triggers the hotkey
// mid-dismissal), the old loop notices its token no longer matches the live
// one and exits without firing its onComplete. This prevents hide() from
// running after a user has already asked to re-open.
// =============================================================================

// Timings, tuned to feel like macOS Spotlight's own appear/dismiss. Fade-out
// is slightly faster than fade-in so dismissal feels snappy while appearance
// feels deliberate.
const FADE_IN_MS = 80; //120;
const FADE_OUT_MS = 60; //100;

let fadeIntervalId: NodeJS.Timeout | null = null;
let fadeToken = 0;

// Ease-in-out cubic. One curve for both directions keeps the motion feel
// consistent. The tweens are short enough that the difference between
// ease-out-only and ease-in-only is imperceptible.
function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Animate `win.setOpacity` from its current value to `target` over
 * `durationMs`, then call `onComplete` — UNLESS a newer fade has been
 * started in the meantime. Safe to call from anywhere that needs to drive
 * window opacity; always cancels the prior fade.
 */
function fadeWindowOpacity(
  win: BrowserWindow,
  target: number,
  durationMs: number,
  onComplete?: () => void
): void {
  // Mint a new token and cancel any currently-running loop. Any interval
  // that was running belongs to an older token and will exit on its next
  // tick when it notices.
  fadeToken++;
  const myToken = fadeToken;
  if (fadeIntervalId) {
    clearInterval(fadeIntervalId);
    fadeIntervalId = null;
  }

  const startOpacity = win.getOpacity();
  const startTime = Date.now();

  fadeIntervalId = setInterval(() => {
    // Another fade has superseded us — abort without firing onComplete.
    if (myToken !== fadeToken) {
      clearInterval(fadeIntervalId!);
      fadeIntervalId = null;
      return;
    }

    const elapsed = Date.now() - startTime;
    const t = Math.min(1, elapsed / durationMs);
    const eased = easeInOutCubic(t);
    const opacity = startOpacity + (target - startOpacity) * eased;

    // Defensive check for win being destroyed mid-animation.
    if (win.isDestroyed()) {
      clearInterval(fadeIntervalId!);
      fadeIntervalId = null;
      return;
    }

    win.setOpacity(opacity);

    if (t >= 1) {
      // Snap exactly to target to avoid lingering fractional values from
      // floating-point accumulation.
      win.setOpacity(target);
      clearInterval(fadeIntervalId!);
      fadeIntervalId = null;
      if (onComplete) onComplete();
    }
  }, 16); // ~60fps
}

// -----------------------------------------------------------------------------
// Positioning helper — returns (x, y) so the overlay is centered horizontally
// on the display that currently holds the cursor, and sits roughly 1/3 from
// the top (matching macOS Spotlight's "above center" placement).
//
// On a multi-monitor setup, the user expects the overlay to appear on the
// screen they're actively using, NOT always on display #1. So we pick the
// display via cursor position at the moment the overlay is invoked, re-running
// this calculation on every show(). `getPrimaryDisplay()` would be wrong here.
// -----------------------------------------------------------------------------
function getOverlayPosition(): { x: number; y: number } {
  const cursor = screen.getCursorScreenPoint();
  const display = screen.getDisplayNearestPoint(cursor);

  // workArea excludes the menu bar and Dock — the region a window can safely
  // occupy. `bounds` would overlap those.
  const { x: dx, y: dy, width: dw, height: dh } = display.workArea;

  const x = Math.round(dx + (dw - WINDOW_WIDTH) / 2);
  // Place the TOP edge about 1/3 of the way down the usable area. We
  // deliberately use a projected "full-height" reference here (rather than
  // the current window height) so that:
  //   - the top of the search bar lands in the same spot regardless of
  //     whether the overlay is currently tall (results + preview shown) or
  //     short (just the search bar), and
  //   - when content expands downward, the search bar doesn't jump up.
  const y = Math.round(dy + (dh - WINDOW_MAX_HEIGHT) / 3);
  return { x, y };
}

// -----------------------------------------------------------------------------
// createWindow — one-time setup of the overlay BrowserWindow.
// -----------------------------------------------------------------------------
function createWindow(): void {
  const { x, y } = getOverlayPosition();

  mainWindow = new BrowserWindow({
    width: WINDOW_WIDTH,
    height: WINDOW_INITIAL_HEIGHT,
    x,
    y,

    // ----- chrome -------------------------------------------------------
    // `frame: false` removes the OS title bar + traffic lights + border.
    // We do NOT set titleBarStyle alongside it — combining those two has
    // produced weird behaviour (ghost drag regions, phantom traffic lights
    // on focus) in past Electron versions.
    frame: false,

    // ----- resize / min / max --------------------------------------------
    // Overlay is a fixed-size panel. Disabling these prevents the OS from
    // offering resize cursors at the edges or `Cmd+M` minimising it into
    // the Dock (which we don't even have, but defensive either way).
    resizable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,

    // ----- drag ---------------------------------------------------------
    // movable: false — Spotlight isn't draggable; neither are we. Also, with
    // frame:false there is no natural drag surface, so any drag would have
    // to be opted into via CSS `-webkit-app-region: drag`. Safer off.
    movable: false,

    // ----- taskbar / tab switcher ---------------------------------------
    // `skipTaskbar: true` is a no-op on macOS (we handle Dock/CmdTab via
    // setActivationPolicy below) but matters on Windows if we ever port.
    skipTaskbar: true,

    // ----- initial visibility -------------------------------------------
    // Start hidden. The first show() happens when the user presses
    // Cmd+Shift+Space. This also means there's no "paint flash" on launch.
    show: false,

    // -----------------------------------------------------------------------
    // Frosted-glass transparency (Subtask 4).
    // -----------------------------------------------------------------------
    //   - `transparent: true` makes the areas outside our CSS `border-radius`
    //     on `.overlay` render as actual transparency, so the rounded-corner
    //     illusion works (no white triangles in the corners).
    //   - `backgroundColor: '#00000000'` = fully transparent fill. Without
    //     this, Electron can paint a brief opaque frame before our CSS kicks
    //     in, producing a flash on first show.
    //   - `vibrancy: 'hud'` enables the native macOS NSVisualEffectView
    //     behind the window — a real blur of whatever is on the desktop
    //     beneath Atlas. CSS `backdrop-filter: blur()` only blurs other
    //     elements in the page; to blur THROUGH the window to the OS layers
    //     underneath, native vibrancy is the only option. `hud` is the
    //     closest macOS material to the real Spotlight overlay.
    //   - `visualEffectState: 'active'` keeps the blur on even when the
    //     window technically loses focus (important because we hide-on-blur,
    //     but there's a brief window where focus is in transit).
    //   - `hasShadow: true` still works with transparent windows and draws
    //     a drop shadow around the VISIBLE region (i.e., around the
    //     rounded rectangle, not the full window bounds). Subtle but key.
    // -----------------------------------------------------------------------
    transparent: true,
    backgroundColor: "#00000000",
    vibrancy: "hud",
    visualEffectState: "active",
    hasShadow: true,

    alwaysOnTop: true,

    webPreferences: {
      // Preload script runs in the renderer process but has access to Node.
      // It uses contextBridge to expose a tiny, sandboxed `window.atlasAPI`
      // surface (hide + onShow). The compiled path lives next to main.js in
      // dist-electron/.
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // `setAlwaysOnTop(true, 'floating', 1)` upgrades the z-order to macOS's
  // "floating" level with a small additional offset. This keeps the overlay
  // above normal windows while sitting BELOW critical OS surfaces like
  // screen-saver and full-screen presentations. Electron's default
  // `alwaysOnTop: true` uses a lower level that other alwaysOnTop windows
  // (e.g. a screen-share toolbar) could outrank.
  mainWindow.setAlwaysOnTop(true, "floating", 1);

  // Make the overlay appear even when the user is currently on a different
  // macOS Space or in a fullscreen app. Without `visibleOnFullScreen: true`,
  // hitting the hotkey while watching a fullscreen video would switch Space
  // instead of overlaying — a jarring break from Spotlight-like behaviour.
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // Load the renderer.
  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
    // Detached DevTools are opt-in: opening them adds ~1–2s to startup and
    // visual clutter you don't usually need. Set OPEN_DEVTOOLS=1 when you
    // actually want to inspect the renderer.
    if (process.env.OPEN_DEVTOOLS === "1") {
      mainWindow.webContents.openDevTools({ mode: "detach" });
    }
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }

  // ---------------------------------------------------------------------
  // Click-outside dismissal — window 'blur' fires whenever focus leaves.
  //
  // Guarded by the DevTools check: when we open detached DevTools in dev,
  // clicking into them triggers blur on the overlay. Without this guard,
  // the overlay would vanish the instant you tried to inspect it. In
  // production there's no DevTools so the guard is a no-op.
  // ---------------------------------------------------------------------
  mainWindow.on("blur", () => {
    if (!mainWindow) return;
    if (mainWindow.webContents.isDevToolsOpened()) return;
    // Funnel through hideOverlay() so the dismissal fade plays on
    // click-outside, not just on Escape/hotkey. Keeps all three dismissal
    // paths (blur / Escape / hotkey) visually identical.
    hideOverlay();
  });

  // Fire "atlas:show" to the renderer every time the overlay (re-)appears.
  // The renderer uses this to reset its transient state — per the spec,
  // each invocation of the overlay should be a clean slate (empty input,
  // no selected result, no dwell timers). Wiring it up in Subtask 2 so
  // later subtasks can just consume the event.
  mainWindow.on("show", () => {
    mainWindow?.webContents.send("atlas:show");
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// -----------------------------------------------------------------------------
// showOverlay — reposition on the active display, then show + focus.
// -----------------------------------------------------------------------------
function showOverlay(): void {
  if (!mainWindow) return;
  const win = mainWindow;

  const { x, y } = getOverlayPosition();
  // Always RESET to the initial (search-bar-only) height on show. If the
  // last time the overlay was used it grew to show results + preview, we
  // don't want it to flash back in at that size before the renderer's
  // state reset has a chance to shrink it. Starting small and growing
  // under content control is what makes the overlay feel responsive.
  win.setBounds({
    x,
    y,
    width: WINDOW_WIDTH,
    height: WINDOW_INITIAL_HEIGHT,
  });

  // Fade-in path. Two cases:
  //   (a) window is currently hidden → start opacity at 0, show, fade to 1.
  //   (b) window is visible but we're mid-fade-out → reverse course, fade
  //       back to 1 from whatever the current opacity is.
  if (!win.isVisible()) {
    win.setOpacity(0);
    win.show();
  }
  // Explicit focus is needed because show() alone doesn't always activate
  // the window — especially on macOS when the activation policy is
  // 'accessory' (no Dock icon). Without focus the search input can't
  // receive keystrokes.
  win.focus();

  fadeWindowOpacity(win, 1, FADE_IN_MS);
}

// -----------------------------------------------------------------------------
// hideOverlay — fade to 0, then hide. All dismissal paths (Escape IPC,
// blur, hotkey-toggle) funnel through here so the animation is uniform.
// -----------------------------------------------------------------------------
function hideOverlay(): void {
  if (!mainWindow || !mainWindow.isVisible()) return;
  const win = mainWindow;

  fadeWindowOpacity(win, 0, FADE_OUT_MS, () => {
    // Only runs if our token wasn't superseded by a reopen. Safe to hide.
    if (win.isDestroyed()) return;
    win.hide();
    // Restore opacity to 1 so the next show() starts with a known-good
    // value. (Our show path sets it back to 0 before fading in, but only
    // when the window was hidden — this keeps the two paths decoupled.)
    win.setOpacity(1);
  });
}

// -----------------------------------------------------------------------------
// toggleOverlay — flip between visible and hidden. Bound to the hotkey.
// -----------------------------------------------------------------------------
function toggleOverlay(): void {
  if (!mainWindow) return;
  if (mainWindow.isVisible()) {
    hideOverlay();
  } else {
    showOverlay();
  }
}

// -----------------------------------------------------------------------------
// App lifecycle wiring
// -----------------------------------------------------------------------------
app.whenReady().then(() => {
  // Hide the app from the Dock and the Cmd+Tab switcher. The 'accessory'
  // activation policy tells macOS "this is a helper-style app" — it can
  // still take focus (required for typing into the search input), but won't
  // show a Dock icon or participate in the app switcher. Has no effect on
  // non-Mac platforms, so the guard is defensive more than necessary.
  if (process.platform === "darwin") {
    app.setActivationPolicy("accessory");
  }

  createWindow();

  // Register the global toggle hotkey. `register` returns false if another
  // app already owns the combo. We surface that clearly so a developer
  // cloning the repo can diagnose it (e.g. if they have Raycast on
  // Cmd+Shift+Space). `Cmd+Space` is Spotlight; `Ctrl+Cmd+Space` is the
  // system emoji picker — both avoided.
  const registered = globalShortcut.register(
    // "CommandOrControl+Shift+Space",
    "Option+Space",
    toggleOverlay
  );
  if (!registered) {
    console.error(
      "[atlas] Failed to register Cmd+Shift+Space — another app may own it."
    );
  }

  // IPC handler for renderer-side Escape → hide the overlay. Keeping the
  // Escape handling in the renderer (rather than as a second global
  // shortcut) means other apps' Escape behaviour is unaffected.
  ipcMain.on("atlas:hide", () => {
    // Route through hideOverlay() so Escape dismissal also plays the
    // fade-out animation rather than vanishing instantly.
    hideOverlay();
  });

  // -------------------------------------------------------------------------
  // IPC handler: reveal a filesystem path in Finder.
  // -------------------------------------------------------------------------
  // The renderer passes an absolute path from the daemon's source_path field.
  // Files: shell.showItemInFolder selects the file in its parent folder.
  // Directories (e.g. find_directory hits): shell.openPath opens the folder.
  // -------------------------------------------------------------------------
  ipcMain.on("atlas:reveal-in-finder", (_event, rawPath: unknown) => {
    if (typeof rawPath !== "string" || rawPath.length === 0) return;
    if (process.platform !== "darwin") {
      console.warn("[atlas] reveal-in-finder is only supported on macOS");
      return;
    }
    try {
      if (!fs.existsSync(rawPath)) {
        console.error("[atlas] reveal path does not exist:", rawPath);
        return;
      }
      const stat = fs.statSync(rawPath);
      if (stat.isDirectory()) {
        void shell.openPath(rawPath);
      } else {
        shell.showItemInFolder(rawPath);
      }
    } catch (err) {
      console.error("[atlas] reveal-in-finder failed:", err);
    }
  });

  // -------------------------------------------------------------------------
  // IPC handler: open Finder folder picker for filesystem indexing.
  // -------------------------------------------------------------------------
  // Returns the chosen absolute folder path to the renderer, or null when
  // cancelled. The renderer then calls the daemon's /index/filesystem route.
  // -------------------------------------------------------------------------
  ipcMain.handle("atlas:pick-folder", async () => {
    const focusedWindow = BrowserWindow.getFocusedWindow() ?? mainWindow;
    const result = focusedWindow
      ? await dialog.showOpenDialog(focusedWindow, {
          title: "Choose folder to index",
          properties: ["openDirectory", "createDirectory"],
          buttonLabel: "Index Folder",
        })
      : await dialog.showOpenDialog({
          title: "Choose folder to index",
          properties: ["openDirectory", "createDirectory"],
          buttonLabel: "Index Folder",
        });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0] ?? null;
  });

  // -------------------------------------------------------------------------
  // IPC handler: interaction log events (Subtask 9).
  // -------------------------------------------------------------------------
  // The renderer sends one event per user interaction (query typed, result
  // selected, dwell on a result, "Open in ..." activated). We stamp it with
  // an ISO timestamp and APPEND one JSON object per line (JSONL) to the
  // atlas-log.json file in the app's userData directory.
  //
  // Why JSONL and not a single JSON array?
  //   - Append-only is safe: if the process crashes mid-write, we lose at
  //     most a partial line — never corrupt the whole file. A single JSON
  //     array would need to be read, parsed, pushed to, and rewritten each
  //     time, which is slow and fragile.
  //   - Multiple sessions can append concurrently without coordination.
  //   - Every line is valid JSON on its own — easy to grep / awk / jq
  //     across many sessions for post-hoc analysis.
  //
  // Path: app.getPath("userData") resolves on macOS to
  //   ~/Library/Application Support/Atlas/atlas-log.json
  // The directory exists as soon as Electron has booted; no mkdir needed.
  //
  // Errors are logged to stdout rather than surfaced back to the renderer —
  // logging is best-effort, and a failed log write shouldn't interrupt the
  // user's interaction with the overlay.
  // -------------------------------------------------------------------------
  const logFilePath = path.join(app.getPath("userData"), "atlas-log.json");

  ipcMain.on("atlas:log", (_event, rawEvent: unknown) => {
    // Defensive: ignore anything that doesn't look like an object payload.
    if (rawEvent === null || typeof rawEvent !== "object") return;

    // Stamp with ISO-8601 UTC for analysis stability. Timestamp is added
    // server-side (main) so clock skew / tampering in the renderer can't
    // corrupt the record's ordering.
    const stamped = {
      timestamp: new Date().toISOString(),
      ...(rawEvent as Record<string, unknown>),
    };

    const line = JSON.stringify(stamped) + "\n";

    // Async append — does NOT block the main-process event loop, so it
    // won't stall show/hide animations or IPC on other channels if the
    // disk is slow.
    fs.appendFile(logFilePath, line, (err) => {
      if (err) {
        console.error("[atlas] log write failed:", err);
      }
    });
  });

  // Log the file path once at startup so a developer can find it quickly.
  // In dev, this prints to the terminal that started Electron; in a
  // packaged build nobody sees it, but that's fine.
  console.log(`[atlas] interaction log: ${logFilePath}`);

  // -------------------------------------------------------------------------
  // IPC handler: content height reporting from the renderer.
  // -------------------------------------------------------------------------
  // The renderer observes its own overlay element via ResizeObserver and
  // sends the measured pixel height here on every change. We resize the
  // window so it's exactly as tall as the overlay's rendered content, which
  // is how we avoid the "ghost frosted area" below the visible content —
  // native macOS vibrancy fills the whole window, so any window bigger than
  // the content would still blur the desktop in the gap.
  //
  // We clamp to [WINDOW_INITIAL_HEIGHT, WINDOW_MAX_HEIGHT]:
  //   - Lower bound keeps a runaway observer (or a collapsed DOM) from
  //     shrinking the window below the search bar and making the overlay
  //     unusable. WINDOW_INITIAL_HEIGHT is the correct floor because the
  //     search bar is always present.
  //   - Upper bound prevents a pathologically long preview body from
  //     pushing the overlay to fill the whole screen. When the cap kicks
  //     in, the relevant zone inside the overlay handles its own overflow
  //     scroll (see zone styles in styles.css).
  //
  // We keep x and y pinned — only height changes — so the top edge of the
  // search bar stays anchored and content unfurls downward.
  // -------------------------------------------------------------------------
  ipcMain.on("atlas:set-content-height", (_event, rawHeight: unknown) => {
    if (!mainWindow) return;
    // Defensive coercion: preload sends a number, but an untyped IPC
    // boundary is cheap to harden against a future refactor.
    if (typeof rawHeight !== "number" || !Number.isFinite(rawHeight)) return;

    const clamped = Math.max(
      WINDOW_INITIAL_HEIGHT,
      Math.min(WINDOW_MAX_HEIGHT, Math.round(rawHeight))
    );

    const current = mainWindow.getBounds();
    // Skip the setBounds round-trip if we're already at the target height —
    // avoids IPC storms during React re-renders that don't actually change
    // layout (e.g. unrelated state updates).
    if (current.height === clamped) return;

    mainWindow.setBounds({
      x: current.x,
      y: current.y,
      width: WINDOW_WIDTH,
      height: clamped,
    });
  });

  // macOS: if all BrowserWindows go away and somehow come back (e.g. after
  // a crash-restart), recreate. Edge-casey for an always-hidden overlay app
  // but cheap insurance.
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Clean up the global shortcut on quit so we don't leave it registered in
// the OS (which would block other apps from claiming it until reboot).
// Also cancel any in-flight fade so the interval doesn't keep the process
// alive past its natural quit.
app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  if (fadeIntervalId) {
    clearInterval(fadeIntervalId);
    fadeIntervalId = null;
  }
});

// NOTE: we deliberately do NOT quit on `window-all-closed`. The overlay is
// hidden (not closed) on dismiss, so `window-all-closed` shouldn't fire in
// normal use. Even if it did, Atlas's whole value proposition is "always
// one hotkey away" — quitting would break that promise. A future menu-bar
// icon (out of scope here) will expose an explicit Quit action.
