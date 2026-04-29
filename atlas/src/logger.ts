// =============================================================================
// Atlas — Interaction Logger (Subtask 9)
// =============================================================================
//
// Renderer-side façade over the preload-exposed `window.atlasAPI.logEvent`.
// Main process receives every event and appends it to:
//
//   <app.getPath("userData")>/atlas-log.json       // JSONL, append-only
//
// On macOS that path resolves to:
//
//   ~/Library/Application Support/Atlas/atlas-log.json
//
// Why JSONL (one JSON object per line) and not a single JSON array?
//   - Append-only: crashes can only lose the tail, not corrupt prior data.
//   - Concurrent sessions can all append without coordination.
//   - Every line is valid JSON on its own — easy to grep/awk/jq across many
//     user sessions.
//
// Main also stamps every event with an ISO-8601 UTC timestamp at write time.
// Call sites in App.tsx don't (and shouldn't) stamp their own — main is the
// single source of truth for clock values so client-side skew can't corrupt
// ordering during analysis.
//
// Event taxonomy (the union below is the schema):
//   query_typed      — what the user typed + which mock query (if any) matched.
//                      Debounced in App.tsx so we log per typing burst, not
//                      per keystroke.
//   result_selected  — a USER-initiated selection change (arrow key / click).
//                      NOT fired for the automatic default-to-row-0 behaviour
//                      that happens on a new query match.
//   result_dwell     — how long the previous selection was viewed before the
//                      user moved on. Fired when selection changes or when
//                      the matched query changes. Filtered below a 200ms
//                      floor to drop keyboard-mashing fly-throughs.
//   open_in_app      — user activated the "Open in …" button or pressed
//                      Enter on a selected row.
// =============================================================================

// Discriminated union of every event we log. New event types should be
// added here — the main-process handler doesn't care about the shape
// beyond `typeof event === "object"`, so adding a type is a pure
// renderer-side change.
export type LogEvent =
  | {
      type: "query_typed";
      query: string;
      matchedQueryId: string | null;
    }
  | {
      type: "result_selected";
      queryId: string;
      resultId: string;
    }
  | {
      type: "result_dwell";
      queryId: string;
      resultId: string;
      dwellMs: number;
    }
  | {
      type: "open_in_app";
      queryId: string;
      resultId: string;
    };

/**
 * Forward an interaction event to the main process for appending to the
 * JSONL interaction log. Timestamp is added by main.
 *
 * Safe to call before preload is ready (e.g. during an HMR reload race):
 * we feature-test `window.atlasAPI.logEvent` first and silently drop the
 * event if missing. Losing a log entry is preferable to crashing the UI
 * with a "cannot read properties of undefined" error.
 */
export function logEvent(event: LogEvent): void {
  // Narrow `window.atlasAPI` without tripping the strict-mode "possibly
  // undefined" checker: the ambient d.ts declares it as always-present,
  // but during the brief preload-boot window it may not be.
  const api = window.atlasAPI as unknown as
    | { logEvent?: (e: unknown) => void }
    | undefined;

  if (!api || typeof api.logEvent !== "function") return;
  api.logEvent(event);
}
