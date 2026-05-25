// =============================================================================
// Atlas — Shared types
// =============================================================================
//
// These types describe the shape of a query and its results in the UI. They
// were originally co-located with the mock-data file (src/mockData.ts), but
// other modules (api.ts, components, hooks) need to import them without
// pulling in the mock data — so they live here.
//
// Naming note: "Mock*" is preserved for now because the rest of the codebase
// already speaks that vocabulary, and the daemon-backed code path shapes its
// responses into the same contract. If/when the mock data goes away entirely,
// these can be renamed to `Query` / `Result` / `Source` without changing
// behaviour, just import paths.
// =============================================================================

// Source category a result belongs to. Drives the chip colour/label in the
// results list AND the "Open in [app]" link in the expanded preview. Kept as
// a string literal union (rather than enum) so it round-trips cleanly through
// JSON if we ever serialise to the log file or a mock API boundary.
export type SourceType = "Mail" | "Messages" | "Documents" | "Calendar";

// One row in the Top K list AND the data backing the expanded preview below
// it. We use a single shape for both views because the prototype is small
// enough that splitting "list item vs detail" would just add indirection —
// the fields not used by the list (body, highlights, openInApp) are free
// within the preview and ignored elsewhere.
export interface MockResult {
  // Stable identity for selection state and logging. Format: "q<queryId>-r<n>".
  id: string;

  // Which surface this fake result came from. Determines the chip + icon
  // styling and the "Open in __" target app.
  source: SourceType;

  // First line shown in the list row AND as the preview header. Should read
  // like a real subject line / event title / file name / message preview.
  title: string;

  // Secondary line in the list row: snippet / attendee list / file path /
  // preceding-message-author — whatever makes sense for the source type.
  subtitle: string;

  // "From" field surfaced in the preview metadata table. For Calendar this
  // is the organiser; for Messages, the other party; for Documents, the
  // author/owner.
  from: string;

  // Human-readable absolute date. Format chosen to match Apple's Mail/
  // Calendar style: "MMM d, yyyy · h:mm a". Kept as a string (not Date)
  // because this value is only ever displayed, never compared.
  date: string;

  // Full preview body shown under the header of the expanded preview. For
  // Mail this is the email body; for Calendar, the event description/agenda;
  // for Documents, the first paragraph or summary; for Messages, a thread
  // snippet. Plain text — the renderer bolds `highlights` at display time.
  body: string;

  // Terms in `body` that should render bolded/highlighted when the preview
  // is shown. We store the literal terms (rather than offsets) because the
  // renderer does a per-term substring pass, which is robust to edits.
  highlights: string[];

  // Label for the "Open in ___ →" link at the bottom of the preview. Example
  // values per source type:
  //   Mail       → "Mail"
  //   Messages   → "Messages"
  //   Calendar   → "Calendar"
  //   Documents  → "Numbers" | "Pages" | "Figma" | "Code" | "Finder" (varies)
  openInApp: string;

  // Absolute path on disk for filesystem hits (and directory results from
  // find_directory). Used by the Electron main process to reveal the item in
  // Finder. Omitted for Mail / Messages / Calendar and mock-only results.
  sourcePath?: string;
}

// One pre-baked query. Typing any case-insensitive substring of `trigger`
// surfaces this query's answer + results.
export interface MockQuery {
  // Stable identity for logging. Format: "q<n>".
  id: string;

  // The "canonical" phrase a user might type. Matching is substring-based
  // and case-insensitive, so short inputs like "bud" or "onboard" will
  // match "q2 budget" and "onboarding" respectively.
  trigger: string;

  // Atlas's synthesized prose answer for the AtlasAnswer block. Aim: 2–4
  // sentences, conversational, name-checks people and concrete details.
  // Should read like a senior teammate summarizing the situation, not like
  // a search engine echo.
  answer: string;

  // Ranked Top K — 4 entries, spread across ≥3 source types. Order matters:
  // the first entry is the default selection when the query resolves.
  results: MockResult[];
}
