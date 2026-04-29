// =============================================================================
// Atlas — Display formatting helpers
// =============================================================================
//
// Pure presentation utilities used by multiple components in the renderer.
// Kept here (rather than co-located with one component) so that ResultRow,
// ExpandedPreview, and any future renderer can share them without setting up
// import cycles back through App.tsx.
// =============================================================================

import type { SourceType } from "./types";

/**
 * Map a SourceType to the CSS modifier slug used on .result-chip. Kept
 * as a tiny lookup (rather than `.toLowerCase()` at the call site) so any
 * future source type with non-trivial casing (e.g. "iMessage" → "imessage")
 * stays isolated to one place.
 */
export function sourceChipModifier(source: SourceType): string {
  switch (source) {
    case "Mail":
      return "mail";
    case "Messages":
      return "messages";
    case "Documents":
      return "documents";
    case "Calendar":
      return "calendar";
  }
}

/**
 * Trim a long display date like "Apr 15, 2026 · 11:23 AM" down to just
 * "Apr 15" for the right-aligned date cell in a row. The full date+time is
 * still shown in the expanded preview (Subtask 7), so information is never
 * lost — just compressed for list density.
 *
 * Works by slicing up to the first comma. Any date missing a comma (a
 * pathological mock value we haven't authored) falls back to the raw
 * string rather than returning an empty one.
 */
export function shortDate(date: string): string {
  const comma = date.indexOf(",");
  return comma > 0 ? date.slice(0, comma) : date;
}
