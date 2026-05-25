// =============================================================================
// Atlas — Display formatting helpers
// =============================================================================
//
// Pure presentation utilities used by multiple components in the renderer.
// Kept here (rather than co-located with one component) so that ResultRow,
// ExpandedPreview, and any future renderer can share them without setting up
// import cycles back through App.tsx.
// =============================================================================

import type { MockResult, SourceType } from "./types";

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

/** Inputs for {@link getAtlasSummaryText}. */
export type AtlasSummaryState =
  | { kind: "idle" }
  | { kind: "loading"; asking?: boolean }
  | { kind: "placeholder" }
  | { kind: "empty" }
  | { kind: "results"; results: MockResult[] };

/**
 * One-line status copy for the Atlas label row (right of the blue badge).
 * Returns null when there is nothing to show (empty query).
 */
export function getAtlasSummaryText(state: AtlasSummaryState): string | null {
  switch (state.kind) {
    case "idle":
      return null;
    case "loading":
      return state.asking ? "Asking Atlas..." : "Searching Atlas...";
    case "placeholder":
      return "Search results will appear here.";
    case "empty":
      return "No matching results for your search.";
    case "results":
      return buildResultsSummary(state.results);
  }
}

/** Turn the current Top K list into a short natural-language summary. */
export function buildResultsSummary(results: MockResult[]): string {
  const count = results.length;
  if (count === 0) {
    return "No matching results for your search.";
  }

  const sources = [...new Set(results.map((r) => r.source))];
  const topTitles = results.slice(0, 2).map((r) => r.title);
  const titlePhrase = formatTitleList(topTitles);

  if (count === 1) {
    return `Atlas found 1 matching item: ${results[0].title}.`;
  }

  const onlyDocuments =
    sources.length === 1 && sources[0] === "Documents";
  if (onlyDocuments) {
    return `Atlas found ${count} matching files, including ${titlePhrase}.`;
  }

  const hasDocuments = sources.includes("Documents");
  if (hasDocuments) {
    return `Atlas found ${count} matching items, including ${titlePhrase} and related indexed text.`;
  }

  const sourcePhrase = formatSourceMix(sources);
  return `Atlas found ${count} matching items, including ${titlePhrase}, from ${sourcePhrase}.`;
}

function formatTitleList(titles: string[]): string {
  const filtered = titles.filter(Boolean);
  if (filtered.length === 0) return "your search";
  if (filtered.length === 1) return filtered[0];
  return `${filtered[0]} and ${filtered[1]}`;
}

function formatSourceMix(sources: SourceType[]): string {
  const labels: Record<SourceType, string> = {
    Documents: "indexed files",
    Messages: "messages",
    Mail: "mail",
    Calendar: "calendar",
  };
  const parts = sources.map((s) => labels[s]);
  if (parts.length === 1) return parts[0];
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
  return `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
}
