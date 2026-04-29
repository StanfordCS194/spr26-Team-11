// =============================================================================
// Atlas — Backend API Client
// =============================================================================
//
// Thin wrapper around the local FastAPI daemon at http://127.0.0.1:8765.
// Maps the daemon's chunk-shaped response into the renderer's MockQuery /
// MockResult shape so the existing UI can render real data without changes.
//
// The daemon binds to the loopback interface only; the renderer reaches it
// over the network stack but no traffic leaves the machine. CORS is allowed
// for localhost origins on the daemon side (see backend/main.py).
// =============================================================================

import type { MockQuery, MockResult, SourceType } from "./mockData";

const DAEMON_URL = "http://127.0.0.1:8765";

// Mirror of backend/main.py:SearchResult.
interface BackendSearchResult {
  source_type: string;
  source_path: string;
  snippet: string;
  score: number;
}

// -----------------------------------------------------------------------------
// Path helpers — work on POSIX-style paths returned by the daemon.
// -----------------------------------------------------------------------------

function pathBasename(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx >= 0 ? path.slice(idx + 1) : path;
}

function pathDirname(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx >= 0 ? path.slice(0, idx) : "";
}

// -----------------------------------------------------------------------------
// Source-type mapping.
// -----------------------------------------------------------------------------
// The backend speaks "filesystem" / "imessage". The UI's SourceType union
// is "Mail" | "Messages" | "Documents" | "Calendar" — derived from the
// original mock-data design. We collapse filesystem chunks into "Documents"
// and iMessage into "Messages". Mail and Calendar aren't yet indexed.
function mapSource(backend: string): SourceType {
  switch (backend) {
    case "imessage":
      return "Messages";
    case "filesystem":
    default:
      return "Documents";
  }
}

function mapResult(
  r: BackendSearchResult,
  queryId: string,
  idx: number,
  queryTerms: string[],
): MockResult {
  const source = mapSource(r.source_type);
  const isMessage = source === "Messages";
  return {
    id: `${queryId}-r${idx}`,
    source,
    title: isMessage ? r.source_path : pathBasename(r.source_path),
    subtitle: isMessage ? "iMessage thread" : pathDirname(r.source_path),
    from: isMessage ? r.source_path : "Atlas index",
    // The /search endpoint doesn't surface chunk timestamps yet. Empty
    // string keeps the renderer's date column from breaking layout; we'll
    // wire real timestamps through when the backend exposes them.
    date: "",
    body: r.snippet,
    // Pass tokenised query terms so the existing HighlightedBody renderer
    // can bold them in the preview pane without any extra work.
    highlights: queryTerms,
    openInApp: isMessage ? "Messages" : "Finder",
  };
}

// -----------------------------------------------------------------------------
// Public surface.
// -----------------------------------------------------------------------------

/**
 * Hit the daemon's /search endpoint and return the response shaped as a
 * MockQuery so the existing UI can render it. Returns null for empty input
 * or no results.
 *
 * Pass an AbortSignal so the caller (a debounced effect in App.tsx) can
 * cancel a stale request when the user keeps typing.
 */
export async function searchDaemon(
  query: string,
  signal: AbortSignal,
): Promise<MockQuery | null> {
  const trimmed = query.trim();
  if (!trimmed) return null;

  const url = `${DAEMON_URL}/search?q=${encodeURIComponent(trimmed)}&limit=10`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`Daemon /search returned ${res.status}`);
  }
  const results: BackendSearchResult[] = await res.json();
  if (results.length === 0) return null;

  // Each fetch produces a fresh MockQuery id so the App's
  // [matchedQuery?.id] effects (e.g. resetting selectedIndex to 0) fire
  // correctly when the response changes.
  const queryId = `live-${Date.now()}`;
  const queryTerms = trimmed
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);

  return {
    id: queryId,
    trigger: trimmed,
    // The daemon's /search doesn't synthesize a prose answer (that's the
    // /ask endpoint's job and it's slower). Empty string makes the
    // AtlasAnswer block render its blank state until we wire /ask.
    answer: "",
    results: results.map((r, i) => mapResult(r, queryId, i, queryTerms)),
  };
}
