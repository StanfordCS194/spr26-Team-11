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

import type { MockQuery, MockResult, SourceType } from "./lib/types";

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
// The backend speaks "filesystem" / "imessage" / "gcal". The UI's SourceType
// union is "Mail" | "Messages" | "Documents" | "Calendar". Filesystem chunks
// map to "Documents", iMessage to "Messages", gcal to "Calendar". Mail isn't
// indexed yet.
function mapSource(backend: string): SourceType {
  switch (backend) {
    case "imessage":
      return "Messages";
    case "gcal":
      return "Calendar";
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

  let title: string;
  let subtitle: string;
  let from: string;
  let openInApp: string;
  if (source === "Messages") {
    title = r.source_path;
    subtitle = "iMessage thread";
    from = r.source_path;
    openInApp = "Messages";
  } else if (source === "Calendar") {
    // r.snippet is "title\n\ndescription..." (see backend/gcal.py). The first
    // line is the event title; the rest is body. r.source_path is opaque
    // (gcal://<cal_id>/<event_id>) so we don't surface it.
    const firstNewline = r.snippet.indexOf("\n");
    title = firstNewline >= 0 ? r.snippet.slice(0, firstNewline) : r.snippet;
    subtitle = "Calendar event";
    from = "Google Calendar";
    openInApp = "Calendar";
  } else {
    title = pathBasename(r.source_path);
    subtitle = pathDirname(r.source_path);
    from = "Atlas index";
    openInApp = "Finder";
  }

  return {
    id: `${queryId}-r${idx}`,
    source,
    title,
    subtitle,
    from,
    // The /search endpoint doesn't surface chunk timestamps yet. Empty
    // string keeps the renderer's date column from breaking layout; we'll
    // wire real timestamps through when the backend exposes them.
    date: "",
    body: r.snippet,
    // Pass tokenised query terms so the existing HighlightedBody renderer
    // can bold them in the preview pane without any extra work.
    highlights: queryTerms,
    openInApp,
    ...(source === "Documents" ? { sourcePath: r.source_path } : {}),
  };
}

// -----------------------------------------------------------------------------
// Public surface.
// -----------------------------------------------------------------------------

// Build a fresh MockQuery from a list of backend results. Shared by the
// /search and /ask paths so the result-shaping logic stays in one place.
function buildQuery(
  trimmed: string,
  results: BackendSearchResult[],
  idPrefix: "search" | "ask",
): MockQuery | null {
  if (results.length === 0) return null;
  const queryId = `${idPrefix}-${Date.now()}`;
  const queryTerms = trimmed
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);
  return {
    id: queryId,
    trigger: trimmed,
    // /search doesn't synthesize a prose answer; /ask currently doesn't
    // either (the backend uses the LLM only to route intent, not yet for
    // RAG-style answer synthesis). The AtlasAnswer block will render its
    // blank state in both cases until that endpoint is wired.
    answer: "",
    results: results.map((r, i) => mapResult(r, queryId, i, queryTerms)),
  };
}

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
  return buildQuery(trimmed, results, "search");
}

/**
 * Hit the daemon's /ask endpoint. Same return shape as searchDaemon, but
 * routed through the local LLM so the daemon can interpret intent
 * (find_directory, find_file, source filter, etc.) before retrieving.
 *
 * Slower than /search by ~1-2 s on first call (LLM warm-up) and a few
 * hundred ms thereafter. Triggered by an explicit user action (Tab/Enter)
 * rather than the as-you-type debounce.
 */
export async function askDaemon(
  query: string,
  signal: AbortSignal,
): Promise<MockQuery | null> {
  const trimmed = query.trim();
  if (!trimmed) return null;

  const res = await fetch(`${DAEMON_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: trimmed, limit: 10 }),
    signal,
  });
  if (!res.ok) {
    throw new Error(`Daemon /ask returned ${res.status}`);
  }
  const results: BackendSearchResult[] = await res.json();
  return buildQuery(trimmed, results, "ask");
}

/**
 * Fetch the daemon's parser configuration so the UI can show a small
 * indicator when the cloud parser is enabled. Returns "local" if the
 * endpoint can't be reached — the badge defaults to off, not error.
 */
export async function fetchDaemonConfig(): Promise<{ parser_mode: "local" | "cloud" }> {
  try {
    const res = await fetch(`${DAEMON_URL}/config`);
    if (!res.ok) return { parser_mode: "local" };
    return await res.json();
  } catch {
    return { parser_mode: "local" };
  }
}

/**
 * Start a filesystem indexing job for the selected folder.
 */
export async function indexFilesystemPath(
  folderPath: string,
): Promise<{ status: string; target: string }> {
  const res = await fetch(`${DAEMON_URL}/index/filesystem`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: folderPath }),
  });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((v: { detail?: string }) => v.detail ?? `status ${res.status}`)
      .catch(() => `status ${res.status}`);
    throw new Error(`Failed to start indexing: ${detail}`);
  }
  return res.json();
}
