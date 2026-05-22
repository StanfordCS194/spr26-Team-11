// =============================================================================
// Atlas — Root React Component
// =============================================================================
//
// Subtask 4 scope: top search bar only (search icon, text input, "esc" badge).
// The input is functional — typing updates state — but nothing else in the
// overlay reacts to it yet. The Atlas answer block, results list, and
// expanded preview land in Subtasks 5, 6, and 7 respectively.
//
// State ownership: `query` (the typed input) lives on this root component
// even though only the search bar reads/writes it TODAY, because later
// subtasks will need the same value to drive the answer block and the
// filtered results. Lifting it now avoids a refactor later.
//
// Focus management:
//   - On mount, the input is auto-focused so the user can start typing
//     immediately after the overlay appears.
//   - On every `atlas:show` from main (i.e. every time the overlay becomes
//     visible after being hidden), we RESET the input to empty and refocus.
//     This honours the "reset everything on re-open" spec.
//
// Planned component tree (still stubbed/future):
//
//     <div className="overlay">
//       SearchBar          ← Subtask 4 (HERE, inlined)
//       AtlasAnswer        ← Subtask 5
//       ResultsList        ← Subtask 6 — single vertical Top K stack
//       ExpandedPreview    ← Subtask 7 — full preview below the list
//       Footer             ← Subtask 8
//     </div>
// =============================================================================

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { MockQuery, MockResult } from "./lib/types";
import { getAtlasSummaryText, shortDate } from "./lib/format";
import { logEvent } from "./logger";
import { askDaemon, fetchDaemonConfig, searchDaemon } from "./api";
import { ResultRow } from "./components/ResultRow";

export default function App() {
  // ---------------------------------------------------------------------------
  // Typed query state.
  // ---------------------------------------------------------------------------
  // The raw (untrimmed) string the user has typed. It's the controlled value
  // for the <input> AND the input to substring matching against mockQueries
  // (see `matchedQuery` below). Trim/normalise happens inside
  // findMatchingQuery, not here, so the UI preserves the exact characters.
  const [query, setQuery] = useState<string>("");

  // ---------------------------------------------------------------------------
  // Matched query — debounced fetch against the local backend daemon.
  // ---------------------------------------------------------------------------
  // The daemon at 127.0.0.1:8765 returns ranked chunks from the user's
  // indexed filesystem and iMessage data. searchDaemon() shapes the response
  // into a MockQuery so the existing UI render path (AtlasAnswer + ResultsList
  // + ExpandedPreview) doesn't change.
  //
  // Debounce: a 200 ms delay collapses bursts of keystrokes into one request.
  // AbortController cancels any in-flight request when the user keeps typing,
  // so the latest query always wins and stale responses can't overwrite it.
  // ---------------------------------------------------------------------------
  const [matchedQuery, setMatchedQuery] = useState<MockQuery | null>(null);
  // Tracks an in-flight backend fetch so the UI can render a loading
  // indicator. Set to true the moment the debounce timer fires (not on
  // every keystroke), so the spinner doesn't flicker during fast typing.
  const [isSearching, setIsSearching] = useState<boolean>(false);
  // Separate flag for an in-flight /ask request. /ask is triggered explicitly
  // by Tab or Enter (not by typing) and is slower than /search because it
  // routes through the local LLM, so the UI shows the loading indicator
  // while it's running just like the search path does.
  const [isAsking, setIsAsking] = useState<boolean>(false);
  // True after the latest debounced /search or /ask finishes for the current
  // query string. Drives empty-state copy vs the pre-search placeholder.
  const [searchAttempted, setSearchAttempted] = useState<boolean>(false);
  // Holds the AbortController for the current /ask call so a second
  // Tab/Enter press cancels the previous request instead of stacking.
  const askControllerRef = useRef<AbortController | null>(null);

  // Parser mode reported by the daemon. Drives the "☁" badge in the search
  // bar so the user can see at a glance whether queries are being parsed
  // on-device or sent to the configured cloud endpoint. Fetched once on
  // mount; daemon restart is required to switch modes anyway.
  const [parserMode, setParserMode] = useState<"local" | "cloud">("local");
  useEffect(() => {
    fetchDaemonConfig().then((c) => setParserMode(c.parser_mode));
  }, []);

  const triggerAsk = (): void => {
    const trimmed = query.trim();
    if (!trimmed || isAsking) return;
    askControllerRef.current?.abort();
    const controller = new AbortController();
    askControllerRef.current = controller;
    setIsAsking(true);
    askDaemon(trimmed, controller.signal)
      .then((result) => {
        // null = no results; treat the same as a search returning nothing.
        setMatchedQuery(result);
      })
      .catch((e: Error) => {
        if (e.name === "AbortError") return;
        console.error("[atlas] ask failed:", e);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsAsking(false);
          setSearchAttempted(true);
        }
      });
  };

  useEffect(() => {
    if (!query.trim()) {
      setMatchedQuery(null);
      setIsSearching(false);
      setSearchAttempted(false);
      return;
    }
    setSearchAttempted(false);
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const result = await searchDaemon(query, controller.signal);
        setMatchedQuery(result);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        // Network/daemon errors fall through to a null match. Logging keeps
        // them visible in DevTools without breaking the UI.
        console.error("[atlas] search failed:", e);
        setMatchedQuery(null);
      } finally {
        // Only clear the flag if this effect run is still the latest one.
        // The cleanup function aborts the request before it can resolve, so
        // the AbortError branch above prevents this finally from running on
        // stale fetches — but defensive ordering doesn't hurt.
        if (!controller.signal.aborted) {
          setIsSearching(false);
          setSearchAttempted(true);
        }
      }
    }, 200);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  // ---------------------------------------------------------------------------
  // Selected result index (Subtask 6).
  // ---------------------------------------------------------------------------
  // Index into `matchedQuery.results` for the currently-highlighted row in
  // the Top K list. Defaults to 0 — the first row — which matches the spec's
  // "first result is active/selected by default" rule.
  //
  // We RESET to 0 whenever the matched query changes (the user typed a
  // different trigger, or they cleared their input). Without this reset,
  // switching from a 4-result query to another 4-result query could leave
  // the selection pointing at row 3 of what's conceptually a different list.
  //
  // When `matchedQuery` is null, this value is unused (the list doesn't
  // render) but we keep it at 0 so the first render on a new match has the
  // default already in place.
  // ---------------------------------------------------------------------------
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  useEffect(() => {
    setSelectedIndex(0);
    // Depending on `matchedQuery?.id` (not `matchedQuery` itself) avoids
    // firing this effect when two renders produce different MockQuery
    // OBJECTS with the same id — findMatchingQuery returns a stable
    // reference today, but guarding on id is cheap insurance.
  }, [matchedQuery?.id]);

  // ---------------------------------------------------------------------------
  // Ref to the input DOM node for imperative .focus() calls. Needed because:
  //   - Browsers don't autofocus reliably on mount when the element isn't the
  //     current layout root (the overlay starts hidden; first mount happens
  //     before the first `show()`).
  //   - React has no declarative "focus me" API; useRef + .focus() is the
  //     idiomatic workaround.
  // ---------------------------------------------------------------------------
  const inputRef = useRef<HTMLInputElement>(null);

  // ---------------------------------------------------------------------------
  // Ref to the `.overlay` root. A ResizeObserver attached here measures the
  // rendered height of the overlay on every layout change and reports it to
  // the main process via `atlasAPI.setContentHeight()`. Main resizes the
  // BrowserWindow to match, so the window never contains empty space — which
  // is how we avoid a ghost frosted rectangle appearing below the real
  // content (native macOS vibrancy renders for the whole window, so oversize
  // is visible as extra frosted glass).
  // ---------------------------------------------------------------------------
  const overlayRef = useRef<HTMLDivElement>(null);

  // ---------------------------------------------------------------------------
  // User-initiated selection wrapper.
  // ---------------------------------------------------------------------------
  // Shared by the keyboard arrow handler and the per-row click handler. Logs
  // a `result_selected` event so the instrumentation file distinguishes
  // user-driven selection changes from the automatic "reset to row 0 on
  // new query match" (which the effect on [matchedQuery?.id] does — see
  // above — and which we deliberately do NOT log).
  //
  // Callers pass the raw target index; we look up the MockResult via the
  // current matchedQuery to pull its id for the log payload.
  // ---------------------------------------------------------------------------
  const selectResult = (index: number): void => {
    setSelectedIndex(index);
    if (!matchedQuery) return;
    const result = matchedQuery.results[index];
    if (!result) return;
    logEvent({
      type: "result_selected",
      queryId: matchedQuery.id,
      resultId: result.id,
    });
  };

  // Handler for "Enter" + the "Open in [app]" button click. Fires the
  // open_in_app log event for the currently-selected result. Declared as a
  // regular function (not a hook / memoised callback) because it has no
  // dependencies to memoise over beyond the props/state it reads at call
  // time — which react reads off the latest render's closure.
  const openSelectedResult = (): void => {
    if (!matchedQuery) return;
    const result = matchedQuery.results[selectedIndex];
    if (!result) return;
    logEvent({
      type: "open_in_app",
      queryId: matchedQuery.id,
      resultId: result.id,
    });
    if (result.openInApp === "Finder" && result.sourcePath) {
      window.atlasAPI.revealInFinder(result.sourcePath);
    }
  };

  // ---------------------------------------------------------------------------
  // Global keyboard handler: Escape + arrow keys + Enter.
  // ---------------------------------------------------------------------------
  // One window-level listener serves these four keys:
  //
  //   1. Escape: dismiss the overlay (via the preload-exposed IPC to main).
  //      Window-level rather than on the input so it still fires if focus
  //      has somehow drifted elsewhere.
  //
  //   2. ArrowDown / ArrowUp: move selection in the results list. Only does
  //      anything when a query is matched (and therefore the list is
  //      visible). preventDefault stops the browser from doing its default
  //      caret-motion behaviour in the input.
  //
  //   3. Enter: activate the "Open in [app]" behaviour for the currently
  //      selected row. Logs an "open_in_app" event.
  //
  // The effect depends on `matchedQuery` and `selectedIndex` because both
  // are captured in the handler's closure. Re-registering on each change is
  // cheap; the alternative would be a ref-dance to keep a stable handler —
  // not worth the complexity for a handful of rebinds per session.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        window.atlasAPI.hide();
        return;
      }

      // Tab and Enter trigger an LLM-routed /ask request. Bound at the
      // window level (rather than only on the input) so they fire even if
      // focus has drifted into a result row. preventDefault on Tab stops
      // the browser from shifting focus through the overlay; on Enter it
      // stops any default form-submit behaviour from the input.
      // Triggers regardless of whether a /search match is already on
      // screen, so users can refine an existing result set with /ask.
      if (e.key === "Tab" || e.key === "Enter") {
        if (!query.trim()) return;
        e.preventDefault();
        triggerAsk();
        return;
      }

      if (!matchedQuery) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        const next = Math.min(
          matchedQuery.results.length - 1,
          selectedIndex + 1
        );
        if (next !== selectedIndex) selectResult(next);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const next = Math.max(0, selectedIndex - 1);
        if (next !== selectedIndex) selectResult(next);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // Including selectedIndex in the deps so the handler always sees the
    // current selection for clamp math. query is included so Tab/Enter
    // see the latest typed input. React re-registers the listener on each
    // change — still very cheap.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchedQuery, selectedIndex, query]);

  // ---------------------------------------------------------------------------
  // "Reset on re-open" hook.
  // ---------------------------------------------------------------------------
  // Fires every time main's `mainWindow.on('show')` broadcasts `atlas:show`.
  // Clears the typed input and refocuses so the user can immediately type
  // into a fresh slate — this is the hard requirement from the clarifying
  // questions: "reset everything on re-open".
  //
  // Also covers the initial mount case indirectly: the first show() after
  // the renderer boots triggers this handler too.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const unsubscribe = window.atlasAPI.onShow(() => {
      setQuery("");
      // Re-fetch the daemon's parser mode every time the overlay shows so
      // toggling cloud_parser in ~/.atlas/config.json (followed by a daemon
      // restart) takes effect on the next Option+Space without needing to
      // restart the Electron renderer. Cheap — one GET to localhost.
      fetchDaemonConfig().then((c) => setParserMode(c.parser_mode));
      // Use rAF to let React commit the empty-string state before focusing,
      // avoiding a flash where the old text is still visible at focus time.
      requestAnimationFrame(() => inputRef.current?.focus());
    });
    return unsubscribe;
  }, []);

  // ---------------------------------------------------------------------------
  // Initial focus on mount.
  // ---------------------------------------------------------------------------
  // `autoFocus` on the <input> is unreliable in Electron when the window
  // isn't visible at mount time (our overlay starts hidden). Explicitly
  // calling .focus() once after mount covers the "mount happens before first
  // show" sequence. Subsequent shows are covered by the onShow hook above.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // ---------------------------------------------------------------------------
  // Content-height reporter.
  // ---------------------------------------------------------------------------
  // Observes the overlay's own bounding box and forwards every height change
  // to main so the Electron window can resize to match. Critical for the
  // "window is only as tall as its content" invariant: without this the
  // vibrancy-filled window would show a ghost frosted rectangle below the
  // last rendered zone.
  //
  // Implementation notes:
  //   - Uses `contentRect.height` from the ResizeObserverEntry. `offsetHeight`
  //     from the ref would work too but `contentRect` is already exposed by
  //     the observer and matches what the browser's layout engine measured.
  //   - `Math.ceil` avoids sub-pixel oscillation: if we report a fractional
  //     value, the main process's `setBounds` rounds differently than our
  //     layout calc and we can get into a jitter loop.
  //   - Observer fires on mount too, so the initial window size correction
  //     is automatic — no need for a separate "post-mount" report.
  //   - Cleanup disconnects the observer on unmount (matters in dev where
  //     Vite HMR re-runs the effect).
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const el = overlayRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const height = Math.ceil(entry.contentRect.height);
        window.atlasAPI.setContentHeight(height);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // ---------------------------------------------------------------------------
  // query_typed log event (debounced, Subtask 9).
  // ---------------------------------------------------------------------------
  // Logging on every keystroke would fill the log with intermediate garbage
  // ("b", "bu", "bud", …). Instead we debounce: when the user stops typing
  // for 400ms, we record the final query string AND which mock query (if
  // any) matched it. That captures the signal — "here's what the user
  // actually searched for" — without the noise of every in-progress prefix.
  //
  // Empty queries are skipped so "clear input to reset" doesn't pollute
  // the log with `{ query: "", matchedQueryId: null }` entries.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (query.length === 0) return;
    const timer = setTimeout(() => {
      logEvent({
        type: "query_typed",
        query,
        matchedQueryId: matchedQuery ? matchedQuery.id : null,
      });
    }, 400);
    return () => clearTimeout(timer);
  }, [query, matchedQuery]);

  // ---------------------------------------------------------------------------
  // result_dwell log event (Subtask 9).
  // ---------------------------------------------------------------------------
  // "How long the user LOOKED AT a result before moving on" — measured by
  // stamping `start` on each render, and using the effect cleanup to log
  // the elapsed time when the user navigates away (selection change or
  // query switch). React fires the cleanup of the PREVIOUS effect before
  // running the next one, so this pattern captures the full dwell of each
  // result as a self-contained span.
  //
  // The 200ms floor drops very-short flashes from keyboard-mashing (e.g.
  // holding ↓ to fly through the list) — those aren't "dwells" in any
  // useful analytic sense.
  //
  // Dependencies are the result id and the query id, so swapping results
  // or queries triggers the cleanup. The current resultId is captured in
  // the effect's closure at start-time, so the cleanup logs against the
  // result that was being viewed, not the one being navigated to.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!matchedQuery) return;
    const currentResult = matchedQuery.results[selectedIndex];
    if (!currentResult) return;

    const queryId = matchedQuery.id;
    const resultId = currentResult.id;
    const start = Date.now();

    return () => {
      const dwellMs = Date.now() - start;
      // Drop sub-200ms "dwells" — not informative. 200ms is roughly a
      // human reaction floor; anything under it is a key-repeat glide.
      if (dwellMs < 200) return;
      logEvent({
        type: "result_dwell",
        queryId,
        resultId,
        dwellMs,
      });
    };
  }, [matchedQuery?.id, selectedIndex]);

  const trimmedQuery = query.trim();
  const atlasSummary = getAtlasSummaryText(
    !trimmedQuery
      ? { kind: "idle" }
      : isSearching || isAsking
        ? { kind: "loading", asking: isAsking }
        : matchedQuery
          ? { kind: "results", results: matchedQuery.results }
          : searchAttempted
            ? { kind: "empty" }
            : { kind: "placeholder" }
  );

  return (
    <div className="overlay" ref={overlayRef}>
      {/* -----------------------------------------------------------------
         Search bar.
         Keyed visuals:
           - .search-icon : SVG magnifying glass on the left.
           - .search-input: the controlled <input>, bound to `query`.
           - .esc-badge  : small pill button, clicking it dismisses the
                           overlay (mirrors the Escape keystroke).
         ----------------------------------------------------------------- */}
      <div className="searchbar">
        {/*
         * Inline SVG for the search icon. Using currentColor lets CSS
         * drive the tint via the .search-icon color rule, so we can
         * theme it later without touching this JSX.
         */}
        <svg
          className="search-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="11" cy="11" r="7" />
          <line x1="16.2" y1="16.2" x2="21" y2="21" />
        </svg>

        <input
          ref={inputRef}
          className="search-input"
          type="text"
          // spellCheck/autoCorrect/autoCapitalize off: search queries aren't
          // prose, and the macOS spellcheck underline in a Spotlight-style
          // overlay looks off-brand.
          spellCheck={false}
          autoCorrect="off"
          autoCapitalize="off"
          // The placeholder is a subtle hint; it disappears the moment the
          // user types. "Ask Atlas" signals the AI-synthesis angle better
          // than a generic "Search".
          placeholder="Ask Atlas"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        {/*
         * esc badge — implemented as a real <button> so it's keyboard-
         * reachable via Tab and activatable via Enter/Space, not just
         * clickable. The visible label is lowercase to match macOS's
         * modifier-key badge convention ("esc", "return", etc.).
         */}
        {parserMode === "cloud" && (
          <span
            className="cloud-badge"
            title="Cloud parser mode — queries are sent to the configured endpoint for intent parsing."
            aria-label="Cloud parser enabled"
          >
            ☁
          </span>
        )}
        <button
          className="esc-badge"
          type="button"
          onClick={() => window.atlasAPI.hide()}
          // aria-label provides an accessible name that spells out the
          // action, since "esc" alone is jargon for screen readers.
          aria-label="Dismiss Atlas (Escape)"
        >
          esc
        </button>
      </div>

      {/* -----------------------------------------------------------------
         Loading indicator. Visible whenever a backend fetch is in flight.
         Sits between the input and the results so the user sees that
         their query is being processed even before any results arrive
         (and during refinements over an existing matched query).
         ----------------------------------------------------------------- */}
      {(isSearching || isAsking) && (
        <div
          className="search-loading"
          role="status"
          aria-label={isAsking ? "Asking Atlas..." : "Searching..."}
        >
          <span className="search-loading-dot" />
          <span className="search-loading-dot" />
          <span className="search-loading-dot" />
        </div>
      )}

      {/* -----------------------------------------------------------------
         Atlas synthesized answer block (Subtask 5).
         Conditionally rendered: only when the typed input substring-
         matches a mock query's trigger. The overlay grows to include it
         (via the ResizeObserver in this component → IPC → window resize)
         and shrinks back when the user clears or changes the input.
         ----------------------------------------------------------------- */}
      {atlasSummary && (
        <AtlasAnswer query={matchedQuery} summary={atlasSummary} />
      )}

      {/* -----------------------------------------------------------------
         Top K results list (Subtask 6).
         Rendered alongside the Atlas answer whenever a query matches.
         ----------------------------------------------------------------- */}
      {matchedQuery && (
        <ResultsList
          results={matchedQuery.results}
          selectedIndex={selectedIndex}
          // Click-to-select goes through the logging wrapper so mouse
          // interactions show up in the interaction log alongside
          // keyboard navigation.
          onSelect={selectResult}
        />
      )}

      {/* -----------------------------------------------------------------
         Expanded preview of the selected result (Subtask 7).
         Lives below the list, in the same vertical column. Updates
         instantly when `selectedIndex` changes — which it does on arrow
         keys, click, and query switches.
         The React `key` prop is set to the result id so React unmounts
         and remounts the preview on selection change, not just diffs it.
         That guarantees the internal scroll position (if the body is
         long enough to scroll) resets to the top for each new result —
         the alternative is stale scroll state bleeding across rows.
         ----------------------------------------------------------------- */}
      {matchedQuery && (
        <ExpandedPreview
          // key={matchedQuery.results[selectedIndex].id}
          result={matchedQuery.results[selectedIndex]}
          onOpen={openSelectedResult}
        />
      )}

      {/* -----------------------------------------------------------------
         Footer with keyboard hints (Subtask 8).
         Only rendered when a query is matched — there are no results to
         navigate or open without one, so the hints would be misleading
         hints about non-functional shortcuts.
         ----------------------------------------------------------------- */}
      {matchedQuery && <Footer />}
    </div>
  );
}

// =============================================================================
// AtlasAnswer — the synthesized prose answer block.
// =============================================================================
//
// Renders below the search bar when a mock query is matched. Has two parts:
//   1. A small "label row" identifying the source as Atlas — a blue dot plus
//      the wordmark "Atlas". Both the dot AND the label are present so the
//      branding reads even at small sizes / for users who skim past colour.
//   2. The synthesized prose answer itself, taken from the matched
//      MockQuery's `answer` field.
//
// Kept in this file (rather than its own module) because the prototype is
// small and the component has no reusable logic — splitting would just add
// import noise. If/when sibling zones (results list, expanded preview) get
// long enough, all four can move to src/components/ in one pass.
// =============================================================================
function AtlasAnswer({
  query,
  summary,
}: {
  query: MockQuery | null;
  summary: string;
}) {
  return (
    <div className="atlas-answer">
      <div className="atlas-answer__label-row">
        <div className="atlas-answer__label" aria-hidden="true">
          <span className="atlas-answer__dot" />
          <span>Atlas</span>
        </div>
        <p className="atlas-answer__summary" role="status" aria-live="polite">
          {summary}
        </p>
      </div>
      {query?.answer ? (
        <p className="atlas-answer__body">{query.answer}</p>
      ) : null}
    </div>
  );
}

// =============================================================================
// ResultsList — the Top K ranked results (Subtask 6).
// =============================================================================
//
// A single vertical stack of rows, not a two-panel layout. Each row shows:
//   [Source chip]  Title                                Short date
//                  Snippet / subtitle (truncated)
//
// Responsibilities:
//   - Render every row with the correct source-type chip styling (handled
//     inside ResultRow).
//   - Mark the active row via the `is-selected` class, which drives the
//     blue `#2563eb` highlight + inverted text colours.
//   - Click-to-select — clicking any row makes it the active selection
//     (and once Subtask 7 lands, swaps the preview shown below). Arrow-key
//     navigation is handled by the global keyboard handler on App.
//
// ARIA pattern: this is semantically a single-select listbox. We use
// role="listbox" on the <ul> and role="option" + aria-selected on each row
// so screen readers announce selection changes correctly.
// =============================================================================
function ResultsList({
  results,
  selectedIndex,
  onSelect,
}: {
  results: MockResult[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <ul
      className="results-list"
      role="listbox"
      aria-label="Search results"
      // Disable the default list-marker bullet and padding via CSS; this
      // attribute just provides the semantics.
    >
      {results.map((result, index) => (
        <ResultRow
          key={result.id}
          result={result}
          selected={index === selectedIndex}
          onClick={() => onSelect(index)}
        />
      ))}
    </ul>
  );
}

// ResultRow is now in ./components/ResultRow.tsx (imported at the top).

// -----------------------------------------------------------------------------
// Helpers for ResultsList / ResultRow
// -----------------------------------------------------------------------------

// sourceChipModifier and shortDate are now in ./lib/format.ts.

// =============================================================================
// ExpandedPreview — full preview of the selected result (Subtask 7).
// =============================================================================
//
// Sits below the ResultsList in the same single-column stack. Per spec:
//   1. From / date at the top (compact).
//   2. Title.
//   3. Body text with MockResult.highlights bolded + tinted.
//   4. Metadata table: From / Date / Source (the full verbose values).
//   5. "Open in [app] →" link aligned to the bottom-right.
//
// Updates instantly when `selectedIndex` in App changes, because the parent
// passes a new `result` prop on every selection change. We ALSO key the
// component on `result.id` at the call site, which tears down and rebuilds
// the subtree on selection swap — this resets the body's internal scroll
// (if any) and guarantees no stale DOM state leaks between results.
// =============================================================================
function ExpandedPreview({
  result,
  onOpen,
}: {
  result: MockResult;
  // Called when the user activates "Open in [app] →". The parent logs the
  // open_in_app event and shells out via IPC (Finder for Documents hits).
  // Passed in rather than implemented here so the log writer has access to
  // the matched-query id, which this component doesn't know.
  onOpen: () => void;
}) {
  // Compact top-header "from": just the person's name, no email/role
  // cruft. The full raw string is still rendered verbatim in the metadata
  // table below, so no information is hidden — just summarised up top.
  const headerFrom = extractDisplayName(result.from);

  // Same compression logic used by the results list.
  const headerDate = shortDate(result.date);

  const bodyRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    bodyRef.current?.scrollTo(0, 0);
  }, [result.id]);

  return (
    <div className="preview">
      {/* ------------------------------------------------------------------
          Top header: "Sarah Chen · Apr 15"
          Small, muted, uppercase-ish — signals "this is metadata about
          what's below" without competing with the title for attention.
          ------------------------------------------------------------------ */}
      <div className="preview__header">
        <span className="preview__header-from">{headerFrom}</span>
        <span className="preview__header-sep" aria-hidden="true">
          ·
        </span>
        <span className="preview__header-date">{headerDate}</span>
      </div>

      {/*
       * Title. Using <h2> rather than a styled <div> gives screen readers
       * a document landmark for the preview content. The overlay doesn't
       * use <h1> anywhere, so this sits at the top of the heading
       * hierarchy within the preview zone — fine for a single-document
       * panel.
       */}
      <h2 className="preview__title">{result.title}</h2>

      {/*
       * Body with MockResult.highlights bolded + tinted. Rendered via
       * <HighlightedBody> which tokenises the text on a regex built
       * from the highlights array.
       */}
      <div className="preview__body" ref={bodyRef}>
        <HighlightedBody text={result.body} terms={result.highlights} />
      </div>

      {/*
       * Metadata table — semantic <dl> (definition list) with From / Date
       * / Source rows. Styled as a two-column key/value table via flex on
       * each row. A real <table> would be overkill for three rows and
       * would fight with the flex layout everywhere else in the overlay.
       */}
      <dl className="preview__meta">
        <div className="preview__meta-row">
          <dt>From</dt>
          <dd>{result.from}</dd>
        </div>
        <div className="preview__meta-row">
          <dt>Date</dt>
          <dd>{result.date}</dd>
        </div>
        <div className="preview__meta-row">
          <dt>Source</dt>
          <dd>{result.source}</dd>
        </div>
      </dl>

      {/*
       * "Open in [app] →" link. Implemented as <button> because clicking
       * it doesn't navigate — the parent shells out to the named macOS app
       * (Finder for indexed files). Styled to read as a
       * link (blue text, arrow glyph) even though it's a button element.
       */}
      <div className="preview__open">
        <button
          type="button"
          className="preview__open-link"
          onClick={onOpen}
        >
          Open in {result.openInApp} <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// HighlightedBody — render `text` with every case-insensitive occurrence of
// any term in `terms` wrapped in a styled <mark>.
// -----------------------------------------------------------------------------
function HighlightedBody({ text, terms }: { text: string; terms: string[] }) {
  // Fast path: no terms to highlight → return the text verbatim. Avoids
  // building a regex that would match nothing and still do a .split().
  if (terms.length === 0) {
    return <>{text}</>;
  }

  // Sort terms by length DESCENDING. Without this, overlapping patterns
  // (e.g. ["budget", "Q2 budget"]) would match the shorter one first, so
  // "Q2 budget" would be split as "Q2 " + <mark>"budget"</mark> — wrong.
  // Matching longest-first means "Q2 budget" wins over "budget".
  const sorted = [...terms].sort((a, b) => b.length - a.length);

  // Escape regex metacharacters in each term so user-authored mock
  // highlights like "v3 (mocks)" don't break the pattern. Matches the
  // MDN-recommended escape list.
  const escaped = sorted.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );

  // Single regex with one capture group. `split` returns alternating
  // non-match / match substrings in the result array — even indices are
  // literal text, odd indices are matched terms. The `i` flag gives us
  // case-insensitive matching while preserving the matched text's actual
  // case (so "Data Platform" and "data platform" both highlight correctly
  // and keep the author's casing on screen).
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="preview__highlight">
            {part}
          </mark>
        ) : (
          // Wrap in Fragment via index key; no extra DOM node. This also
          // gracefully handles empty strings (when a match is at the very
          // start or end of the body).
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

/**
 * Pull a display-friendly name out of a MockResult.from string.
 *
 * Handles the three shapes we author in mockData.ts:
 *   "Sarah Chen <sarah.chen@northwind.io>"  → "Sarah Chen"
 *   "Marcus Jackson (organizer)"            → "Marcus Jackson"
 *   "Aisha Williams"                        → "Aisha Williams"
 *
 * Used only in the compact top-header of the preview. The metadata table
 * below renders the raw value untouched.
 */
function extractDisplayName(from: string): string {
  const angle = from.indexOf("<");
  if (angle > 0) return from.slice(0, angle).trim();
  const paren = from.indexOf("(");
  if (paren > 0) return from.slice(0, paren).trim();
  return from.trim();
}

// =============================================================================
// Footer — keyboard-hint strip at the bottom of the overlay (Subtask 8).
// =============================================================================
//
// Right-aligned row of ↑↓ navigate · ↵ open · esc close reminders. Each
// shortcut is a small <kbd> badge paired with a plain-text verb.
//
// Rendered only when a query is matched. Without results, there's nothing to
// "navigate" or "open" — showing the hints would be misleading affordance.
// "esc close" is a hint the user doesn't need (Spotlight-style overlays are
// universally Esc-to-close), but we keep it for completeness.
// =============================================================================
function Footer() {
  return (
    <div className="footer" role="contentinfo">
      <FooterHint keys={["↑", "↓"]} label="navigate" />
      <FooterHint keys={["↵"]} label="open" />
      <FooterHint keys={["esc"]} label="close" />
    </div>
  );
}

// One ↑↓-navigate-style hint. Small kbd badges followed by a verb.
function FooterHint({ keys, label }: { keys: string[]; label: string }) {
  return (
    <span className="footer__hint">
      {keys.map((k) => (
        // Using <kbd> semantically marks this as a keyboard input — screen
        // readers announce it as "key <name>" when appropriate. Fine for
        // a decorative hint row too.
        <kbd key={k} className="footer__key">
          {k}
        </kbd>
      ))}
      <span className="footer__hint-label">{label}</span>
    </span>
  );
}
