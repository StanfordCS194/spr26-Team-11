# Atlas

A locally-run, privacy-first AI search overlay for macOS. Press a global hotkey, type a question, get ranked results from your filesystem, iMessages, and Google Calendar. **By default, all retrieval, ranking, and query parsing happen on-device — no data leaves your machine.** An optional cloud parser (Groq-compatible) can be enabled for users who prefer it; see "Optional: cloud query parser" below. Google Calendar pulls events through Google's API but stores them only in the local index — see Step 2.

**Team SSH**: Andrew Grant, Ji Qi Ni, Nick Donaldson, Olatayo Sobomehin, Ose Okhihan.

---

## 1. Frontend implementation

The frontend is a macOS Spotlight-style overlay window built on a standard web stack wrapped in a native shell.

### Stack

- **Electron** — provides the native macOS window, global keyboard shortcut, and frosted-glass vibrancy. Runs the renderer (a Chromium tab) and a Node.js main process side-by-side, with a small preload bridge for IPC.
- **Vite** — dev server with hot-module reload for the React renderer. Bundles to static assets for production.
- **React 18 + TypeScript** — declarative UI, typed props, hooks for state and effects.
- **Plain CSS** — single stylesheet, no preprocessor or CSS-in-JS. Apple system font, semi-transparent layers tuned to sit on top of native vibrancy.

### Architecture

```
atlas/
├── main.ts             Electron main process — window, hotkey, IPC, log writer
├── preload.ts          Tiny security bridge: exposes window.atlasAPI to renderer
├── index.html          Single-mount React shell + CSP
└── src/
    ├── main.tsx        React entry — mounts <App />
    ├── App.tsx         Top-level state (query, matchedQuery, selectedIndex), keyboard handler
    ├── api.ts          HTTP client for the daemon: searchDaemon() and askDaemon()
    ├── styles.css      Global stylesheet
    ├── logger.ts       Typed wrapper around the IPC log channel
    ├── mockData.ts     Original prototype data, retained as a fallback shape contract
    ├── lib/
    │   ├── types.ts    MockQuery / MockResult / SourceType
    │   └── format.ts   sourceChipModifier, shortDate
    └── components/
        └── ResultRow.tsx
```

The Electron main process owns the BrowserWindow (frameless, transparent, `vibrancy: "hud"`, frosted-glass border). It registers `Cmd+Shift+Space` as a global toggle and listens for blur events to dismiss the overlay on click-outside. The React renderer talks to the FastAPI daemon over HTTP (CORS-allowed loopback).

### Improvements made on top of the base UI

- **Debounced live search.** Typing fires `/search` after 200 ms of no keystrokes — collapses bursts of input into a single request, with `AbortController` cancelling stale requests.
- **Tab / Enter triggers `/ask`.** Live typing stays on the cheap `/search` path; an explicit Tab or Enter routes through the local LLM via `/ask`, which interprets intent (`find_directory`, `find_file`, source filter) before retrieving.
- **Loading indicator.** Three pulsing dots appear when a fetch is in flight. The dots are gated on the debounce timer firing, so they don't flicker during fast typing.
- **Scrollable results list.** The list caps at 5 visible rows (~280 px) with a custom thin overlay scrollbar matching macOS conventions. Larger result sets scroll inside the list while the answer block, expanded preview, and footer stay pinned.
- **Synchronous scroll-into-view.** Arrow-key navigation past the visible boundary triggers `scrollIntoView({ block: "nearest" })` from a `useLayoutEffect`, ensuring the new highlight and the new scroll position land in the same paint frame — no "blue flash" at the wrong position.
- **In-place `ExpandedPreview` updates.** The preview component stays mounted across selection changes; body scroll resets via a ref instead of a remount. Avoids tearing down DOM on every arrow press.
- **Auto-resizing window.** A `ResizeObserver` on the overlay reports the rendered height back to the main process via IPC, which calls `setBounds()` to grow/shrink the window to fit the content.

---

## 2. Backend implementation

The backend is a Python FastAPI daemon that indexes local data, embeds it for semantic search, and serves ranked results over HTTP.

### Stack

- **FastAPI + Uvicorn** — async REST framework. Serves `/search`, `/ask`, `/index/*`, `/status`, `/clear`. Lifespan handler eager-loads models at startup.
- **fastembed** — ONNX-Runtime embeddings (`all-MiniLM-L6-v2`, 384-dim). Replaced sentence-transformers for ~8× faster cold start.
- **SQLite + numpy + scipy** — lightweight vector store. SQLite holds chunks, blobs, and metadata; numpy provides cosine search; scipy sparse matrices accelerate path-keyword scoring.
- **fastembed cross-encoder** — `Xenova/ms-marco-MiniLM-L-6-v2` for two-stage re-ranking.
- **mlx-lm + Qwen2.5-0.5B-Instruct (4-bit)** — local LLM for query parsing on Apple Silicon. ~280 MB on disk. Output is constrained to a valid JSON schema via the `outlines` library, so the regex-extract fallback rarely fires.
- **(optional) cloud parser** — an OpenAI-compatible chat-completions endpoint (default Groq + `llama-3.1-8b-instant`, free tier) replaces the local Qwen when enabled via `cli.py config set-cloud-parser true`. API key stored in the macOS Keychain via `keyring`. Off by default — when off, no network calls leave the daemon.
- **Typer + Rich** — CLI framework and pretty terminal output.

### Architecture

```
backend/
├── main.py        FastAPI app, lifespan warmup, CORS, BackgroundTasks
├── cli.py         Thin HTTP client over the daemon + daemon lifecycle
├── config.py      Constants (paths, model names, daemon URL)
├── ingest.py      File walker, parser, chunker, path-context extractor (filesystem + iMessage)
├── gcal.py        Google Calendar OAuth, event fetch, chunking, index entry point
├── embed.py       fastembed wrapper (lazy-loaded singleton)
├── rerank.py      Cross-encoder wrapper (lazy-loaded singleton)
├── llm.py         Query-parser: Qwen2.5-0.5B (local) or OpenAI-compatible (cloud)
├── store.py       SQLite store + in-memory cache + sparse path-keyword matrix
└── search.py      Hybrid retrieval + dedup + cross-encoder rerank + ask routing
```

The daemon binds to `127.0.0.1:8765`, holds models warm in memory, and processes requests with sub-second median latency. The CLI auto-spawns the daemon on first use; subsequent commands hit the warm process over HTTP.

### Search-quality improvements (in implementation order)

1. **Path-aware embedding.** During indexing, each chunk is prefixed with a cleaned location label (e.g. `[Location: Stanford CS 107 cs107 maxwell_demon]`). The label is part of the embedding, so the bi-encoder can match query terms against directory context, not just chunk text.

2. **Hybrid scoring (semantic + path keyword).** A second per-chunk score is computed from token-overlap between query tokens and cleaned path tokens. The combined distance is `0.7 × semantic + 0.3 × (1 − path_score)`.

3. **Adaptive semantic weight.** Queries with digit-bearing tokens (e.g. `cs107`, `194w`) shift the weight to `0.5` so path scoring dominates — course codes and version strings usually mean the user wants the document at that location, not text that happens to mention it. Pure natural-language queries stay at `0.7`. iMessage queries skip path scoring entirely (chunks have no path tokens) and use `1.0`.

4. **Per-source filtering.** `/search?source=imessage` and `?source=filesystem` filter at the SQL level via the in-memory cache mask, so source-filtered queries don't waste cycles ranking the wrong chunks.

5. **In-memory split cache.** All embeddings are loaded into a numpy matrix at first query and held warm; document text stays on disk and is fetched only for top-k winners. Eliminates per-query disk I/O and cuts memory footprint vs caching documents in RAM too.

6. **Vectorized path-keyword scoring.** Path tokens are encoded as a `(n_chunks × n_vocab)` scipy CSR sparse matrix at cache-load time. At query time the per-chunk path score becomes a single sparse-times-dense matrix multiplication — replaced a Python loop that was ~900 ms on a 259 K-chunk index. Median search latency dropped from ~1 s to ~170 ms.

7. **Result dedup by source path.** `/search` returns at most one chunk per file (the best-scoring), so users never see three rows from the same PDF.

8. **Cross-encoder re-ranking.** Top-50 candidates from the fast bi-encoder are re-scored by a cross-encoder that processes (query, document) pairs jointly. Dramatically more accurate at the cost of ~700 ms per query.

9. **LLM query parsing on `/ask`.** Qwen2.5-0.5B (local default) or an OpenAI-compatible cloud endpoint parses conversational queries into `{search_terms, intent, source_filter}` and routes to `search` / `find_files` / `find_directories`. Triggered explicitly by Tab or Enter so the as-you-type fast path stays cheap. JSON output is schema-constrained (`outlines` for local, `response_format` for cloud), so malformed responses are essentially impossible.

### Performance characteristics

| Operation | Latency |
|---|---|
| Daemon cold start (first time, includes model downloads) | ~7 min |
| Daemon warm restart (models cached locally) | ~20 s |
| `/search` warm | ~170 ms (bi-encoder only) / ~900 ms (with rerank) |
| `/ask` warm | ~1–2 s (LLM parse + retrieval) |
| Full reindex (≈2 K files / 260 K chunks) | ~1.5–2 hr |

`main.py` sets `HF_HUB_OFFLINE=1` automatically once the fastembed cache (`~/.atlas/models/`) is populated, so subsequent daemon starts never block on Hugging Face revision checks. On a fresh cache the daemon downloads the models on first run instead. The LLM warmup runs in a background thread so `/search` is reachable immediately even if the LLM is still loading; in cloud-parser mode no local LLM is loaded at all.

### Quality benchmark

`backend/eval/` holds a small benchmark used to measure search-quality changes. Five hand-picked queries with known expected matches; the runner records the rank of the expected match in top-10 (0 = top hit, -1 = miss) plus HTTP latency. With the daemon running:

```bash
cd ~/Desktop/cs194w/backend
.venv/bin/python eval/run.py --label baseline    # save as eval/runs/<timestamp>-baseline.json
.venv/bin/python eval/run.py --label with-hyde   # after a change; compare with diff
```

Add or refine queries in `eval/queries.py`. Match types supported: `path_exact`, `path_startswith` (for directory expectations), `snippet_contains` (for iMessage text expectations).

---

## 3. How to use the product

These instructions assume a fresh clone of the repo on macOS (Apple Silicon recommended for MLX support).

### Prerequisites

- macOS 12+ (vibrancy + frameless window APIs)
- Apple Silicon recommended (the LLM uses `mlx-lm`; Intel falls back to `llama-cpp-python`)
- Python 3.13
- Node.js 20+ and npm
- ~5 GB free disk for model weights

### Step 1 — Backend setup

```bash
cd ~/Desktop/cs194w/backend

# Create the virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# First-time-only: pre-download the LLM and re-ranker models.
# Pulls ~280 MB Qwen2.5-0.5B (MLX) + ~80 MB cross-encoder. The fastembed
# bi-encoder downloads automatically on first daemon run (~80 MB).
HF_HUB_OFFLINE=0 .venv/bin/python -c "
from fastembed.rerank.cross_encoder import TextCrossEncoder
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')
from mlx_lm import load
load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
"
```

#### Optional: cloud query parser

By default the daemon parses `/ask` queries on-device with Qwen2.5-0.5B. To swap in a cloud endpoint (default Groq + `llama-3.1-8b-instant`, free tier) — saves ~400 MB of resident memory and the model download:

```bash
# Get a free key at https://console.groq.com/keys
.venv/bin/python cli.py config set-cloud-parser true   # prompts for the key, stores in Keychain
.venv/bin/python cli.py daemon restart                 # required to pick up the new mode

# Inspect current settings
.venv/bin/python cli.py config show

# Disable / clear the key
.venv/bin/python cli.py config set-cloud-parser false
.venv/bin/python cli.py config clear-cloud-key
```

When cloud mode is on, the overlay shows a small ☁ badge next to the esc button so you know queries are being sent off-device. Indexed content always stays on your machine; only the short query string is sent to the parser endpoint.

### Step 2 — Index your data

The CLI auto-starts the daemon on the first command. The first call will warm models (~20 s if pre-downloaded; longer if not).

```bash
# Index a directory (recursive, runs on the daemon with a live progress bar)
.venv/bin/python cli.py index ~/Desktop

# Index a specific subdirectory
.venv/bin/python cli.py index "~/Desktop/Stanford/CS 107"

# Index iMessage history (requires Full Disk Access for Terminal in
# System Settings → Privacy & Security → Full Disk Access)
.venv/bin/python cli.py index --imessage

# Index Google Calendar events (see "Google Calendar setup" below for OAuth)
.venv/bin/python cli.py index --gcal

# Wipe the index and rebuild from a directory (use after upgrading the
# embedding model or changing path-context logic)
.venv/bin/python cli.py reindex ~/Desktop

# Check how many chunks are stored
.venv/bin/python cli.py status
```

#### Google Calendar setup

The first `--gcal` run opens a browser tab for Google's OAuth consent screen. Token (access + refresh) is persisted to the macOS Keychain via `keyring`, so subsequent runs are silent.

Prerequisites:

1. **OAuth client credentials.** Download `gcal_client.json` (Desktop-app type) from Google Cloud Console → APIs & Services → Credentials, and drop it next to [backend/config.py](backend/config.py). The file is gitignored — every developer / tester uses their own.
2. **Calendar API enabled** on the Cloud project that issued the client.
3. **Test-user allowlist** (only matters while the OAuth consent screen is in "Testing" mode, which is the default): Cloud Console → OAuth consent screen → Test users → add the Gmail address that will be authorizing. Up to 100 entries. Without this, sign-in fails with "Access blocked: app has not completed Google's verification process". `calendar.readonly` is a sensitive scope, so unverified apps can't open to the public without Google's review.

Indexing behaviour:

- Pulls events from every calendar where the visibility checkbox is on in Google Calendar's UI (so "US Holidays" / "Birthdays" auto-drop when hidden).
- Time window: `-2 years` to `+6 months`. Recurring events collapse to the series so the weekly standup doesn't flood results.
- Each event becomes a chunk prefixed with `[Calendar: ... | When: YYYY-MM-DD | With: ... | Where: ...]` so the bi-encoder picks up date/attendee/location context, mirroring the `[Location: ...]` trick used for filesystem chunks.

```bash
# Force re-auth (e.g. after the 7-day testing-mode refresh-token expiry)
.venv/bin/python cli.py config gcal-clear
```

### Step 3 — Daemon lifecycle

The daemon stays warm across CLI calls and across terminal sessions. Most of the time you don't need to touch it directly — `cli.py search` / `ask` / `index` auto-spawn it if it's not running.

```bash
# Explicit start (no-op if already running)
.venv/bin/python cli.py daemon start

# Stop the daemon (SIGTERM; releases port 8765)
.venv/bin/python cli.py daemon stop

# Restart (useful after editing backend code)
.venv/bin/python cli.py daemon restart

# Tail the daemon log
.venv/bin/python cli.py daemon logs -f
```

The daemon writes its PID to `~/.atlas/daemon.pid` and its log to `~/.atlas/daemon.log`. The vector index lives in `~/.atlas/db/atlas.db`.

### Step 4 — Test from the CLI

Before launching the UI, verify the daemon is returning results:

```bash
.venv/bin/python cli.py search "memory allocator"
.venv/bin/python cli.py search "cs107" --limit 5
.venv/bin/python cli.py search "ryan" --source imessage
.venv/bin/python cli.py search "team standup" --source gcal

# /ask routes through the local LLM for intent parsing
.venv/bin/python cli.py ask "find the directory with my cs107 homework"
.venv/bin/python cli.py ask "what meetings do I have with sarah next week"
```

### Step 5 — Frontend setup

```bash
cd ~/Desktop/cs194w/atlas

# One-time install (~150 MB for Electron + dependencies)
npm install

# Launch the UI
npm start
```

`npm start` runs Vite + Electron concurrently. Two windows will appear: a DevTools window (you can close it) and the actual overlay (which starts hidden — see Step 6).

### Step 6 — Using the overlay

| Action | Key |
|---|---|
| Summon overlay | **⌥+Space** |
| Live search | type into the input |
| LLM-routed ask | **Tab** or **Enter** |
| Navigate results | **↓** / **↑** |
| Dismiss | **Esc**, click outside, or **⌥+Space** again |

Live results appear ~200 ms after you stop typing. The result list shows up to 5 rows at once and scrolls if there are more. Pressing Tab or Enter triggers `/ask`, which uses the local LLM to interpret intent (e.g. "find the folder with my CS 107 homework" routes to a directory search).

### Troubleshooting

- **"Daemon failed to start"**: check `~/.atlas/daemon.log`. Most common cause is a missing model — re-run the pre-download in Step 1.
- **`Cmd+Shift+Space` does nothing**: another app (Raycast, Alfred, etc.) owns the binding. Either disable that app's hotkey or change the accelerator string in `atlas/main.ts:460` and `npm start` again.
- **Empty search results**: confirm the index isn't empty with `cli.py status`. If 0 chunks, run `cli.py index <some-path>`.
- **iMessage indexing fails**: open System Settings → Privacy & Security → Full Disk Access and add Terminal (or whatever shell you're using). Fully quit and reopen the terminal afterwards — the permission only takes effect on a fresh process.
- **Google Calendar OAuth fails / "Access blocked"**: the tester's Gmail isn't on the Test users list. Add it in Google Cloud Console → OAuth consent screen → Test users. If a previously-working token has stopped working after a week, the testing-mode refresh-token expired — run `cli.py config gcal-clear` and re-index.
- **`gcal_client.json missing` error**: download the Desktop-app OAuth client JSON from your Cloud project's Credentials page and save it to [backend/gcal_client.json](backend/gcal_client.json).
- **Frontend can't reach daemon**: check `curl http://127.0.0.1:8765/status` from a terminal. If that fails, the daemon isn't running — `cli.py daemon start`.
