# Atlas

A locally-run, privacy-first AI search overlay for macOS. Press a global hotkey, type a question, get ranked results from your filesystem and iMessages. **By default, all retrieval, ranking, and query parsing happen on-device — no data leaves your machine.** An optional cloud parser (Groq-compatible) can be enabled for users who prefer it; see "Optional: cloud query parser" below.

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

The Electron main process owns the BrowserWindow (frameless, transparent, `vibrancy: "hud"`, frosted-glass border). It registers `Option+Space` as a global toggle and listens for blur events to dismiss the overlay on click-outside. The React renderer talks to the FastAPI daemon over HTTP (CORS-allowed loopback).

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
├── ingest.py      File walker, parser, chunker, path-context extractor
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

# Wipe the index and rebuild from a directory (use after upgrading the
# embedding model or changing path-context logic)
.venv/bin/python cli.py reindex ~/Desktop

# Check how many chunks are stored
.venv/bin/python cli.py status
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

# /ask routes through the local LLM for intent parsing
.venv/bin/python cli.py ask "find the directory with my cs107 homework"
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
- **`Option+Space` does nothing**: another app (the macOS character viewer, Raycast, Alfred, etc.) owns the binding. Either disable that app's hotkey or change the accelerator string in `atlas/main.ts:464` and `npm start` again.
- **Empty search results**: confirm the index isn't empty with `cli.py status`. If 0 chunks, run `cli.py index <some-path>`.
- **iMessage indexing fails**: open System Settings → Privacy & Security → Full Disk Access and add Terminal (or whatever shell you're using).
- **Frontend can't reach daemon**: check `curl http://127.0.0.1:8765/status` from a terminal. If that fails, the daemon isn't running — `cli.py daemon start`.
- **Cloud parser shows "no API key in Keychain"**: re-run `cli.py config set-cloud-parser true` and paste the key when prompted. Use `cli.py config show` to confirm whether a key is stored.
- **Cloud badge (☁) doesn't update after toggling `cloud_parser`**: the renderer refetches `/config` on every overlay show, so a fresh `Option+Space` after `cli.py daemon restart` is enough. If it still doesn't appear, force a renderer reload (close and re-launch `npm start`).

---

## 4. Next steps

Milestones still ahead, roughly in order of impact:

### Search quality

- **Path injection into the reranker.** Prepend `[Path: <file path>]` to each document before the cross-encoder scores it. The cross-encoder currently sees chunk text only, which is why a CS 106B page that mentions "CS107" can outrank actual CS 107 files. Cheapest open quality lever; ~5 lines in `search.py`.
- **Document-level LLM tagging.** One LLM call per file at index time extracts `{topics, document_type, summary}` and stores them as a new `tags` column. Use as a retrieval signal or score boost. ~3 hours of one-time indexing for ~5,000 files; promising for disambiguating similar paths.
- **HyDE for `/ask`.** Expand the user's query into a hypothetical answer via the local LLM, embed *that*, and search. Adds ~1–2 s to `/ask` only; big recall win on vague queries like "find my notes on memory management".
- **Better embedding model.** Swap `MiniLM-L6` (384-dim) for `bge-large-en-v1.5` (1024-dim) for a strong retrieval-benchmark improvement. Full reindex required; ~3× per-chunk embedding time.
- **BM25 hybrid.** Add term-frequency scoring as a third signal alongside semantic + path-keyword, fused via reciprocal rank fusion. Catches exact-phrase matches that semantic similarity misses entirely.

### New data sources

- **Phase 2 RAG Q&A — wire `/query` into the UI.** The endpoint exists (synthesizes an answer from retrieved chunks via the local or cloud LLM, with `[1]/[2]` citations) but the overlay still only routes Tab/Enter to `/ask`. Needs a second keybinding (e.g. Shift+Enter) and an answer-display block above the result list.
- **Speed up `/query`.** End-to-end latency is currently ~2.2 s on local Qwen 0.5B (vs ~0.8 s for `/search`). Options to investigate: skip the rerank stage since the LLM does its own relevance weighting during synthesis (saves ~1 s), shorten `_RAG_MAX_TOKENS`, or stream tokens to the UI so first-byte latency is what the user feels instead of total. Cloud mode is already ~1.3 s on Groq llama-3.1-8b — close to acceptable without further work.
- **Calendar (EventKit).** Index local Calendar events. Requires a native macOS bridge since the frontend is Electron, not Swift.
- **Apple Mail** (`.emlx`). Parse `~/Library/Mail/` and index message bodies + headers. Needs Full Disk Access.

### Performance & polish

- **MLX embeddings on Apple Silicon.** Replace ONNX-CPU bi-encoder with an MLX port for ~5–10× faster batch indexing of large directories. Sits naturally alongside the existing `mlx-lm` setup for the LLM.
- **Packaging with `electron-builder`.** Produce a signed `.app` for distribution. Today `npm start` is the only entry point; no installable bundle yet.
- **Re-baseline `backend/eval/`.** Re-run the 5-query eval set against Qwen 2.5 0.5B (current local parser) and Groq `llama-3.1-8b-instant` (cloud parser) to confirm parse quality didn't regress after the Phi-3 swap, and to compare local vs cloud parse fidelity.
