# -*- coding: utf-8 -*-
"""Quick end-to-end test. Run: python test_pipeline.py"""
import sys
from pathlib import Path

log = open("/tmp/atlas_test.log", "w")

def p(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()

p("1. importing embed...")
import embed
p("2. embed imported")

p("3. embedding test text...")
vec = embed.embed_one("the quick brown fox")
p(f"4. got vector of length {len(vec)}")

p("5. importing store...")
import store
p(f"6. store imported, current count: {store.count()}")

p("7. importing ingest...")
import ingest
p("8. ingest imported")

p("9. indexing ~/Desktop/cs194w ...")
n = ingest.index_filesystem(Path.home() / "Desktop" / "cs194w", verbose=True)
p(f"10. indexed {n} files, total chunks: {store.count()}")

p("11. searching...")
import search
results = search.search("AI privacy local data indexing")
p(f"12. got {len(results)} results")
for r in results[:3]:
    p(f"    [{r.source_type}] score={r.score} | {r.snippet[:60]}")

p("DONE")
log.close()
