---
terminal: I
title: "SPLADE learned sparse retrieval"
campaign: retrieval-quality-v2
wave: 2
status: in_progress
branch: improvement/splade-retrieval
writes_to:
  - src/archex/index/splade.py
  - src/archex/index/bm25.py
issue_refs:
  - "#79"
target: "SPLADE recall exceeds BM25F by >= 0.10 on external-large repos"
blocked_by: []
started: 2026-03-29
updated: 2026-03-29
---

# Terminal I — SPLADE learned sparse retrieval

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #79 | Add SPLADE learned sparse retrieval as BM25 alternative | SPLADE recall > BM25F + 0.10 | in_progress | SPLADEIndex + tests committed, PR pending |

## Results

### Expected Performance Characteristics

| Metric | BM25F (baseline) | SPLADE (expected) |
|--------|------------------|-------------------|
| Indexing time | ~instant (FTS5) | ~2-5s per 1K chunks (MLM forward pass) |
| Storage overhead | FTS5 internal | ~3-10x more inverted entries (vocabulary expansion) |
| Recall on NL→code queries | Limited by exact term matching | +0.10-0.15 via vocabulary expansion |
| Query latency | <10ms (FTS5 MATCH) | ~50-100ms (encode + inverted lookup) |

### Integration Path

SPLADE should run **alongside** BM25 as a third signal in fusion, not replace it:
- BM25 remains fast/cheap for exact-match queries (function names, error messages)
- SPLADE adds value specifically for NL→code queries with vocabulary gap
- Fusion weight: start at bm25=0.35, vector=0.35, splade=0.30 when all three signals available

## Log

- **2026-03-29 — started**: Reading codebase, understanding BM25Index interface, planning SPLADEIndex implementation
- **2026-03-29 — commit 7ef1aa2**: SPLADEIndex implemented with SparseEncoder protocol, inverted-index SQLite storage, save/load, 19 unit tests passing. splade optional dep added to pyproject.toml. Pushed to origin.
