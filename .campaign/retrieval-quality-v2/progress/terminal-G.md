---
terminal: G
title: "Structural context breadcrumbs"
campaign: retrieval-quality-v2
wave: 2
status: complete
branch: improvement/structural-breadcrumbs
writes_to:
  - src/archex/pipeline/chunking/
  - src/archex/models.py
issue_refs:
  - "#80"
target: "Chunks include structural breadcrumbs, recall improvement on external-large bucket"
blocked_by: []
started: "2026-03-29T00:00:00Z"
updated: "2026-03-29T00:00:00Z"
---

# Terminal G — Structural context breadcrumbs

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #80 | Enrich chunks with structural context breadcrumbs | Breadcrumbs in BM25+vector, recall improvement | complete | Breadcrumbs in model, chunker, BM25 FTS5, vector embeddings. 28 tests. |

## Results

## Log

- **2026-03-29** — Started implementation. Added breadcrumbs field to CodeChunk, breadcrumb generation in chunker, FTS5 column in BM25, vector embedding prepend. Writing tests.
- **2026-03-29** — Committed e32feb3: all 410 index tests pass, 28 new breadcrumb tests. Pushed. Opening PR and closing #80.
