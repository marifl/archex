---
terminal: C
title: "MRR and reranking verification"
campaign: retrieval-quality-v2
wave: 1
status: in_progress
branch: improvement/mrr-reranking
writes_to:
  - src/archex/index/rerank.py
  - src/archex/api.py
issue_refs:
  - "#76"
target: "MRR >= 0.75 with reranking enabled"
blocked_by: []
started: 2026-03-29
updated: 2026-03-29
---

# Terminal C — MRR and reranking verification

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #76 | Improve MRR by verifying reranking integration | MRR >= 0.75 | in_progress | reranking auto-enabled |

## Results

- Switched default model from Jina Reranker v2 to ms-marco-MiniLM-L-6-v2 (no custom code deps)
- Auto-enable reranking when sentence-transformers is installed (was opt-in only)
- Increased DEFAULT_TOP_K from 20 to 30, MAX_CONTENT_CHARS from 3072 to 4096
- Added is_available() and _maybe_reranker() for auto-detection
- 17 tests covering reranker, auto-enable logic, score replacement

## Log

- **2026-03-29**: Diagnosis: reranking never activated (IndexConfig.rerank=False default, benchmarks don't pass rerank=True). Root fix: auto-enable when deps available.
- **2026-03-29**: Committed feat(retrieval): auto-enable reranking and expand candidate pool. Model switched to MiniLM for compatibility. Tests pass (97/97).
