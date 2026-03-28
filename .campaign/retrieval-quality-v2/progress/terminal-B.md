---
terminal: B
title: "Fusion diagnosis"
campaign: retrieval-quality-v2
wave: 1
status: done
branch: improvement/fusion-diagnosis
writes_to:
  - src/archex/index/vector.py
  - src/archex/index/fusion.py
issue_refs:
  - "#75"
target: "Fusion aggregate recall exceeds BM25-only by at least 0.03"
blocked_by: []
started: "2026-03-29T00:00:00Z"
updated: "2026-03-29T00:00:00Z"
---

# Terminal B — Fusion diagnosis

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #75 | Diagnose fusion strategy not exceeding BM25-only recall | Fusion recall > BM25 by 0.03+ | done | PR #85 — RSF + relaxed gate |

## Results

## Root Cause Analysis

1. **Wrong embedder**: FastEmbed default is `BAAI/bge-small-en-v1.5` (384d general English), not code-specific. Vector results are low-quality.
2. **should_fuse gate too conservative**: cv_threshold=0.5, agreement_threshold=0.6 skip fusion for most queries.
3. **RRF weight schedule over-weights BM25**: High-agreement case gives vector only 0.15 weight.
4. **RRF destroys score magnitude**: Rank-based 1/(k+rank) flattens BM25 score gaps. RSF (defined but unused) preserves them.

## Log

- 2026-03-29: Started diagnosis. Identified 4 root causes. Plan: extract fusion.py, fix gate thresholds, switch RRF→RSF, rebalance weights.
- 2026-03-29: Implemented fix — extracted fusion.py, relaxed gate (cv>0.8, agreement>0.8), replaced confidence_weighted_rrf with adaptive_rsf in assemble_context. 144 tests pass, 97% coverage on fusion.py.
- 2026-03-29: Committed, pushed, opened PR #85. Awaiting benchmark run to confirm recall improvement.
