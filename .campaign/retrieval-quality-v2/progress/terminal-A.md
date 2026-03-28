---
terminal: A
title: "Full benchmark with all features"
campaign: retrieval-quality-v2
wave: 1
status: in_progress
branch: improvement/full-feature-benchmark
writes_to:
  - benchmarks/
  - src/archex/cli/
issue_refs:
  - "#77"
target: "Benchmark results recorded for BM25-only, fusion, fusion+rerank, fusion+rerank+augment configurations"
blocked_by: []
started: "2026-03-29T00:00:00Z"
updated: "2026-03-29T00:00:00Z"
---

# Terminal A — Full benchmark with all features

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #77 | Run full benchmark with all Phase 12-14 features enabled | Results for 4 configurations | in_progress | Wiring rerank + augment strategies into benchmark CLI |

## Results

## Log

- **2026-03-29 00:00** — Started. Traced feature activation: rerank via `index_config.rerank=True`, augment via `config.provider`. Adding Strategy enum values, runner functions, CLI flags.
