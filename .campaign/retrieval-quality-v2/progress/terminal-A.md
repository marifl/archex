---
terminal: A
title: "Full benchmark with all features"
campaign: retrieval-quality-v2
wave: 1
status: done
branch: improvement/full-feature-benchmark
writes_to:
  - benchmarks/
  - src/archex/cli/
issue_refs:
  - "#77"
target: "Benchmark results recorded for BM25-only, fusion, fusion+rerank, fusion+rerank+augment configurations"
blocked_by: []
started: "2026-03-29T00:00:00Z"
updated: "2026-03-29T00:01:00Z"
---

# Terminal A — Full benchmark with all features

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #77 | Run full benchmark with all Phase 12-14 features enabled | Results for 4 configurations | done | 4 strategies wired: archex_query, archex_query_fusion, archex_query_fusion_rerank, archex_query_fusion_rerank_augment |

## Results

All 4 configurations runnable via benchmark CLI:

```bash
# Config 1: BM25-only (default)
uv run archex benchmark run

# Config 2: BM25 + vector fusion
uv run archex benchmark run --query-fusion

# Config 3: Fusion + cross-encoder reranking
uv run archex benchmark run --rerank

# Config 4: Fusion + reranking + LLM query augmentation (requires OPENAI_API_KEY)
uv run archex benchmark run --augment
```

## Log

- **2026-03-29 00:00** — Started. Traced feature activation: rerank via `index_config.rerank=True`, augment via `config.provider`. Adding Strategy enum values, runner functions, CLI flags.
- **2026-03-29 00:01** — Committed and pushed. 2 new Strategy enum values, 2 runner functions, 2 CLI flags (--rerank, --augment), registry wiring, 173 benchmark tests passing.
