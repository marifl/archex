---
terminal: D
title: "0.333-recall plateau diagnosis"
campaign: retrieval-quality-v2
wave: 1
status: in_progress
branch: improvement/recall-plateau-fix
writes_to:
  - src/archex/serve/context.py
issue_refs:
  - "#73"
target: "Aggregate recall above 0.55, no regressions on 0.667+ tasks"
blocked_by: []
started: "2026-03-29T00:00:00Z"
updated: "2026-03-29T00:00:00Z"
---

# Terminal D — 0.333-recall plateau diagnosis

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #73 | Diagnose and fix 0.333-recall plateau across 13 tasks | Aggregate recall > 0.55 | in_progress | Fix committed, awaiting benchmark |

## Results

- **Root cause**: sum-based file aggregation in `file_agg` inflated BM25-heavy files, pushing expansion files below `FILE_SCORE_CUTOFF`. Combined with `_adaptive_max_files` capping at 4 files, exactly 1 of 3 expected files survived.
- **Fix**: 6 coordinated changes in `context.py` (commit `cd2be9c`):
  1. `file_agg`: sum → max (removes chunk-count bias)
  2. `IMPORTER_DECAY`: 0.20 → 0.35
  3. `MAX_EXPANSION_FILES`: 5 → 8
  4. `MAX_FILES`: 6 → 8
  5. `FILE_SCORE_CUTOFF`: 0.15 → 0.10
  6. `_adaptive_max_files` min: 4 → 5
- **Tests**: 11 new tests in `test_context_recall.py`, all passing. Full suite: 1751 passed.
- **Expected impact**: 0.333→0.667+ for tasks where missing files are reachable via graph expansion from BM25 seeds. Estimated aggregate recall improvement from ~0.49 to ~0.60+.

## Log

- **2026-03-29 00:00** — Started diagnosis. Reading benchmark tasks, baseline.json, and context.py to trace where expected files are lost.
- **2026-03-29 00:30** — Root cause identified: sum-based file aggregation + aggressive adaptive_max_files cap. Implemented 6 fixes in context.py. 11 new tests in test_context_recall.py, 1751/1751 passing. Pushed commit `cd2be9c`.
- **2026-03-29 00:45** — PR #84 created. Awaiting benchmark run to confirm recall improvement.
