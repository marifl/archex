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
| #79 | Add SPLADE learned sparse retrieval as BM25 alternative | SPLADE recall > BM25F + 0.10 | in_progress | implementing SPLADEIndex |

## Results

## Log

- **2026-03-29 — started**: Reading codebase, understanding BM25Index interface, planning SPLADEIndex implementation
