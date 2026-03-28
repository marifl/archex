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
| #73 | Diagnose and fix 0.333-recall plateau across 13 tasks | Aggregate recall > 0.55 | in_progress | Diagnosing loss points |

## Results

## Log

- **2026-03-29 00:00** — Started diagnosis. Reading benchmark tasks, baseline.json, and context.py to trace where expected files are lost.
