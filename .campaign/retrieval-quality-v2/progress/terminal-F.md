---
terminal: F
title: "Chunk summary validation"
campaign: retrieval-quality-v2
wave: 2
status: in_progress
branch: improvement/chunk-summary-validation
writes_to:
  - src/archex/pipeline/summarize.py
issue_refs:
  - "#78"
target: "Summary quality audited, external-large recall improves with summaries"
blocked_by: []
started: "2026-03-29T10:00:00Z"
updated: "2026-03-29T10:00:00Z"
---

# Terminal F — Chunk summary validation

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #78 | Validate LLM chunk summarization quality and recall impact | external-large recall improvement | in_progress | audit complete: summaries are dead code, prompt is generic, no BM25/vector integration |

## Results

## Log

- **2026-03-29 10:00** — Audit complete. Findings: (1) summarize_chunks/enrich_chunk_content never called in pipeline, (2) CodeChunk has no summary field, (3) BM25 FTS5 has no summary column, (4) vector surrogates don't include summaries, (5) prompt is generic — doesn't emit domain-specific vocabulary. Plan: fix prompt, add summary field, integrate into BM25+vector, wire into pipeline.
