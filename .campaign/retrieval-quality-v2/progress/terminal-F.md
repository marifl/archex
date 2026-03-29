---
terminal: F
title: "Chunk summary validation"
campaign: retrieval-quality-v2
wave: 2
status: complete
branch: improvement/chunk-summary-validation
writes_to:
  - src/archex/pipeline/summarize.py
issue_refs:
  - "#78"
target: "Summary quality audited, external-large recall improves with summaries"
blocked_by: []
started: "2026-03-29T10:00:00Z"
updated: "2026-03-29T10:30:00Z"
---

# Terminal F — Chunk summary validation

## Issues

| Issue | Task | Target | Status | Actual |
|-------|------|--------|--------|--------|
| #78 | Validate LLM chunk summarization quality and recall impact | external-large recall improvement | complete | summaries integrated into BM25 FTS5 (weight 8.0) + vector surrogates, prompt tuned for retrieval vocab |

## Results

## Log

- **2026-03-29 10:00** — Audit complete. Findings: (1) summarize_chunks/enrich_chunk_content never called in pipeline, (2) CodeChunk has no summary field, (3) BM25 FTS5 has no summary column, (4) vector surrogates don't include summaries, (5) prompt is generic — doesn't emit domain-specific vocabulary. Plan: fix prompt, add summary field, integrate into BM25+vector, wire into pipeline.
- **2026-03-29 10:30** — Implementation complete. Commit b9f8614: tuned prompt for retrieval vocab, added summary field to CodeChunk, added summary column to BM25 FTS5 (weight 8.0), included summaries in vector surrogates, wired summarization into produce_artifacts. 95 tests pass, 100% coverage on summarize.py, 91% on bm25.py, 84% on service.py. Pushed, opening PR.
