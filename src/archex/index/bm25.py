"""BM25 keyword index: build and query a sparse text retrieval index over CodeChunks."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    symbol_name,
    file_path,
    tokenize='porter unicode61'
);
"""

_DROP_FTS_ROWS = "DELETE FROM chunks_fts;"

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "of",
    "on", "or", "so", "that", "the", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you",
})

_AND_FALLBACK_THRESHOLD = 3


def _sanitize_tokens(query: str) -> list[str]:
    """Extract and sanitize query tokens, stripping FTS5 operators and stopwords."""
    safe: list[str] = []
    for token in query.split():
        cleaned = re.sub(r"[^a-zA-Z0-9_.]", "", token)
        if cleaned and cleaned.lower() not in _STOPWORDS:
            safe.append(f'"{cleaned}"')
    return safe


def escape_fts_query(query: str) -> str:
    """Sanitize query tokens and AND-join for FTS5 precise matching.

    Stopwords are stripped before joining. AND-join requires all remaining
    terms present, improving precision for multi-term queries.
    """
    safe = _sanitize_tokens(query)
    return " AND ".join(safe) if safe else ""


def _escape_fts_query_or(query: str) -> str:
    """OR-join variant used as fallback when AND returns too few results."""
    safe = _sanitize_tokens(query)
    return " OR ".join(safe) if safe else ""


class BM25Index:
    """BM25 keyword index using SQLite FTS5."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        store.conn.execute(_CREATE_FTS)
        store.conn.commit()

    def build(self, chunks: list[CodeChunk]) -> None:
        from archex.index.chunker import _expand_identifiers

        conn = self._store.conn
        conn.execute(_DROP_FTS_ROWS)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, content, symbol_name, file_path) "
            "VALUES (?, ?, ?, ?)",
            [
                (c.id, _expand_identifiers(c.content), c.symbol_name or "", c.file_path)
                for c in chunks
            ],
        )
        conn.commit()

    def _execute_fts(
        self, escaped: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Run a single FTS5 MATCH query, returning (chunk_id, score) pairs."""
        conn = self._store.conn
        try:
            cur = conn.execute(
                "SELECT chunk_id, bm25(chunks_fts, 1.0, 2.0, 1.5) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (escaped, top_k),
            )
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for: %s", escaped, exc_info=True)
            return []
        return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def search(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        if not query.strip():
            return []

        escaped = escape_fts_query(query)
        if not escaped:
            return []

        rows = self._execute_fts(escaped, top_k)

        # Fall back to OR-join when AND returns too few results
        if len(rows) < _AND_FALLBACK_THRESHOLD:
            or_escaped = _escape_fts_query_or(query)
            if or_escaped and or_escaped != escaped:
                or_rows = self._execute_fts(or_escaped, top_k)
                if len(or_rows) > len(rows):
                    rows = or_rows

        if not rows:
            return []

        # Batch-fetch all chunks in one query instead of N individual lookups
        chunk_ids = [cid for cid, _ in rows]
        score_map = {cid: -score for cid, score in rows}
        fetched = self._store.get_chunks_by_ids(chunk_ids)

        results: list[tuple[CodeChunk, float]] = []
        for chunk in fetched:
            score = score_map.get(chunk.id, 0.0)
            results.append((chunk, score))

        # Preserve BM25 ranking order
        results.sort(key=lambda r: r[1], reverse=True)
        return results
