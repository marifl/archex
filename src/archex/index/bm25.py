"""BM25 keyword index: build and query a sparse text retrieval index over CodeChunks."""

from __future__ import annotations

import itertools
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
    docstring,
    tokenize='porter unicode61'
);
"""

_DROP_FTS_ROWS = "DELETE FROM chunks_fts;"


def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
    """Ensure the FTS table has the current schema (including docstring column).

    FTS5 does not support ALTER TABLE, so we detect stale schemas
    by checking column count and recreate if needed.
    """
    try:
        # Probe for the docstring column by attempting a dummy query
        conn.execute("SELECT chunk_id FROM chunks_fts WHERE docstring MATCH 'probe' LIMIT 0")
    except sqlite3.OperationalError:
        # docstring column missing — drop and recreate with new schema
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(_CREATE_FTS)
        conn.commit()


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
    }
)

_GRADUATED_THRESHOLD = 10


def _sanitize_tokens(query: str) -> list[str]:
    """Extract and sanitize query tokens, stripping FTS5 operators and stopwords."""
    safe: list[str] = []
    for token in query.split():
        cleaned = re.sub(r"[^a-zA-Z0-9_.]", "", token)
        if cleaned and cleaned.lower() not in _STOPWORDS:
            safe.append(f'"{cleaned}"')
    return safe


def escape_fts_query(query: str) -> str:
    """Sanitize query tokens and AND-join for FTS5 precise matching."""
    safe = _sanitize_tokens(query)
    return " AND ".join(safe) if safe else ""


class BM25Index:
    """BM25 keyword index using SQLite FTS5."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        store.conn.execute(_CREATE_FTS)
        _ensure_fts_schema(store.conn)
        store.conn.commit()

    @property
    def has_data(self) -> bool:
        """Check if FTS table already has indexed rows (avoids rebuild on cache hit)."""
        row = self._store.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        return bool(row and row[0] > 0)

    def build(self, chunks: list[CodeChunk]) -> None:
        from archex.index.chunker import expand_identifiers

        conn = self._store.conn
        conn.execute(_DROP_FTS_ROWS)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, content, symbol_name, file_path, docstring) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    c.id,
                    expand_identifiers(c.content),
                    c.symbol_name or "",
                    c.file_path,
                    c.docstring or "",
                )
                for c in chunks
            ],
        )
        conn.commit()

    def _execute_fts(
        self,
        escaped: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Run a single FTS5 MATCH query, returning (chunk_id, score) pairs."""
        conn = self._store.conn
        try:
            cur = conn.execute(
                "SELECT chunk_id, bm25(chunks_fts, 0.5, 10.0, 5.0, 6.0) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (escaped, top_k),
            )
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for: %s", escaped, exc_info=True)
            return []
        return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def _graduated_search(
        self,
        tokens: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Graduated fallback: AND-all → AND-subsets → OR-all.

        1. AND all terms — if >= threshold results, return
        2. Drop one term at a time (N-1 subsets), merge results
        3. If still < threshold, try pairs of terms, merge
        4. Final fallback: OR all terms
        """
        # Stage 1: AND all terms
        if len(tokens) >= 2:
            and_query = " AND ".join(tokens)
            rows = self._execute_fts(and_query, top_k)
            if len(rows) >= _GRADUATED_THRESHOLD:
                return rows

        # Stage 2: drop one term at a time (N-1 subsets)
        merged: dict[str, float] = {}
        if len(tokens) >= 3:
            for subset in itertools.combinations(tokens, len(tokens) - 1):
                sub_query = " AND ".join(subset)
                sub_rows = self._execute_fts(sub_query, top_k)
                for cid, score in sub_rows:
                    # Keep the best score per chunk across subsets
                    if cid not in merged or score < merged[cid]:  # FTS5 scores are negative
                        merged[cid] = score
            if len(merged) >= _GRADUATED_THRESHOLD:
                return sorted(merged.items(), key=lambda x: x[1])[:top_k]

        # Stage 3: pairs of terms
        if len(tokens) >= 4:
            for pair in itertools.combinations(tokens, 2):
                pair_query = " AND ".join(pair)
                pair_rows = self._execute_fts(pair_query, top_k)
                for cid, score in pair_rows:
                    if cid not in merged or score < merged[cid]:
                        merged[cid] = score
            if len(merged) >= _GRADUATED_THRESHOLD:
                return sorted(merged.items(), key=lambda x: x[1])[:top_k]

        # Stage 4: OR all terms
        or_query = " OR ".join(tokens)
        or_rows = self._execute_fts(or_query, top_k)
        # Merge OR results with anything we found earlier
        for cid, score in or_rows:
            if cid not in merged or score < merged[cid]:
                merged[cid] = score

        return sorted(merged.items(), key=lambda x: x[1])[:top_k]

    def avg_idf(self, query: str) -> float:
        """Compute average Inverse Document Frequency of query terms against the corpus.

        Low AvgIDF (< ~2.0) means query terms are common across the corpus —
        BM25 scores will be flat and unreliable for ranking.
        High AvgIDF means terms are specific — BM25 can discriminate effectively.

        Used as a pre-retrieval Query Performance Prediction signal to decide
        whether to force fusion with vector search.
        """
        import math

        tokens = _sanitize_tokens(query)
        if not tokens:
            return 0.0

        conn = self._store.conn
        total_docs_row = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        total_docs = total_docs_row[0] if total_docs_row else 0
        if total_docs == 0:
            return 0.0

        idfs: list[float] = []
        for token in tokens:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (token,),
                ).fetchone()
                df = row[0] if row else 0
            except sqlite3.OperationalError:
                continue
            if df > 0:
                idfs.append(math.log(total_docs / df))

        return sum(idfs) / len(idfs) if idfs else 0.0

    def search(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        if not query.strip():
            return []

        tokens = _sanitize_tokens(query)
        if not tokens:
            return []

        rows = self._graduated_search(tokens, top_k)

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
