"""Cross-encoder reranking stage for candidate refinement."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError

if TYPE_CHECKING:
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# Maximum content length passed to the cross-encoder per chunk.
# Jina Reranker v2 has a 1024-token context window.
MAX_CONTENT_CHARS = 3072


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder model.

    Cross-encoders compute full query-chunk attention, capturing
    token-level interactions that bi-encoder similarity misses.
    Applied as a post-fusion refinement stage over the top-N
    candidates to improve precision without affecting recall.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ArchexIndexError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        print(
            f"Loading reranker model '{self._model_name}' (downloading if not cached)...",
            file=sys.stderr,
            flush=True,
        )
        self._model = CrossEncoder(self._model_name)
        logger.info("Loaded cross-encoder reranker: %s", self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        top_k: int = 20,
    ) -> list[tuple[CodeChunk, float]]:
        """Rerank candidates by cross-encoder relevance score.

        Args:
            query: The search query.
            candidates: (chunk, score) pairs from prior retrieval stages.
            top_k: Maximum number of results to return.

        Returns:
            Re-scored (chunk, cross_encoder_score) pairs sorted by relevance.
        """
        if not candidates:
            return []

        self._load_model()

        pairs = [(query, chunk.content[:MAX_CONTENT_CHARS]) for chunk, _ in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        scored = sorted(
            zip(candidates, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(ce_score)) for (chunk, _), ce_score in scored[:top_k]]
