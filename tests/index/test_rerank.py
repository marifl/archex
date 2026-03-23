"""Tests for CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from archex.index.rerank import _DEFAULT_MODEL, _MAX_CONTENT_CHARS, CrossEncoderReranker
from archex.models import CodeChunk, SymbolKind


def _make_chunk(chunk_id: str, content: str = "def fn(): pass") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=1,
        symbol_name=chunk_id,
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
    )


class TestCrossEncoderReranker:
    def test_init_lazy_no_model_loaded(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker._model is None

    def test_default_model_name(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker._model_name == _DEFAULT_MODEL

    def test_custom_model_name(self) -> None:
        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker._model_name == "custom/model"

    def test_rerank_empty_candidates(self) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_sorts_by_cross_encoder_score(self, mock_ce_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        candidates = [(c, float(i)) for i, c in enumerate(chunks)]

        result = reranker.rerank("query", candidates)

        assert len(result) == 3
        assert result[0][0].id == "b"
        assert result[1][0].id == "c"
        assert result[2][0].id == "a"

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_respects_top_k(self, mock_ce_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.5, 0.1])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        candidates = [(c, float(i)) for i, c in enumerate(chunks)]

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_truncates_content(self, mock_ce_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0])
        mock_ce_cls.return_value = mock_model

        long_content = "x" * (_MAX_CONTENT_CHARS + 1000)
        chunk = _make_chunk("long", content=long_content)
        reranker = CrossEncoderReranker()
        reranker.rerank("query", [(chunk, 1.0)])

        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs[0][1]) == _MAX_CONTENT_CHARS

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_returns_float_scores(self, mock_ce_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.75])
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [(_make_chunk("a"), 1.0)])
        assert isinstance(result[0][1], float)
        assert result[0][1] == 0.75
