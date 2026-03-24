"""Tests for LLM chunk summarization module."""

from __future__ import annotations

from unittest.mock import MagicMock

from archex.models import CodeChunk, SymbolKind
from archex.pipeline.summarize import (
    enrich_chunk_content,
    summarize_chunk,
    summarize_chunks,
)

# Mirrors archex.pipeline.summarize._MAX_CONTENT_FOR_SUMMARY — update if the constant changes.
_MAX_CONTENT_FOR_SUMMARY = 2000


def _make_chunk(
    chunk_id: str = "router.py:dispatch:1",
    content: str = "def dispatch(request): ...",
    file_path: str = "router.py",
    start_line: int = 1,
    end_line: int = 5,
    symbol_name: str | None = "dispatch",
    symbol_kind: SymbolKind | None = SymbolKind.FUNCTION,
    language: str = "python",
    token_count: int = 20,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        language=language,
        token_count=token_count,
    )


def test_summarize_chunk_returns_summary() -> None:
    # Arrange
    provider = MagicMock()
    provider.complete.return_value = "Handles HTTP request routing and dispatch."
    chunk = _make_chunk()

    # Act
    result = summarize_chunk(chunk, provider)

    # Assert
    assert result == "Handles HTTP request routing and dispatch."


def test_summarize_chunk_provider_error_returns_empty() -> None:
    # Arrange
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("LLM unavailable")
    chunk = _make_chunk()

    # Act
    result = summarize_chunk(chunk, provider)

    # Assert
    assert result == ""


def test_summarize_chunk_truncates_long_content() -> None:
    # Arrange
    long_content = "x" * (_MAX_CONTENT_FOR_SUMMARY + 500)
    chunk = _make_chunk(content=long_content)
    provider = MagicMock()
    provider.complete.return_value = "Does something."

    # Act
    summarize_chunk(chunk, provider)

    # Assert — inspect the prompt sent to the provider
    call_args = provider.complete.call_args
    prompt_sent: str = call_args[0][0]
    # The truncated content should appear in the prompt
    assert "x" * _MAX_CONTENT_FOR_SUMMARY in prompt_sent
    # The extra chars beyond the limit must not be present
    assert "x" * (_MAX_CONTENT_FOR_SUMMARY + 1) not in prompt_sent


def test_enrich_chunk_content_prepends_summary() -> None:
    # Arrange
    chunk = _make_chunk(content="def dispatch(request): ...")
    summary = "Handles HTTP request routing and dispatch."

    # Act
    result = enrich_chunk_content(chunk, summary)

    # Assert
    assert result.startswith("# Summary: ")
    assert summary in result
    assert chunk.content in result


def test_enrich_chunk_content_empty_summary_returns_original() -> None:
    # Arrange
    chunk = _make_chunk(content="def dispatch(request): ...")

    # Act
    result = enrich_chunk_content(chunk, "")

    # Assert
    assert result == chunk.content


def test_summarize_chunks_batch() -> None:
    # Arrange
    chunks = [_make_chunk(chunk_id=f"file.py:fn{i}:{i}", symbol_name=f"fn{i}") for i in range(5)]
    provider = MagicMock()
    provider.complete.return_value = "Does something useful."

    # Act
    result = summarize_chunks(chunks, provider)

    # Assert
    assert len(result) == 5
    for chunk in chunks:
        assert chunk.id in result
        assert result[chunk.id] == "Does something useful."


def test_summarize_chunks_partial_failure() -> None:
    # Arrange
    chunks = [_make_chunk(chunk_id=f"file.py:fn{i}:{i}", symbol_name=f"fn{i}") for i in range(5)]
    provider = MagicMock()

    call_counter = {"n": 0}

    def counting_complete(prompt: str, **kwargs: object) -> str:
        call_counter["n"] += 1
        if call_counter["n"] == 3:
            raise RuntimeError("LLM error on chunk 3")
        return "Does something useful."

    provider.complete.side_effect = counting_complete

    # Act
    result = summarize_chunks(chunks, provider)

    # Assert — 4 chunks succeed, chunk at index 2 fails and is omitted
    assert len(result) == 4
    failing_chunk_id = chunks[2].id
    assert failing_chunk_id not in result
    for i, chunk in enumerate(chunks):
        if i != 2:
            assert chunk.id in result


def test_summary_prompt_includes_file_path_and_symbol() -> None:
    # Arrange
    provider = MagicMock()
    provider.complete.return_value = "Registers routes with the Flask application."
    chunk = _make_chunk(
        file_path="src/web/routes.py",
        symbol_name="register_routes",
    )

    # Act
    summarize_chunk(chunk, provider)

    # Assert
    call_args = provider.complete.call_args
    prompt_sent: str = call_args[0][0]
    assert "src/web/routes.py" in prompt_sent
    assert "register_routes" in prompt_sent
