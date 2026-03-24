"""LLM chunk summarization: generate NL descriptions for code chunks at index time."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import CodeChunk
    from archex.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_CONTENT_FOR_SUMMARY = 2000  # chars of code to send to the LLM

_SUMMARY_SYSTEM = (
    "You are a code documentation assistant. For each code snippet, "
    "write a concise 1-2 sentence description of what the code does. "
    "Use plain English. Focus on the purpose and behavior, not the implementation details. "
    "Do not include code. Do not start with 'This function' or 'This class'."
)

_SUMMARY_PROMPT = (
    "Describe what this {language} code does in 1-2 sentences:\n\n"
    "File: {file_path}\n"
    "Symbol: {symbol_name}\n"
    "```{language}\n{content}\n```"
)


def summarize_chunk(chunk: CodeChunk, provider: LLMProvider) -> str:
    """Generate a concise NL summary for a single code chunk.

    Args:
        chunk: The code chunk to summarize.
        provider: LLM provider for generating the summary.

    Returns:
        A 1-2 sentence NL description of the chunk's purpose.
    """
    content = chunk.content[:_MAX_CONTENT_FOR_SUMMARY]
    prompt = _SUMMARY_PROMPT.format(
        language=chunk.language or "code",
        file_path=chunk.file_path,
        symbol_name=chunk.symbol_name or "(module-level)",
        content=content,
    )
    try:
        summary = provider.complete(
            prompt,
            system=_SUMMARY_SYSTEM,
            max_tokens=100,
        )
        return summary.strip()
    except Exception:
        logger.warning(
            "Failed to summarize chunk %s, skipping",
            chunk.id,
            exc_info=True,
        )
        return ""


def enrich_chunk_content(chunk: CodeChunk, summary: str) -> str:
    """Prepend a summary to chunk content for enriched indexing.

    The enriched content is used for both BM25 indexing and vector
    embedding, bridging the vocabulary gap between NL queries and code.

    Args:
        chunk: The original code chunk.
        summary: The NL summary to prepend.

    Returns:
        The enriched content string with summary prepended.
    """
    if not summary:
        return chunk.content
    return f"# Summary: {summary}\n\n{chunk.content}"


def summarize_chunks(
    chunks: list[CodeChunk],
    provider: LLMProvider,
    *,
    batch_size: int = 10,
) -> dict[str, str]:
    """Generate NL summaries for a batch of code chunks.

    Returns a mapping of chunk_id → summary string. Chunks that fail
    summarization are omitted from the result (empty string).

    Args:
        chunks: Code chunks to summarize.
        provider: LLM provider for generating summaries.
        batch_size: Number of chunks to process before logging progress.

    Returns:
        Dict mapping chunk.id to its NL summary string.
    """
    summaries: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, provider)
        if summary:
            summaries[chunk.id] = summary
        if (i + 1) % batch_size == 0:
            logger.info(
                "Summarized %d/%d chunks (%d successful)",
                i + 1,
                len(chunks),
                len(summaries),
            )
    if chunks:
        logger.info(
            "Chunk summarization complete: %d/%d chunks summarized",
            len(summaries),
            len(chunks),
        )
    return summaries
