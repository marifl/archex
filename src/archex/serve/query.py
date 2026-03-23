"""Query augmentation: expand NL queries with code vocabulary via LLM."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Maximum tokens for the LLM expansion response.
_MAX_EXPANSION_TOKENS = 80

_EXPANSION_PROMPT = (
    "Given this code search query, list the likely class names, function names, "
    "and identifiers that would appear in the implementation. "
    "Return ONLY a space-separated list of identifiers, nothing else. "
    "Do not include common words — only specific code symbols.\n\n"
    "Query: {query}"
)


def augment_query(question: str, provider: LLMProvider | None) -> str:
    """Expand a natural language query with code-vocabulary identifiers.

    Uses an LLM to generate likely class names, function names, and
    identifiers that would appear in the implementation targeted by the
    query. The generated identifiers are appended to the original query
    for BM25 search, bridging the vocabulary gap between NL and code.

    The original query is always preserved — augmentation only adds terms.
    Falls back to the original query on provider error or when no provider
    is available.

    Args:
        question: The original natural language search query.
        provider: An LLM provider implementing the complete() method.
            When None, returns the question unchanged.

    Returns:
        The augmented query string (original + generated identifiers).
    """
    if provider is None:
        return question

    prompt = _EXPANSION_PROMPT.format(query=question)
    try:
        expansion = provider.complete(prompt, max_tokens=_MAX_EXPANSION_TOKENS)
    except Exception:
        logger.warning("Query augmentation failed, using original query", exc_info=True)
        return question

    # Sanitize: keep only valid identifier-like tokens
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", expansion)
    if not tokens:
        return question

    augmented = f"{question} {' '.join(tokens)}"
    logger.debug("Augmented query: %r → %r", question, augmented)
    return augmented
