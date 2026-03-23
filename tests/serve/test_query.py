"""Tests for query augmentation module."""

from __future__ import annotations

from unittest.mock import MagicMock

from archex.serve.query import augment_query


def test_augment_query_no_provider_returns_original() -> None:
    result = augment_query("How does Django's ORM build SQL queries?", None)

    assert result == "How does Django's ORM build SQL queries?"


def test_augment_query_appends_identifiers() -> None:
    provider = MagicMock()
    provider.complete.return_value = "QuerySet SQLCompiler as_sql execute_sql"

    result = augment_query("How does Django build SQL queries?", provider)

    assert "How does Django build SQL queries?" in result
    assert "QuerySet" in result
    assert "SQLCompiler" in result
    assert "as_sql" in result
    assert "execute_sql" in result


def test_augment_query_sanitizes_expansion() -> None:
    provider = MagicMock()
    provider.complete.return_value = "- QuerySet\n- SQLCompiler (class)\n* as_sql"

    result = augment_query("How does Django build SQL queries?", provider)

    assert "QuerySet" in result
    assert "SQLCompiler" in result
    assert "as_sql" in result
    # Punctuation should not appear as standalone tokens
    assert "- " not in result
    assert "* " not in result
    assert "(class)" not in result


def test_augment_query_provider_error_fallback() -> None:
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("LLM unavailable")

    result = augment_query("How does Django build SQL queries?", provider)

    assert result == "How does Django build SQL queries?"


def test_augment_query_empty_expansion_returns_original() -> None:
    provider = MagicMock()
    provider.complete.return_value = ""

    result = augment_query("How does Django build SQL queries?", provider)

    assert result == "How does Django build SQL queries?"


def test_augment_query_whitespace_expansion_returns_original() -> None:
    provider = MagicMock()
    provider.complete.return_value = "   \n\t  "

    result = augment_query("How does Django build SQL queries?", provider)

    assert result == "How does Django build SQL queries?"


def test_augment_query_preserves_original() -> None:
    original = "How does Django's ORM build SQL queries?"
    provider = MagicMock()
    provider.complete.return_value = "QuerySet SQLCompiler as_sql"

    result = augment_query(original, provider)

    assert result.startswith(original)


def test_expansion_prompt_contains_query() -> None:
    query = "How does Django build SQL queries?"
    provider = MagicMock()
    provider.complete.return_value = "QuerySet"

    augment_query(query, provider)

    call_args = provider.complete.call_args
    prompt_sent = call_args[0][0]
    assert query in prompt_sent


def test_expansion_prompt_uses_template() -> None:
    """Verify the prompt template produces a prompt containing the query."""
    query = "find the authentication middleware"
    provider = MagicMock()
    provider.complete.return_value = "AuthMiddleware authenticate"

    augment_query(query, provider)

    prompt_sent: str = provider.complete.call_args[0][0]
    assert query in prompt_sent
    assert "identifier" in prompt_sent.lower() or "class names" in prompt_sent.lower()
