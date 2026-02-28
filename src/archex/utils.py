"""Shared utilities for archex CLI and integrations."""

from __future__ import annotations

from archex.models import RepoSource


def resolve_source(path_or_url: str) -> RepoSource:
    """Build a RepoSource from a local path or HTTP(S) URL string."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return RepoSource(url=path_or_url)
    return RepoSource(local_path=path_or_url)
