"""Top-level public API: analyze, query, and compare entry points."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

from archex.acquire import clone_repo, discover_files, open_local
from archex.index.graph import DependencyGraph
from archex.models import ArchProfile, Config, RepoMetadata, RepoSource
from archex.parse import (
    LanguageAdapter,
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.parse.adapters.python import PythonAdapter
from archex.serve.profile import build_profile

if TYPE_CHECKING:
    from pathlib import Path

    from archex.models import (
        ComparisonResult,
        ContextBundle,
        IndexConfig,
    )


def analyze(
    source: RepoSource,
    config: Config | None = None,
    index_config: IndexConfig | None = None,
) -> ArchProfile:
    """Acquire, parse, index, and analyze a repository."""
    if config is None:
        config = Config()

    # Determine repo path
    repo_path: Path
    url: str | None = None
    local_path: str | None = None

    if source.url and (source.url.startswith("http://") or source.url.startswith("https://")):
        url = source.url
        target_dir = tempfile.mkdtemp()
        repo_path = clone_repo(source.url, target_dir)
    elif source.local_path is not None:
        local_path = source.local_path
        repo_path = open_local(source.local_path)
    else:
        raise ValueError("RepoSource must have a url or local_path")

    # Discover files
    files = discover_files(repo_path, languages=config.languages)

    # Build engine and adapters
    engine = TreeSitterEngine()
    adapters: dict[str, LanguageAdapter] = {"python": PythonAdapter()}

    # Parse symbols and imports
    parsed_files = extract_symbols(files, engine, adapters)
    import_map = parse_imports(files, engine, adapters)
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)

    # Build dependency graph
    graph = DependencyGraph.from_parsed_files(parsed_files, resolved_map)

    # Build repo metadata
    lang_counts: dict[str, int] = {}
    for f in files:
        lang_counts[f.language] = lang_counts.get(f.language, 0) + 1

    total_lines = sum(pf.lines for pf in parsed_files)

    repo_metadata = RepoMetadata(
        url=url,
        local_path=local_path,
        languages=lang_counts,
        total_files=len(files),
        total_lines=total_lines,
    )

    return build_profile(repo_metadata, parsed_files, graph)


def query(
    profile: ArchProfile,
    query: str,
    token_budget: int = 8192,
) -> ContextBundle:
    """Retrieve a ranked ContextBundle for a natural-language query."""
    # TODO: Implement in Phase 3
    raise NotImplementedError


def compare(
    source_a: RepoSource,
    source_b: RepoSource,
    config: Config | None = None,
) -> ComparisonResult:
    """Analyze two repositories and return a ComparisonResult."""
    # TODO: Implement in Phase 4
    raise NotImplementedError
