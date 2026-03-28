"""Tests for recall-plateau fixes: max aggregation, adaptive limits, decay tuning."""

from __future__ import annotations

from archex.index.graph import DependencyGraph
from archex.models import CodeChunk, Module, SymbolKind
from archex.serve.context import assemble_context


def make_chunk(
    chunk_id: str,
    file_path: str,
    content: str = "def foo(): pass",
    symbol_kind: SymbolKind | None = None,
    symbol_name: str | None = None,
    token_count: int = 10,
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=5,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        language="python",
        token_count=token_count,
    )


# ---------------------------------------------------------------------------
# Max-based file aggregation
# ---------------------------------------------------------------------------


def test_max_aggregation_prevents_chunk_count_bias() -> None:
    """File with many chunks must not outrank file with fewer but equally scored chunks.

    Sum-based aggregation inflated files with many BM25 chunks. Max-based
    aggregation ranks files by their single best chunk, removing chunk-count bias.
    """
    graph = DependencyGraph()
    graph.add_file_node("many_chunks.py")
    graph.add_file_node("few_chunks.py")
    many = [make_chunk(f"cm{i}", "many_chunks.py", token_count=10) for i in range(8)]
    few = [make_chunk("cf0", "few_chunks.py", token_count=10)]
    all_chunks = many + few
    results = [(c, 5.0) for c in many] + [(few[0], 5.0)]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=5000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "few_chunks.py" in included_files, (
        "few_chunks.py must survive cutoff despite having fewer chunks"
    )


def test_expansion_file_survives_cutoff_with_max_aggregation() -> None:
    """Expansion file with strong neighbor_boost must pass FILE_SCORE_CUTOFF.

    With sum aggregation, seed files with many chunks inflated the cutoff
    threshold, eliminating expansion files. Max aggregation fixes this.
    """
    graph = DependencyGraph()
    graph.add_file_node("seed.py")
    graph.add_file_node("dep.py")
    graph.add_file_edge("seed.py", "dep.py", kind="imports")
    seed_chunks = [make_chunk(f"cs{i}", "seed.py", token_count=10) for i in range(10)]
    dep_chunk = make_chunk("cd0", "dep.py", token_count=10)
    all_chunks = seed_chunks + [dep_chunk]
    results = [(c, 8.0) for c in seed_chunks]
    bundle = assemble_context(results, graph, all_chunks, "q", token_budget=5000)
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "dep.py" in included_files, (
        "dep.py (import target) must survive cutoff even when seed has many chunks"
    )


# ---------------------------------------------------------------------------
# Adaptive max files
# ---------------------------------------------------------------------------


def test_adaptive_max_files_minimum_is_five() -> None:
    """_adaptive_max_files never returns fewer than 5 for non-trivial inputs."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [
        ("top.py", 10.0),
        ("a.py", 0.5),
        ("b.py", 0.3),
        ("c.py", 0.2),
        ("d.py", 0.1),
        ("e.py", 0.05),
    ]
    result = _adaptive_max_files(file_scores)
    assert result >= 5, f"Expected >= 5, got {result}"


def test_adaptive_max_files_returns_8_for_flat_scores() -> None:
    """Flat score distribution returns the full default of 8."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [(f"f{i}.py", 5.0 - i * 0.1) for i in range(10)]
    result = _adaptive_max_files(file_scores)
    assert result == 8


def test_adaptive_max_files_returns_6_for_moderate_separation() -> None:
    """Moderate score separation (2-3x ratio) returns 6."""
    from archex.serve.context import _adaptive_max_files  # pyright: ignore[reportPrivateUsage]

    file_scores = [
        ("top.py", 5.0),
        ("a.py", 3.0),
        ("b.py", 2.0),  # median
        ("c.py", 1.5),
        ("d.py", 1.0),
    ]
    result = _adaptive_max_files(file_scores)
    # top/median = 5.0/2.0 = 2.5 → > 2.0 → 6
    assert result == 6


# ---------------------------------------------------------------------------
# Importer decay
# ---------------------------------------------------------------------------


def test_importer_file_included_with_higher_decay() -> None:
    """Importer (consumer) files with IMPORTER_DECAY=0.35 survive cutoff."""
    graph = DependencyGraph()
    graph.add_file_node("core.py")
    graph.add_file_node("consumer.py")
    graph.add_file_edge("consumer.py", "core.py", kind="imports")
    core_chunk = make_chunk("c_core", "core.py", token_count=10)
    consumer_chunk = make_chunk("c_consumer", "consumer.py", token_count=10)
    results = [(core_chunk, 5.0)]
    bundle = assemble_context(
        results, graph, [core_chunk, consumer_chunk], "q", token_budget=1000
    )
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "consumer.py" in included_files, (
        "consumer.py (importer with 0.35 decay) must be included"
    )


def test_importer_relevance_score_reflects_decay() -> None:
    """Importer chunk gets relevance proportional to IMPORTER_DECAY."""
    from archex.serve.context import IMPORTER_DECAY

    graph = DependencyGraph()
    graph.add_file_node("seed.py")
    graph.add_file_node("importer.py")
    graph.add_file_edge("importer.py", "seed.py", kind="imports")
    seed_chunk = make_chunk("cs", "seed.py", token_count=10)
    imp_chunk = make_chunk("ci", "importer.py", token_count=10)
    results = [(seed_chunk, 5.0)]
    bundle = assemble_context(
        results, graph, [seed_chunk, imp_chunk], "q", token_budget=1000
    )
    imp_rc = next(rc for rc in bundle.chunks if rc.chunk.file_path == "importer.py")
    # Relevance should be seed_normalized * IMPORTER_DECAY = 1.0 * 0.35
    assert abs(imp_rc.relevance_score - IMPORTER_DECAY) < 0.01


# ---------------------------------------------------------------------------
# Architecture query multi-file retrieval
# ---------------------------------------------------------------------------


def test_architecture_query_retrieves_three_related_files() -> None:
    """Architecture query spanning 3 files via imports retrieves all three.

    Simulates the django_middleware / express_middleware pattern where the
    query expects files from a BM25 seed plus two import-connected files.
    """
    graph = DependencyGraph()
    for name in ("handlers/base.py", "handlers/wsgi.py", "middleware/common.py"):
        graph.add_file_node(name)
    graph.add_file_edge("handlers/base.py", "handlers/wsgi.py", kind="imports")
    graph.add_file_edge("middleware/common.py", "handlers/base.py", kind="imports")

    base_chunks = [make_chunk(f"cb{i}", "handlers/base.py", token_count=10) for i in range(5)]
    wsgi_chunk = make_chunk("cw0", "handlers/wsgi.py", token_count=10)
    common_chunk = make_chunk("cc0", "middleware/common.py", token_count=10)
    all_chunks = base_chunks + [wsgi_chunk, common_chunk]

    results = [(c, 6.0) for c in base_chunks] + [(common_chunk, 2.0)]
    bundle = assemble_context(
        results,
        graph,
        all_chunks,
        "How does the middleware chain handle requests?",
        token_budget=5000,
    )
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "handlers/base.py" in included_files
    assert "handlers/wsgi.py" in included_files, (
        "wsgi.py (import target of seed) must be included via expansion"
    )
    assert "middleware/common.py" in included_files, (
        "common.py (BM25 hit) must survive cutoff with max aggregation"
    )


def test_express_like_three_file_retrieval() -> None:
    """Express-like query retrieves router/index.js, route.js, and layer.js.

    Simulates express_middleware where all three files are in BM25 but only
    one survived the old sum-based cutoff.
    """
    graph = DependencyGraph()
    for name in ("lib/router/index.js", "lib/router/route.js", "lib/router/layer.js"):
        graph.add_file_node(name)
    graph.add_file_edge("lib/router/index.js", "lib/router/route.js", kind="imports")
    graph.add_file_edge("lib/router/index.js", "lib/router/layer.js", kind="imports")

    idx_chunks = [
        make_chunk(f"ci{i}", "lib/router/index.js", token_count=10) for i in range(6)
    ]
    route_chunk = make_chunk("cr0", "lib/router/route.js", token_count=10)
    layer_chunk = make_chunk("cl0", "lib/router/layer.js", token_count=10)
    all_chunks = idx_chunks + [route_chunk, layer_chunk]

    # index.js dominates BM25, route and layer have lower scores
    results = [(c, 7.0) for c in idx_chunks] + [(route_chunk, 3.0), (layer_chunk, 2.5)]
    bundle = assemble_context(
        results,
        graph,
        all_chunks,
        "How does the middleware chain and next() function work?",
        token_budget=5000,
    )
    included_files = {rc.chunk.file_path for rc in bundle.chunks}
    assert "lib/router/index.js" in included_files
    assert "lib/router/route.js" in included_files, (
        "route.js must survive cutoff with max aggregation"
    )
    assert "lib/router/layer.js" in included_files, (
        "layer.js must survive cutoff with max aggregation"
    )


# ---------------------------------------------------------------------------
# Expansion file budget
# ---------------------------------------------------------------------------


def test_max_expansion_files_allows_8() -> None:
    """Up to 8 expansion files can enter the candidate pool."""
    from archex.serve.context import MAX_EXPANSION_FILES

    assert MAX_EXPANSION_FILES == 8


def test_file_score_cutoff_is_0_10() -> None:
    """FILE_SCORE_CUTOFF is 0.10 (lowered from 0.15 to reduce false elimination)."""
    from archex.serve.context import FILE_SCORE_CUTOFF

    assert FILE_SCORE_CUTOFF == 0.10
