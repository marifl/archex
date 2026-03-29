"""Tests for structural context breadcrumbs: generation, BM25 indexing, and vector embedding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from archex.index.bm25 import BM25Index
from archex.index.store import IndexStore
from archex.models import CodeChunk, ImportStatement, ParsedFile, Symbol, SymbolKind, Visibility
from archex.pipeline.chunker import (
    ASTChunker,
    _file_path_to_module,  # pyright: ignore[reportPrivateUsage]
    _resolve_parent_kinds,  # pyright: ignore[reportPrivateUsage]
    build_breadcrumbs,
)

# ---------------------------------------------------------------------------
# _file_path_to_module tests
# ---------------------------------------------------------------------------


class TestFilePathToModule:
    def test_python_file_with_src_prefix(self) -> None:
        assert _file_path_to_module("src/archex/pipeline/chunker.py") == "archex.pipeline.chunker"

    def test_python_file_no_prefix(self) -> None:
        assert _file_path_to_module("archex/models.py") == "archex.models"

    def test_init_file_stripped(self) -> None:
        assert _file_path_to_module("src/archex/pipeline/__init__.py") == "archex.pipeline"

    def test_index_file_stripped(self) -> None:
        assert _file_path_to_module("lib/components/index.tsx") == "components"

    def test_windows_path_normalized(self) -> None:
        assert _file_path_to_module("src\\archex\\models.py") == "archex.models"

    def test_java_file(self) -> None:
        assert _file_path_to_module("src/com/example/Service.java") == "com.example.Service"

    def test_non_source_extension_dotted(self) -> None:
        result = _file_path_to_module("data/config.toml")
        assert result == "data.config.toml"

    def test_lib_prefix_stripped(self) -> None:
        assert _file_path_to_module("lib/utils/helper.rb") == "utils.helper"

    def test_app_prefix_stripped(self) -> None:
        assert _file_path_to_module("app/models/user.py") == "models.user"


# ---------------------------------------------------------------------------
# build_breadcrumbs tests
# ---------------------------------------------------------------------------


class TestBuildBreadcrumbs:
    def test_file_level_chunk_module_only(self) -> None:
        result = build_breadcrumbs("src/archex/models.py", symbol=None)
        assert result == "module: archex.models"

    def test_top_level_function(self) -> None:
        sym = Symbol(
            name="hello",
            qualified_name="hello",
            kind=SymbolKind.FUNCTION,
            file_path="src/utils.py",
            start_line=1,
            end_line=3,
        )
        result = build_breadcrumbs("src/utils.py", sym)
        assert result == "module: utils > function: hello"

    def test_top_level_class(self) -> None:
        sym = Symbol(
            name="Greeter",
            qualified_name="Greeter",
            kind=SymbolKind.CLASS,
            file_path="src/app/greeter.py",
            start_line=1,
            end_line=10,
        )
        result = build_breadcrumbs("src/app/greeter.py", sym)
        assert result == "module: app.greeter > class: Greeter"

    def test_method_with_class_parent(self) -> None:
        all_symbols = [
            Symbol(
                name="Greeter",
                qualified_name="Greeter",
                kind=SymbolKind.CLASS,
                file_path="greeter.py",
                start_line=1,
                end_line=10,
            ),
            Symbol(
                name="Greeter.greet",
                qualified_name="Greeter.greet",
                kind=SymbolKind.METHOD,
                file_path="greeter.py",
                start_line=5,
                end_line=8,
                parent="Greeter",
            ),
        ]
        result = build_breadcrumbs("greeter.py", all_symbols[1], all_symbols)
        assert result == "module: greeter > class: Greeter > method: greet"

    def test_nested_class_method(self) -> None:
        all_symbols = [
            Symbol(
                name="Outer",
                qualified_name="Outer",
                kind=SymbolKind.CLASS,
                file_path="nested.py",
                start_line=1,
                end_line=20,
            ),
            Symbol(
                name="Inner",
                qualified_name="Outer.Inner",
                kind=SymbolKind.CLASS,
                file_path="nested.py",
                start_line=5,
                end_line=15,
            ),
            Symbol(
                name="do_thing",
                qualified_name="Outer.Inner.do_thing",
                kind=SymbolKind.METHOD,
                file_path="nested.py",
                start_line=10,
                end_line=14,
                parent="Outer.Inner",
            ),
        ]
        result = build_breadcrumbs("nested.py", all_symbols[2], all_symbols)
        assert result == "module: nested > class: Outer > class: Inner > method: do_thing"

    def test_method_without_all_symbols_defaults_class(self) -> None:
        sym = Symbol(
            name="Greeter.greet",
            qualified_name="Greeter.greet",
            kind=SymbolKind.METHOD,
            file_path="greeter.py",
            start_line=5,
            end_line=8,
            parent="Greeter",
        )
        result = build_breadcrumbs("greeter.py", sym, all_symbols=None)
        assert result == "module: greeter > class: Greeter > method: greet"

    def test_symbol_no_qualified_name(self) -> None:
        sym = Symbol(
            name="anon",
            qualified_name="",
            kind=SymbolKind.FUNCTION,
            file_path="test.py",
            start_line=1,
            end_line=2,
        )
        result = build_breadcrumbs("test.py", sym)
        assert result == "module: test > function: anon"


# ---------------------------------------------------------------------------
# _resolve_parent_kinds tests
# ---------------------------------------------------------------------------


class TestResolveParentKinds:
    def test_no_symbols_defaults_to_class(self) -> None:
        result = _resolve_parent_kinds(["Foo", "Bar"], "test.py", None)
        assert result == ["class", "class"]

    def test_resolves_known_symbol(self) -> None:
        syms = [
            Symbol(
                name="Outer",
                qualified_name="Outer",
                kind=SymbolKind.CLASS,
                file_path="test.py",
                start_line=1,
                end_line=20,
            ),
        ]
        result = _resolve_parent_kinds(["Outer"], "test.py", syms)
        assert result == ["class"]

    def test_unknown_parent_defaults_to_class(self) -> None:
        result = _resolve_parent_kinds(["Unknown"], "test.py", [])
        assert result == ["class"]


# ---------------------------------------------------------------------------
# ASTChunker breadcrumb integration tests
# ---------------------------------------------------------------------------

SOURCE_WITH_CLASS = b"""\
import os

class QuerySet:
    def filter(self, **kwargs):
        return self._clone()

    def exclude(self, **kwargs):
        return self._clone()
"""

SYMBOLS_WITH_CLASS = [
    Symbol(
        name="QuerySet",
        qualified_name="QuerySet",
        kind=SymbolKind.CLASS,
        file_path="src/django/db/models/query.py",
        start_line=3,
        end_line=3,
        visibility=Visibility.PUBLIC,
    ),
    Symbol(
        name="QuerySet.filter",
        qualified_name="QuerySet.filter",
        kind=SymbolKind.METHOD,
        file_path="src/django/db/models/query.py",
        start_line=4,
        end_line=5,
        visibility=Visibility.PUBLIC,
        parent="QuerySet",
    ),
    Symbol(
        name="QuerySet.exclude",
        qualified_name="QuerySet.exclude",
        kind=SymbolKind.METHOD,
        file_path="src/django/db/models/query.py",
        start_line=7,
        end_line=8,
        visibility=Visibility.PUBLIC,
        parent="QuerySet",
    ),
]

PARSED_WITH_CLASS = ParsedFile(
    path="src/django/db/models/query.py",
    language="python",
    symbols=SYMBOLS_WITH_CLASS,
    imports=[ImportStatement(module="os", file_path="src/django/db/models/query.py", line=1)],
    lines=8,
)


class TestChunkerBreadcrumbIntegration:
    def test_method_chunk_has_breadcrumbs(self) -> None:
        chunker = ASTChunker()
        chunks = chunker.chunk_file(PARSED_WITH_CLASS, SOURCE_WITH_CLASS)

        filter_chunks = [c for c in chunks if c.symbol_name == "QuerySet.filter"]
        assert len(filter_chunks) == 1
        bc = filter_chunks[0].breadcrumbs
        assert "module: django.db.models.query" in bc
        assert "class: QuerySet" in bc
        assert "method: filter" in bc

    def test_class_chunk_has_breadcrumbs(self) -> None:
        chunker = ASTChunker()
        chunks = chunker.chunk_file(PARSED_WITH_CLASS, SOURCE_WITH_CLASS)

        class_chunks = [c for c in chunks if c.symbol_name == "QuerySet"]
        assert len(class_chunks) == 1
        bc = class_chunks[0].breadcrumbs
        assert "module: django.db.models.query" in bc
        assert "class: QuerySet" in bc

    def test_file_level_chunk_has_module_breadcrumb(self) -> None:
        chunker = ASTChunker()
        chunks = chunker.chunk_file(PARSED_WITH_CLASS, SOURCE_WITH_CLASS)

        file_chunks = [c for c in chunks if c.symbol_name is None]
        for fc in file_chunks:
            assert "module: django.db.models.query" in fc.breadcrumbs

    def test_all_chunks_have_breadcrumbs(self) -> None:
        chunker = ASTChunker()
        chunks = chunker.chunk_file(PARSED_WITH_CLASS, SOURCE_WITH_CLASS)
        for chunk in chunks:
            assert chunk.breadcrumbs, f"Chunk {chunk.id} has empty breadcrumbs"


# ---------------------------------------------------------------------------
# BM25 breadcrumb indexing tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bm25_store(tmp_path: Path) -> Generator[IndexStore, None, None]:
    store = IndexStore(str(tmp_path / "test.db"))
    yield store
    store.close()


CHUNKS_WITH_BREADCRUMBS = [
    CodeChunk(
        id="query.py:QuerySet.filter:4",
        content="def filter(self, **kwargs):\n    return self._clone()",
        file_path="src/django/db/models/query.py",
        start_line=4,
        end_line=5,
        symbol_name="QuerySet.filter",
        symbol_kind=SymbolKind.METHOD,
        language="python",
        token_count=15,
        breadcrumbs="module: django.db.models.query > class: QuerySet > method: filter",
    ),
    CodeChunk(
        id="views.py:index_view:1",
        content="def index_view(request):\n    return render(request, 'index.html')",
        file_path="src/app/views.py",
        start_line=1,
        end_line=2,
        symbol_name="index_view",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=18,
        breadcrumbs="module: app.views > function: index_view",
    ),
    CodeChunk(
        id="utils.py:calculate:1",
        content="def calculate(a: int, b: int) -> int:\n    return a + b",
        file_path="src/utils.py",
        start_line=1,
        end_line=2,
        symbol_name="calculate",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=15,
        breadcrumbs="module: utils > function: calculate",
    ),
]


class TestBM25BreadcrumbIndexing:
    def test_breadcrumbs_indexed_in_fts5(self, bm25_store: IndexStore) -> None:
        bm25_store.insert_chunks(CHUNKS_WITH_BREADCRUMBS)
        index = BM25Index(bm25_store)
        index.build(CHUNKS_WITH_BREADCRUMBS)

        results = index.search("QuerySet", top_k=5)
        chunk_ids = [c.id for c, _ in results]
        assert "query.py:QuerySet.filter:4" in chunk_ids

    def test_breadcrumbs_module_path_searchable(self, bm25_store: IndexStore) -> None:
        bm25_store.insert_chunks(CHUNKS_WITH_BREADCRUMBS)
        index = BM25Index(bm25_store)
        index.build(CHUNKS_WITH_BREADCRUMBS)

        results = index.search("django models query", top_k=5)
        chunk_ids = [c.id for c, _ in results]
        assert "query.py:QuerySet.filter:4" in chunk_ids

    def test_breadcrumbs_empty_handled(self, bm25_store: IndexStore) -> None:
        chunks = [
            CodeChunk(
                id="bare.py:foo:1",
                content="def foo(): pass",
                file_path="bare.py",
                start_line=1,
                end_line=1,
                symbol_name="foo",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=5,
                breadcrumbs="",
            ),
        ]
        bm25_store.insert_chunks(chunks)
        index = BM25Index(bm25_store)
        index.build(chunks)
        results = index.search("foo", top_k=5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Vector embedding breadcrumb tests
# ---------------------------------------------------------------------------


class TestVectorBreadcrumbEmbedding:
    def test_chunk_embedding_text_with_breadcrumbs(self) -> None:
        from archex.index.vector import _chunk_embedding_text  # pyright: ignore[reportPrivateUsage]

        chunk = CodeChunk(
            id="test:1",
            content="def filter(self): pass",
            file_path="test.py",
            start_line=1,
            end_line=1,
            language="python",
            breadcrumbs="module: test > method: filter",
        )
        text = _chunk_embedding_text(chunk)
        assert text.startswith("module: test > method: filter\n")
        assert "def filter(self): pass" in text

    def test_chunk_embedding_text_no_breadcrumbs(self) -> None:
        from archex.index.vector import _chunk_embedding_text  # pyright: ignore[reportPrivateUsage]

        chunk = CodeChunk(
            id="test:1",
            content="x = 1",
            file_path="test.py",
            start_line=1,
            end_line=1,
            language="python",
            breadcrumbs="",
        )
        text = _chunk_embedding_text(chunk)
        assert text == "x = 1"
