"""C# parse adapter: extract symbols and imports from .cs files using tree-sitter."""

from __future__ import annotations

from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.ts_node import (
    ts_children as _children,
)
from archex.parse.adapters.ts_node import (
    ts_end_line as _end_line,
)
from archex.parse.adapters.ts_node import (
    ts_field as _field,
)
from archex.parse.adapters.ts_node import (
    ts_named_children as _named_children,
)
from archex.parse.adapters.ts_node import (
    ts_start_line as _start_line,
)
from archex.parse.adapters.ts_node import (
    ts_text as _text,
)
from archex.parse.adapters.ts_node import (
    ts_type as _type,
)

# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------

_RETURN_TYPE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "predefined_type",
        "identifier",
        "generic_name",
        "qualified_name",
        "array_type",
        "nullable_type",
        "tuple_type",
    }
)

_MODIFIER_SKIP: frozenset[str] = frozenset(
    {
        "async",
        "static",
        "virtual",
        "override",
        "abstract",
        "sealed",
        "extern",
        "readonly",
        "new",
        "partial",
    }
)


def _map_csharp_visibility(mods: list[str], default: Visibility) -> Visibility:
    if "public" in mods:
        return Visibility.PUBLIC
    if "private" in mods and "protected" in mods:
        return Visibility.PRIVATE
    if "protected" in mods and "internal" in mods:
        return Visibility.INTERNAL
    if "internal" in mods:
        return Visibility.INTERNAL
    if "protected" in mods:
        return Visibility.INTERNAL
    if "private" in mods:
        return Visibility.PRIVATE
    return default


def _get_modifiers(node: object, source: bytes) -> list[str]:
    """Return all modifier keyword strings for a node."""
    return [_text(c, source) for c in _children(node) if _type(c) == "modifier"]


def _is_const_or_static_readonly(node: object, source: bytes) -> bool:
    """Return True if a field has `const` or `static readonly` modifiers."""
    mods = _get_modifiers(node, source)
    if "const" in mods:
        return True
    return "static" in mods and "readonly" in mods


# ---------------------------------------------------------------------------
# Namespace extraction
# ---------------------------------------------------------------------------


def _extract_namespace(root: object, source: bytes) -> str | None:
    """Return the namespace name declared in the file, if any.

    Handles both block-scoped (``namespace Foo { ... }``) and
    file-scoped (``namespace Foo;``) declarations.
    """
    for child in _named_children(root):
        ct = _type(child)
        if ct in ("namespace_declaration", "file_scoped_namespace_declaration"):
            name_node = _field(child, "name")
            if name_node:
                return _text(name_node, source)
    return None


def _qualify(namespace: str | None, name: str) -> str:
    return f"{namespace}.{name}" if namespace else name


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _get_return_type(node: object, source: bytes) -> str:
    """Extract the return type from a method_declaration node.

    In C# tree-sitter, the return type appears as a named child *before*
    the method name identifier, not as a named field.
    """
    name_node = _field(node, "name")
    name_byte_start = _text(name_node, source) if name_node else None

    for child in _named_children(node):
        if _type(child) == "modifier":
            continue
        ct = _type(child)
        if ct in _RETURN_TYPE_NODE_TYPES:
            # Stop before we hit the name
            if name_node is not None:
                child_text = _text(child, source)
                if child_text == name_byte_start:
                    break
            return _text(child, source)
        if ct == "identifier":
            # Could be either return type or name depending on position
            if name_node is not None:
                node_text = _text(child, source)
                name_text = _text(name_node, source)
                if node_text != name_text:
                    return node_text
            break

    return "void"


def _build_method_signature(node: object, source: bytes, name: str, return_type: str) -> str:
    params_node = _field(node, "parameters")
    params = _text(params_node, source) if params_node else "()"
    return f"{return_type} {name}{params}"


# ---------------------------------------------------------------------------
# Symbol extraction helpers
# ---------------------------------------------------------------------------


def _extract_type_decl(
    node: object,
    source: bytes,
    file_path: str,
    kind: SymbolKind,
    namespace: str | None,
    parent_name: str | None,
) -> list[Symbol]:
    """Extract a type declaration (class, interface, struct, record, delegate)."""
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    ns_prefix = parent_name or namespace
    qualified = _qualify(ns_prefix, name)
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, Visibility.INTERNAL)

    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]

    body = _field(node, "body")
    if body is not None:
        default_vis = Visibility.PUBLIC if kind == SymbolKind.INTERFACE else Visibility.PRIVATE
        symbols.extend(
            _extract_body_members(
                body,
                source,
                file_path,
                qualified,
                default_visibility=default_vis,
            )
        )

    return symbols


def _extract_enum_symbols(
    node: object,
    source: bytes,
    file_path: str,
    namespace: str | None,
    parent_name: str | None,
) -> list[Symbol]:
    """Extract an enum declaration and its members."""
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    ns_prefix = parent_name or namespace
    qualified = _qualify(ns_prefix, name)
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, Visibility.INTERNAL)

    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.ENUM,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]

    body = _field(node, "body")
    if body is not None:
        for child in _named_children(body):
            if _type(child) == "enum_member_declaration":
                m_name_node = _field(child, "name")
                if m_name_node:
                    m_name = _text(m_name_node, source)
                    symbols.append(
                        Symbol(
                            name=m_name,
                            qualified_name=f"{qualified}.{m_name}",
                            kind=SymbolKind.CONSTANT,
                            file_path=file_path,
                            start_line=_start_line(child),
                            end_line=_end_line(child),
                            visibility=vis,
                            parent=qualified,
                        )
                    )

    return symbols


def _extract_method(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    default_visibility: Visibility = Visibility.PRIVATE,
) -> Symbol | None:
    """Extract a method_declaration node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, default_visibility)
    ret_type = _get_return_type(node, source)
    sig = _build_method_signature(node, source, name, ret_type)

    return Symbol(
        name=name,
        qualified_name=f"{parent_name}.{name}",
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
        parent=parent_name,
    )


def _extract_constructor(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    default_visibility: Visibility = Visibility.PRIVATE,
) -> Symbol | None:
    """Extract a constructor_declaration (treated as METHOD)."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, default_visibility)
    params_node = _field(node, "parameters")
    params = _text(params_node, source) if params_node else "()"
    sig = f"{name}{params}"

    return Symbol(
        name=name,
        qualified_name=f"{parent_name}.{name}",
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
        parent=parent_name,
    )


def _extract_field_symbols(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    is_event: bool = False,
) -> list[Symbol]:
    """Extract field_declaration or event_field_declaration symbols."""
    symbols: list[Symbol] = []
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, Visibility.PRIVATE)
    is_constant = (not is_event) and _is_const_or_static_readonly(node, source)
    kind = SymbolKind.VARIABLE if not is_constant else SymbolKind.CONSTANT

    for child in _named_children(node):
        if _type(child) != "variable_declaration":
            continue
        for declarator in _named_children(child):
            if _type(declarator) != "variable_declarator":
                continue
            name_node = _field(declarator, "name")
            if name_node is None:
                continue
            name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=f"{parent_name}.{name}",
                    kind=kind,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                    parent=parent_name,
                )
            )

    return symbols


def _extract_property(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    default_visibility: Visibility = Visibility.PRIVATE,
) -> Symbol | None:
    """Extract a property_declaration node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    mods = _get_modifiers(node, source)
    vis = _map_csharp_visibility(mods, default_visibility)

    return Symbol(
        name=name,
        qualified_name=f"{parent_name}.{name}",
        kind=SymbolKind.VARIABLE,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        parent=parent_name,
    )


def _extract_body_members(
    body: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    *,
    default_visibility: Visibility = Visibility.PRIVATE,
) -> list[Symbol]:
    """Extract all member symbols from a type body (declaration_list)."""
    symbols: list[Symbol] = []

    for child in _named_children(body):
        ct = _type(child)

        if ct == "method_declaration":
            sym = _extract_method(child, source, file_path, parent_name, default_visibility)
            if sym:
                symbols.append(sym)
        elif ct == "constructor_declaration":
            sym = _extract_constructor(child, source, file_path, parent_name, default_visibility)
            if sym:
                symbols.append(sym)
        elif ct == "field_declaration":
            symbols.extend(_extract_field_symbols(child, source, file_path, parent_name))
        elif ct == "event_field_declaration":
            symbols.extend(
                _extract_field_symbols(child, source, file_path, parent_name, is_event=True)
            )
        elif ct == "property_declaration":
            sym = _extract_property(child, source, file_path, parent_name, default_visibility)
            if sym:
                symbols.append(sym)
        elif ct == "class_declaration":
            symbols.extend(
                _extract_type_decl(
                    child,
                    source,
                    file_path,
                    SymbolKind.CLASS,
                    namespace=None,
                    parent_name=parent_name,
                )
            )
        elif ct == "interface_declaration":
            symbols.extend(
                _extract_type_decl(
                    child,
                    source,
                    file_path,
                    SymbolKind.INTERFACE,
                    namespace=None,
                    parent_name=parent_name,
                )
            )
        elif ct == "struct_declaration":
            symbols.extend(
                _extract_type_decl(
                    child,
                    source,
                    file_path,
                    SymbolKind.TYPE,
                    namespace=None,
                    parent_name=parent_name,
                )
            )
        elif ct == "enum_declaration":
            symbols.extend(
                _extract_enum_symbols(
                    child, source, file_path, namespace=None, parent_name=parent_name
                )
            )
        elif ct == "record_declaration":
            symbols.extend(
                _extract_type_decl(
                    child,
                    source,
                    file_path,
                    SymbolKind.CLASS,
                    namespace=None,
                    parent_name=parent_name,
                )
            )
        elif ct == "delegate_declaration":
            name_node = _field(child, "name")
            if name_node:
                name = _text(name_node, source)
                mods = _get_modifiers(child, source)
                vis = _map_csharp_visibility(mods, Visibility.PRIVATE)
                symbols.append(
                    Symbol(
                        name=name,
                        qualified_name=f"{parent_name}.{name}",
                        kind=SymbolKind.TYPE,
                        file_path=file_path,
                        start_line=_start_line(child),
                        end_line=_end_line(child),
                        visibility=vis,
                        parent=parent_name,
                    )
                )

    return symbols


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


def _extract_top_level_type(
    node: object,
    source: bytes,
    file_path: str,
    namespace: str | None,
) -> list[Symbol]:
    """Extract a top-level type declaration (class, interface, struct, record, delegate, enum)."""
    ct = _type(node)
    if ct == "class_declaration":
        return _extract_type_decl(
            node, source, file_path, SymbolKind.CLASS, namespace=namespace, parent_name=None
        )
    if ct == "interface_declaration":
        return _extract_type_decl(
            node, source, file_path, SymbolKind.INTERFACE, namespace=namespace, parent_name=None
        )
    if ct == "struct_declaration":
        return _extract_type_decl(
            node, source, file_path, SymbolKind.TYPE, namespace=namespace, parent_name=None
        )
    if ct == "record_declaration":
        return _extract_type_decl(
            node, source, file_path, SymbolKind.CLASS, namespace=namespace, parent_name=None
        )
    if ct == "enum_declaration":
        return _extract_enum_symbols(node, source, file_path, namespace=namespace, parent_name=None)
    if ct == "delegate_declaration":
        name_node = _field(node, "name")
        if name_node is None:
            return []
        name = _text(name_node, source)
        qualified = _qualify(namespace, name)
        mods = _get_modifiers(node, source)
        vis = _map_csharp_visibility(mods, Visibility.INTERNAL)
        return [
            Symbol(
                name=name,
                qualified_name=qualified,
                kind=SymbolKind.TYPE,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=vis,
                parent=None,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _parse_using_directive(node: object, source: bytes, file_path: str) -> ImportStatement | None:
    """Parse a single using_directive node into an ImportStatement."""
    line = _start_line(node)
    is_static = False
    module: str | None = None

    for child in _children(node):
        ct = _type(child)
        if ct == "static":
            is_static = True
        elif ct in ("identifier", "qualified_name"):
            module = _text(child, source)

    if module is None:
        return None

    return ImportStatement(
        module=module,
        symbols=[] if not is_static else [],
        file_path=file_path,
        line=line,
        is_relative=False,
    )


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _resolve_csharp_import(module: str, file_map: dict[str, str]) -> str | None:
    """Resolve a `using Namespace` directive to a local .cs file.

    Matches the last segment of the namespace against filenames in the file map.
    Returns the best match or None if it appears to be an external namespace.
    """
    parts = module.split(".")
    # External .NET namespaces never resolve
    if parts[0] in ("System", "Microsoft", "NUnit", "Xunit", "MSTest"):
        return None

    # Search for a .cs file whose directory path ends with the namespace parts
    candidates: list[tuple[int, str]] = []
    for key, abs_path in file_map.items():
        if not key.endswith(".cs"):
            continue
        import os

        dir_path = os.path.dirname(key).replace("\\", "/")
        dir_segments = [s for s in dir_path.split("/") if s]

        score = 0
        for i, part in enumerate(reversed(parts)):
            idx = len(dir_segments) - 1 - i
            if idx >= 0 and dir_segments[idx] == part:
                score += 1
            else:
                break

        if score > 0:
            candidates.append((score, abs_path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return None


# ---------------------------------------------------------------------------
# CSharpAdapter
# ---------------------------------------------------------------------------


def _is_top_level_statements(content: str) -> bool:
    """Detect C# 9+ top-level statements: executable code without a class/struct wrapper."""
    if "class " in content or "struct " in content:
        return False
    return "Console." in content or "await " in content


class CSharpAdapter:
    """Language adapter for C# source files."""

    @property
    def language_id(self) -> str:
        return "csharp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cs"]

    @property
    def tree_sitter_name(self) -> str:
        return "c_sharp"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a C# parse tree."""
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []

        namespace = _extract_namespace(root, source)

        for child in _named_children(root):
            ct = _type(child)

            if ct in ("namespace_declaration", "file_scoped_namespace_declaration"):
                # Extract from namespace body
                body = _field(child, "body")
                if body is not None:
                    for decl in _named_children(body):
                        symbols.extend(_extract_top_level_type(decl, source, file_path, namespace))
                else:
                    # file-scoped namespace: siblings of the namespace node are the declarations
                    pass
            elif ct in (
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "record_declaration",
                "enum_declaration",
                "delegate_declaration",
            ):
                # Top-level declarations (file-scoped namespace or no namespace)
                symbols.extend(_extract_top_level_type(child, source, file_path, namespace))

        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all using directives from a C# parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _named_children(root):
            if _type(child) == "using_directive":
                imp = _parse_using_directive(child, source, file_path)
                if imp:
                    imports.append(imp)

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a C# using directive to a local file, or None if external."""
        return _resolve_csharp_import(imp.module, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect C# entry points: Main methods and test attributes."""
        entry_points: list[str] = []
        entry_markers = (
            "static void Main(",
            "static async Task Main(",
            "[Fact]",
            "[Test]",
            "[TestMethod]",
        )

        for f in files:
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if any(marker in content for marker in entry_markers) or _is_top_level_statements(
                content
            ):
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
