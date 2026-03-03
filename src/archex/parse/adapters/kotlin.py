"""Kotlin parse adapter: extract symbols and imports from .kt/.kts files using tree-sitter."""

from __future__ import annotations

from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters._jvm_helpers import map_jvm_visibility, resolve_jvm_import
from archex.parse.adapters.ts_node import (
    ts_children as _children,
)
from archex.parse.adapters.ts_node import (
    ts_end_line as _end_line,
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
# Modifier helpers
# ---------------------------------------------------------------------------


def _extract_visibility(node: object, default: Visibility = Visibility.PUBLIC) -> Visibility:
    """Extract visibility from a node's modifiers child.

    *default* is returned when no visibility modifier is present.  Kotlin
    declarations default to PUBLIC (unlike Java's package-private).
    """
    for child in _children(node):
        if _type(child) != "modifiers":
            continue
        for mod in _children(child):
            mod_type = _type(mod)
            if mod_type == "visibility_modifier":
                for inner in _children(mod):
                    inner_text = _type(inner)
                    if inner_text in ("public", "private", "protected", "internal"):
                        return map_jvm_visibility(inner_text, default=default)
    return default


def _has_child_type(node: object, child_type: str) -> bool:
    """Check if a node has a direct child of the given type."""
    return any(_type(c) == child_type for c in _children(node))


def _is_interface(node: object) -> bool:
    """Check if a class_declaration node is an interface."""
    return _has_child_type(node, "interface")


# ---------------------------------------------------------------------------
# Extension function helpers
# ---------------------------------------------------------------------------


def _get_extension_receiver(node: object, source: bytes) -> str | None:
    """Return the receiver type text if this is an extension function, else None.

    Extension functions look like: fun ReceiverType.methodName(...)
    The AST has a user_type child, then a '.' child, then an identifier child.
    """
    children = _children(node)
    for i, child in enumerate(children):
        if (
            _type(child) == "."
            and i > 0
            and _type(children[i - 1]) in ("user_type", "nullable_type", "dynamic_type")
        ):
            return _text(children[i - 1], source)
    return None


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _build_function_signature(node: object, source: bytes, name: str) -> str:
    """Build a function signature string."""
    params_node = None
    for child in _named_children(node):
        if _type(child) == "function_value_parameters":
            params_node = child
            break
    params = _text(params_node, source) if params_node else "()"

    # Find return type (user_type, nullable_type after params)
    ret_type = ""
    past_params = False
    for child in _children(node):
        ct = _type(child)
        if ct == "function_value_parameters":
            past_params = True
            continue
        if past_params and ct in (
            "user_type",
            "nullable_type",
            "dynamic_type",
        ):
            ret_type = _text(child, source)
            break

    if ret_type:
        return f"fun {name}{params}: {ret_type}"
    return f"fun {name}{params}"


# ---------------------------------------------------------------------------
# Symbol extraction — helpers
# ---------------------------------------------------------------------------


def _find_class_body(node: object) -> object | None:
    """Return the class_body child of a class/object/companion node."""
    for child in _named_children(node):
        if _type(child) == "class_body":
            return child
    return None


def _extract_property(
    node: object, source: bytes, file_path: str, parent_name: str
) -> Symbol | None:
    """Extract a property_declaration as a VARIABLE symbol."""
    vis = _extract_visibility(node)
    # variable_declaration holds the name identifier
    var_decl = None
    for child in _named_children(node):
        if _type(child) == "variable_declaration":
            var_decl = child
            break
    if var_decl is None:
        return None
    name_node = None
    for child in _named_children(var_decl):
        if _type(child) == "identifier":
            name_node = child
            break
    if name_node is None:
        return None
    name = _text(name_node, source)
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


def _extract_function(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str | None,
    default_visibility: Visibility = Visibility.PUBLIC,
) -> Symbol | None:
    """Extract a function_declaration as FUNCTION (top-level) or METHOD (member).

    Handles both regular and extension functions.
    """
    receiver = _get_extension_receiver(node, source)
    name_node = None
    for child in _children(node):
        if _type(child) == "identifier":
            name_node = child
            break
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _extract_visibility(node, default=default_visibility)

    if receiver is not None:
        # Extension function
        effective_parent = receiver
        qualified = f"{receiver}.{name}"
        sig = _build_function_signature(node, source, name)
        return Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.METHOD,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            signature=sig,
            parent=effective_parent,
        )

    if parent_name is not None:
        # Class/object member → METHOD
        qualified = f"{parent_name}.{name}"
        sig = _build_function_signature(node, source, name)
        return Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.METHOD,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            signature=sig,
            parent=parent_name,
        )

    # Top-level function → FUNCTION
    sig = _build_function_signature(node, source, name)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
    )


def _extract_companion_object(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str,
) -> list[Symbol]:
    """Extract a companion_object and its members."""
    qualified = f"{parent_name}.Companion"
    symbols: list[Symbol] = [
        Symbol(
            name="Companion",
            qualified_name=qualified,
            kind=SymbolKind.CLASS,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=Visibility.PUBLIC,
            parent=parent_name,
        )
    ]
    body = _find_class_body(node)
    if body is not None:
        symbols.extend(_extract_class_body_members(body, source, file_path, qualified))
    return symbols


def _extract_class_body_members(
    body: object,
    source: bytes,
    file_path: str,
    parent_name: str,
    *,
    default_visibility: Visibility = Visibility.PUBLIC,
) -> list[Symbol]:
    """Extract all member symbols from a class/interface/object body."""
    symbols: list[Symbol] = []

    for child in _named_children(body):
        ct = _type(child)

        if ct == "function_declaration":
            sym = _extract_function(child, source, file_path, parent_name, default_visibility)
            if sym:
                symbols.append(sym)
        elif ct == "property_declaration":
            sym = _extract_property(child, source, file_path, parent_name)
            if sym:
                symbols.append(sym)
        elif ct == "class_declaration":
            symbols.extend(_extract_class_symbols(child, source, file_path, parent_name))
        elif ct == "object_declaration":
            symbols.extend(_extract_object_symbols(child, source, file_path, parent_name))
        elif ct == "companion_object":
            symbols.extend(_extract_companion_object(child, source, file_path, parent_name))

    return symbols


def _extract_class_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract a class_declaration (class, data class, sealed class, interface)."""
    name_node = node  # will overwrite below
    # The name field should be the identifier
    for child in _children(node):
        if _type(child) == "identifier":
            name_node = child
            break
    else:
        return []

    name = _text(name_node, source)
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)

    kind = SymbolKind.INTERFACE if _is_interface(node) else SymbolKind.CLASS

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

    body = _find_class_body(node)
    if body is not None:
        member_default_vis = (
            Visibility.PUBLIC if kind == SymbolKind.INTERFACE else Visibility.PUBLIC
        )
        symbols.extend(
            _extract_class_body_members(
                body,
                source,
                file_path,
                qualified,
                default_visibility=member_default_vis,
            )
        )

    return symbols


def _extract_object_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract an object_declaration (singleton)."""
    name_node = None
    for child in _named_children(node):
        if _type(child) == "identifier":
            name_node = child
            break
    if name_node is None:
        return []

    name = _text(name_node, source)
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)

    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.CLASS,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]

    body = _find_class_body(node)
    if body is not None:
        symbols.extend(_extract_class_body_members(body, source, file_path, qualified))

    return symbols


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _parse_kotlin_import(node: object, source: bytes, file_path: str) -> ImportStatement | None:
    """Parse a single import node (Kotlin uses 'import' node type)."""
    line = _start_line(node)
    module_text = ""
    alias: str | None = None
    is_wildcard = False

    has_wildcard_star = False
    qual_id_text = ""

    for child in _children(node):
        ct = _type(child)
        if ct == "qualified_identifier":
            qual_id_text = _text(child, source)
        elif ct == "*":
            has_wildcard_star = True
        elif ct == "identifier":
            # alias: 'import X as Alias' — identifier after 'as' keyword
            alias = _text(child, source)

    if not qual_id_text:
        return None

    if has_wildcard_star:
        module_text = qual_id_text + ".*"
        is_wildcard = True
    else:
        module_text = qual_id_text

    symbols: list[str] = []
    # Wildcard imports have no specific symbol
    if not is_wildcard and alias is None:
        # Plain import: module = full path, no symbols list
        pass

    return ImportStatement(
        module=module_text,
        symbols=symbols,
        file_path=file_path,
        line=line,
        is_relative=False,
    )


# ---------------------------------------------------------------------------
# KotlinAdapter
# ---------------------------------------------------------------------------


class KotlinAdapter:
    """Language adapter for Kotlin source files."""

    @property
    def language_id(self) -> str:
        return "kotlin"

    @property
    def file_extensions(self) -> list[str]:
        return [".kt", ".kts"]

    @property
    def tree_sitter_name(self) -> str:
        return "kotlin"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a Kotlin parse tree."""
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []

        for child in _named_children(root):
            ct = _type(child)
            if ct == "class_declaration":
                symbols.extend(_extract_class_symbols(child, source, file_path))
            elif ct == "object_declaration":
                symbols.extend(_extract_object_symbols(child, source, file_path))
            elif ct == "function_declaration":
                sym = _extract_function(child, source, file_path, parent_name=None)
                if sym:
                    symbols.append(sym)
            elif ct == "type_alias":
                # typealias Name = Type → TYPE symbol
                name_node = None
                for c in _named_children(child):
                    if _type(c) == "identifier":
                        name_node = c
                        break
                if name_node is not None:
                    name = _text(name_node, source)
                    vis = _extract_visibility(child)
                    symbols.append(
                        Symbol(
                            name=name,
                            qualified_name=name,
                            kind=SymbolKind.TYPE,
                            file_path=file_path,
                            start_line=_start_line(child),
                            end_line=_end_line(child),
                            visibility=vis,
                        )
                    )

        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import declarations from a Kotlin parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _named_children(root):
            if _type(child) == "import":
                imp = _parse_kotlin_import(child, source, file_path)
                if imp:
                    imports.append(imp)

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Kotlin import to a local file, or None if external."""
        return resolve_jvm_import(imp.module, file_map, extensions=(".kt", ".java"))

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect Kotlin entry points: top-level main(), framework annotations."""
        entry_points: list[str] = []
        entry_markers = (
            "fun main()",
            "fun main(args",
            "@SpringBootApplication",
            "@Test",
            "@Composable",
        )

        for f in files:
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if any(marker in content for marker in entry_markers):
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
