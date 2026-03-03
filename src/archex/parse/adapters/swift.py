"""Swift parse adapter: extract symbols and imports from .swift files using tree-sitter."""

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

_VISIBILITY_MAP: dict[str, Visibility] = {
    "open": Visibility.PUBLIC,
    "public": Visibility.PUBLIC,
    "internal": Visibility.INTERNAL,
    "fileprivate": Visibility.PRIVATE,
    "private": Visibility.PRIVATE,
}


def _extract_visibility(node: object, default: Visibility = Visibility.INTERNAL) -> Visibility:
    """Extract visibility from a node's modifiers child.

    Swift default access level is internal when no modifier is present.
    """
    for child in _children(node):
        if _type(child) != "modifiers":
            continue
        for mod in _children(child):
            if _type(mod) != "visibility_modifier":
                continue
            for kw in _children(mod):
                kw_text = _type(kw)
                if kw_text in _VISIBILITY_MAP:
                    return _VISIBILITY_MAP[kw_text]
    return default


def _has_attribute(node: object, source: bytes, attr_name: str) -> bool:
    """Check if a node has a specific attribute (e.g. @main, @testable)."""
    for child in _children(node):
        if _type(child) != "modifiers":
            continue
        for mod in _children(child):
            if _type(mod) != "attribute":
                continue
            for attr_child in _named_children(mod):
                if _text(attr_child, source).strip("@ ") == attr_name:
                    return True
    return False


def _class_keyword(node: object) -> str:
    """Return the keyword used in a class_declaration (class/struct/enum/extension/actor)."""
    for child in _children(node):
        ct = _type(child)
        if ct in ("class", "struct", "enum", "extension", "actor"):
            return ct
    return "class"


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def _get_name(node: object, source: bytes) -> str | None:
    """Get type name from a type_identifier child."""
    for child in _named_children(node):
        if _type(child) == "type_identifier":
            return _text(child, source)
    return None


def _get_extension_name(node: object, source: bytes) -> str | None:
    """Get extended type name from an extension class_declaration."""
    for child in _named_children(node):
        if _type(child) == "user_type":
            # user_type > type_identifier
            for gc in _named_children(child):
                if _type(gc) == "type_identifier":
                    return _text(gc, source)
    return None


def _build_method_signature(node: object, source: bytes, name: str) -> str:
    """Build a function signature by extracting params text between ( and )."""
    raw = _text(node, source)
    # Find first ( ... ) span  for the parameter list
    paren_start = raw.find("(")
    if paren_start == -1:
        return f"{name}()"
    depth = 0
    paren_end = paren_start
    for idx, ch in enumerate(raw[paren_start:], paren_start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                paren_end = idx
                break
    params = raw[paren_start + 1 : paren_end].strip()
    # Capture return type if present
    after = raw[paren_end + 1 :].strip()
    ret = ""
    if after.startswith("->"):
        # take up to first { or end
        ret_part = after[2:].strip()
        brace = ret_part.find("{")
        ret = ret_part[:brace].strip() if brace != -1 else ret_part.strip()
    if ret:
        return f"{name}({params}) -> {ret}"
    return f"{name}({params})"


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


def _extract_class_declaration(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str | None = None,
) -> list[Symbol]:
    """Dispatch a class_declaration node to the appropriate extractor."""
    kw = _class_keyword(node)
    if kw == "extension":
        return _extract_extension_symbols(node, source, file_path)
    if kw == "enum":
        return _extract_enum_symbols(node, source, file_path, parent_name)
    if kw == "struct":
        return _extract_struct_symbols(node, source, file_path, parent_name)
    # class or actor → CLASS
    return _extract_class_symbols(node, source, file_path, parent_name)


def _extract_class_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract a class (or actor) declaration and its members."""
    name = _get_name(node, source)
    if name is None:
        return []
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

    body = _get_body(node)
    if body is not None:
        symbols.extend(_extract_body_members(body, source, file_path, qualified))

    return symbols


def _extract_struct_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract a struct declaration and its members."""
    name = _get_name(node, source)
    if name is None:
        return []
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)
    # @main struct → CLASS kind (it acts as an entry point class)
    kind = SymbolKind.CLASS if _has_attribute(node, source, "main") else SymbolKind.TYPE

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

    body = _get_body(node)
    if body is not None:
        symbols.extend(_extract_body_members(body, source, file_path, qualified))

    return symbols


def _extract_protocol_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract a protocol declaration and its requirements."""
    name = _get_name(node, source)
    if name is None:
        return []
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)

    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.INTERFACE,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]

    # Extract protocol body members
    for child in _named_children(node):
        if _type(child) == "protocol_body":
            symbols.extend(_extract_protocol_body_members(child, source, file_path, qualified))
            break

    return symbols


def _extract_protocol_body_members(
    body: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract method/property requirements from a protocol body."""
    symbols: list[Symbol] = []
    for child in _named_children(body):
        ct = _type(child)
        if ct == "protocol_function_declaration":
            name_node = _get_simple_identifier(child, source)
            if name_node:
                sig = _build_method_signature(child, source, name_node)
                symbols.append(
                    Symbol(
                        name=name_node,
                        qualified_name=f"{parent_name}.{name_node}",
                        kind=SymbolKind.METHOD,
                        file_path=file_path,
                        start_line=_start_line(child),
                        end_line=_end_line(child),
                        visibility=Visibility.PUBLIC,
                        signature=sig,
                        parent=parent_name,
                    )
                )
        elif ct == "protocol_property_declaration":
            pname = _get_protocol_property_name(child, source)
            if pname:
                symbols.append(
                    Symbol(
                        name=pname,
                        qualified_name=f"{parent_name}.{pname}",
                        kind=SymbolKind.VARIABLE,
                        file_path=file_path,
                        start_line=_start_line(child),
                        end_line=_end_line(child),
                        visibility=Visibility.PUBLIC,
                        parent=parent_name,
                    )
                )
    return symbols


def _get_protocol_property_name(node: object, source: bytes) -> str | None:
    """Extract property name from protocol_property_declaration."""
    # pattern > simple_identifier
    for child in _named_children(node):
        ct = _type(child)
        if ct == "pattern":
            for gc in _named_children(child):
                if _type(gc) == "simple_identifier":
                    return _text(gc, source)
            return _text(child, source)
        if ct == "simple_identifier":
            return _text(child, source)
    return None


def _extract_enum_symbols(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> list[Symbol]:
    """Extract an enum declaration and its cases."""
    name = _get_name(node, source)
    if name is None:
        return []
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)

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

    for child in _named_children(node):
        if _type(child) == "enum_class_body":
            symbols.extend(_extract_enum_body(child, source, file_path, qualified, vis))
            break

    return symbols


def _extract_enum_body(
    body: object, source: bytes, file_path: str, parent_name: str, parent_vis: Visibility
) -> list[Symbol]:
    """Extract cases and methods from an enum body."""
    symbols: list[Symbol] = []
    for child in _named_children(body):
        ct = _type(child)
        if ct == "enum_entry":
            # case name (possibly with associated values)
            cname = _get_simple_identifier(child, source)
            if cname:
                symbols.append(
                    Symbol(
                        name=cname,
                        qualified_name=f"{parent_name}.{cname}",
                        kind=SymbolKind.CONSTANT,
                        file_path=file_path,
                        start_line=_start_line(child),
                        end_line=_end_line(child),
                        visibility=parent_vis,
                        parent=parent_name,
                    )
                )
        elif ct == "function_declaration":
            sym = _extract_function(child, source, file_path, parent_name)
            if sym:
                symbols.append(sym)
    return symbols


def _extract_extension_symbols(node: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract symbols from an extension declaration.

    Members get qualified_name = "ExtendedType.member", parent = extended type.
    The extension itself is not emitted as a symbol.
    """
    ext_name = _get_extension_name(node, source)
    if ext_name is None:
        return []

    symbols: list[Symbol] = []
    body = _get_body(node)
    if body is not None:
        symbols.extend(_extract_body_members(body, source, file_path, ext_name))

    return symbols


def _get_body(node: object) -> object | None:
    """Return class_body or enum_class_body child."""
    for child in _named_children(node):
        if _type(child) in ("class_body", "enum_class_body"):
            return child
    return None


def _get_simple_identifier(node: object, source: bytes) -> str | None:
    """Return text of first simple_identifier named child."""
    for child in _named_children(node):
        if _type(child) == "simple_identifier":
            return _text(child, source)
    return None


def _extract_body_members(
    body: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract all member symbols from a class/struct/extension body."""
    symbols: list[Symbol] = []

    for child in _named_children(body):
        ct = _type(child)

        if ct == "function_declaration":
            sym = _extract_function(child, source, file_path, parent_name)
            if sym:
                symbols.append(sym)
        elif ct == "property_declaration":
            syms = _extract_property(child, source, file_path, parent_name)
            symbols.extend(syms)
        elif ct == "subscript_declaration":
            sym = _extract_subscript(child, source, file_path, parent_name)
            if sym:
                symbols.append(sym)
        elif ct == "class_declaration":
            symbols.extend(_extract_class_declaration(child, source, file_path, parent_name))
        elif ct == "protocol_declaration":
            symbols.extend(_extract_protocol_symbols(child, source, file_path, parent_name))
        elif ct == "typealias_declaration":
            sym = _extract_typealias(child, source, file_path, parent_name)
            if sym:
                symbols.append(sym)

    return symbols


def _extract_function(
    node: object, source: bytes, file_path: str, parent_name: str | None
) -> Symbol | None:
    """Extract a function_declaration as METHOD (if has parent) or FUNCTION."""
    name = _get_simple_identifier(node, source)
    if name is None:
        return None
    vis = _extract_visibility(node)
    sig = _build_method_signature(node, source, name)

    if parent_name is not None:
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


def _extract_property(
    node: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract a property_declaration (var/let) as VARIABLE."""
    vis = _extract_visibility(node)
    pname = _get_property_name(node, source)
    if pname is None:
        return []
    return [
        Symbol(
            name=pname,
            qualified_name=f"{parent_name}.{pname}",
            kind=SymbolKind.VARIABLE,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]


def _get_property_name(node: object, source: bytes) -> str | None:
    """Extract property name from property_declaration.

    Structure: modifiers? value_binding_pattern pattern type_annotation? ...
    pattern > simple_identifier
    """
    for child in _named_children(node):
        if _type(child) == "pattern":
            for gc in _named_children(child):
                if _type(gc) == "simple_identifier":
                    return _text(gc, source)
            # pattern text itself may be the name
            return _text(child, source)
    return None


def _extract_subscript(
    node: object, source: bytes, file_path: str, parent_name: str
) -> Symbol | None:
    """Extract a subscript_declaration as METHOD named 'subscript'."""
    vis = _extract_visibility(node)
    sig = _build_method_signature(node, source, "subscript")
    return Symbol(
        name="subscript",
        qualified_name=f"{parent_name}.subscript",
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
        parent=parent_name,
    )


def _extract_typealias(
    node: object, source: bytes, file_path: str, parent_name: str | None = None
) -> Symbol | None:
    """Extract a typealias_declaration."""
    name = _get_name(node, source)
    if name is None:
        return None
    qualified = f"{parent_name}.{name}" if parent_name else name
    vis = _extract_visibility(node)
    return Symbol(
        name=name,
        qualified_name=qualified,
        kind=SymbolKind.TYPE,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        parent=parent_name,
    )


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------

_EXTERNAL_FRAMEWORKS = frozenset(
    {
        "Foundation",
        "UIKit",
        "AppKit",
        "SwiftUI",
        "Combine",
        "CoreData",
        "CoreFoundation",
        "CoreGraphics",
        "CoreLocation",
        "CoreMotion",
        "ARKit",
        "AVFoundation",
        "CFNetwork",
        "CloudKit",
        "EventKit",
        "GameKit",
        "HealthKit",
        "HomeKit",
        "MapKit",
        "MessageUI",
        "MultipeerConnectivity",
        "NotificationCenter",
        "PassKit",
        "Photos",
        "QuartzCore",
        "SafariServices",
        "SceneKit",
        "Security",
        "SpriteKit",
        "StoreKit",
        "UserNotifications",
        "WatchKit",
        "WebKit",
        "XCTest",
        "Swift",
        "Darwin",
        "Dispatch",
        "ObjectiveC",
        "os",
        "simd",
    }
)


def _parse_swift_import(node: object, source: bytes, file_path: str) -> ImportStatement | None:
    """Parse a single import_declaration node."""
    line = _start_line(node)
    module: str | None = None

    for child in _named_children(node):
        ct = _type(child)
        if ct == "identifier":
            module = _text(child, source)
            break
        if ct == "simple_identifier":
            module = _text(child, source)
            break

    if not module:
        return None

    return ImportStatement(
        module=module,
        symbols=[],
        file_path=file_path,
        line=line,
        is_relative=False,
    )


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _resolve_swift_import(module: str, file_map: dict[str, str]) -> str | None:
    """Resolve a Swift module import to a local file, or None if external."""
    if module in _EXTERNAL_FRAMEWORKS:
        return None

    # Swift module names typically correspond to directory names
    # Look for any .swift file in a directory matching the module name
    module_lower = module.lower()
    for path, abs_path in file_map.items():
        parts = path.replace("\\", "/").split("/")
        # Check if any directory component matches the module name
        for part in parts[:-1]:
            if part.lower() == module_lower:
                return abs_path
        # Also check if the file stem matches (single-file modules)
        stem = parts[-1]
        if stem.endswith(".swift") and stem[:-6].lower() == module_lower:
            return abs_path

    return None


# ---------------------------------------------------------------------------
# SwiftAdapter
# ---------------------------------------------------------------------------


class SwiftAdapter:
    """Language adapter for Swift source files."""

    @property
    def language_id(self) -> str:
        return "swift"

    @property
    def file_extensions(self) -> list[str]:
        return [".swift"]

    @property
    def tree_sitter_name(self) -> str:
        return "swift"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a Swift parse tree."""
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []

        for child in _named_children(root):
            ct = _type(child)
            if ct == "class_declaration":
                symbols.extend(_extract_class_declaration(child, source, file_path))
            elif ct == "protocol_declaration":
                symbols.extend(_extract_protocol_symbols(child, source, file_path))
            elif ct == "function_declaration":
                sym = _extract_function(child, source, file_path, None)
                if sym:
                    symbols.append(sym)
            elif ct == "typealias_declaration":
                sym = _extract_typealias(child, source, file_path)
                if sym:
                    symbols.append(sym)

        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import declarations from a Swift parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _named_children(root):
            if _type(child) == "import_declaration":
                imp = _parse_swift_import(child, source, file_path)
                if imp:
                    imports.append(imp)

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Swift module import to a local file, or None if external."""
        return _resolve_swift_import(imp.module, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect Swift entry points: @main, @UIApplicationMain, @NSApplicationMain, XCTestCase."""
        entry_points: list[str] = []
        entry_markers = (
            "@main",
            "@UIApplicationMain",
            "@NSApplicationMain",
        )

        for f in files:
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if any(marker in content for marker in entry_markers) or ": XCTestCase" in content:
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
