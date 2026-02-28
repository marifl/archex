"""Language adapter registry: maps language identifiers to parser adapter implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.go import GoAdapter
from archex.parse.adapters.python import PythonAdapter
from archex.parse.adapters.rust import RustAdapter
from archex.parse.adapters.typescript import TypeScriptAdapter

if TYPE_CHECKING:
    from archex.parse.engine import TreeSitterEngine

ADAPTERS: dict[str, type[LanguageAdapter]] = {
    "python": PythonAdapter,  # type: ignore[type-abstract]
    "typescript": TypeScriptAdapter,  # type: ignore[type-abstract]
    "go": GoAdapter,  # type: ignore[type-abstract]
    "rust": RustAdapter,  # type: ignore[type-abstract]
}


def get_adapter(language_id: str, engine: TreeSitterEngine) -> LanguageAdapter | None:
    """Return an instantiated LanguageAdapter for language_id, or None if unsupported."""
    adapter_class = ADAPTERS.get(language_id)
    if adapter_class is None:
        return None
    _ = engine  # reserved for future adapters that need engine at construction
    return adapter_class()


__all__ = ["LanguageAdapter", "ADAPTERS", "get_adapter"]
