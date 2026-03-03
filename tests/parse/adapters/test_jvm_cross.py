"""Cross-language resolution tests for Java ↔ Kotlin import resolution."""

from __future__ import annotations

from archex.models import ImportStatement
from archex.parse.adapters._jvm_helpers import resolve_jvm_import
from archex.parse.adapters.java import JavaAdapter


def test_kotlin_resolves_java_class() -> None:
    """Kotlin import of a Java class resolves to the .java file."""
    file_map = {
        "models/User.java": "/repo/models/User.java",
        "models/UserExt.kt": "/repo/models/UserExt.kt",
    }
    result = resolve_jvm_import("com.example.models.User", file_map, extensions=(".kt", ".java"))
    assert result == "/repo/models/User.java"


def test_java_resolves_kotlin_class() -> None:
    """Java import of a Kotlin-defined class resolves to the .kt file."""
    file_map = {
        "utils/Extensions.kt": "/repo/utils/Extensions.kt",
    }
    result = resolve_jvm_import(
        "com.example.utils.Extensions", file_map, extensions=(".java", ".kt")
    )
    assert result == "/repo/utils/Extensions.kt"


def test_mixed_project_prefers_matching_extension() -> None:
    """When both .java and .kt exist for a class name, extension order determines priority."""
    file_map = {
        "models/Config.kt": "/repo/models/Config.kt",
        "models/Config.java": "/repo/models/Config.java",
    }
    # Kotlin adapter searches .kt first
    result_kt = resolve_jvm_import(
        "com.example.models.Config", file_map, extensions=(".kt", ".java")
    )
    assert result_kt == "/repo/models/Config.kt"

    # Java adapter searches .java first
    result_java = resolve_jvm_import(
        "com.example.models.Config", file_map, extensions=(".java", ".kt")
    )
    assert result_java == "/repo/models/Config.java"


def test_wildcard_import_returns_none() -> None:
    """Wildcard imports cannot resolve to a single file."""
    file_map = {"models/User.java": "/repo/models/User.java"}
    result = resolve_jvm_import("com.example.models.*", file_map, extensions=(".java", ".kt"))
    assert result is None


def test_java_adapter_external_import_returns_none() -> None:
    """External Java imports (stdlib) return None even in a mixed project."""
    adapter = JavaAdapter()
    file_map = {
        "models/User.java": "/repo/models/User.java",
        "models/UserExt.kt": "/repo/models/UserExt.kt",
    }
    imp = ImportStatement(
        module="java.util.List",
        file_path="Main.java",
        line=3,
        is_relative=False,
    )
    assert adapter.resolve_import(imp, file_map) is None
