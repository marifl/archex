# Enhancement 7: Expanded Language Support — Implementation Plan

> Branch: `feat/expanded-language-support`
> Target: v0.5.4 (Java, Kotlin, C#, Swift), v0.5.5 (Dart, Zig, Elixir), v0.5.6 (C, C++)

---

## 1. Architecture Overview

### Adapter Contract

Each adapter implements the `LanguageAdapter` protocol (`src/archex/parse/adapters/base.py`):

```python
class LanguageAdapter(Protocol):
    @property
    def language_id(self) -> str: ...
    @property
    def file_extensions(self) -> list[str]: ...
    @property
    def tree_sitter_name(self) -> str: ...

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]: ...
    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]: ...
    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None: ...
    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]: ...
    def classify_visibility(self, symbol: Symbol) -> Visibility: ...
```

### Key Models

- `Symbol`: name, qualified_name, kind (SymbolKind), file_path, start_line, end_line, visibility, signature, docstring, decorators, parent
- `SymbolKind`: FUNCTION, CLASS, METHOD, TYPE, VARIABLE, CONSTANT, INTERFACE, ENUM, MODULE
- `Visibility`: PUBLIC, INTERNAL, PRIVATE
- `ImportStatement`: module, symbols, alias, file_path, line, is_relative, resolved_path
- `DiscoveredFile`: path, absolute_path, language, size_bytes

### Registration

1. Adapters register in `src/archex/parse/adapters/__init__.py` via `default_adapter_registry.register("language", AdapterClass)`
2. External adapters via entry point group `archex.language_adapters`
3. File extension mapping in `src/archex/acquire/discovery.py` → `EXTENSION_MAP`

### Shared Helpers

`ts_node.py` provides: `ts_text`, `ts_type`, `ts_children`, `ts_named_children`, `ts_field`, `ts_start_line`, `ts_end_line`

### Integration Points (language-agnostic)

- **Chunker** (`index/chunker.py`): Uses adapter output (symbols) to chunk files. `make_symbol_id()` generates stable IDs from `file_path + qualified_name + kind`.
- **Graph** (`index/graph.py`): Builds dependency edges from `ImportStatement.resolved_path`. Language-agnostic.
- **Delta** (`index/delta.py`): Operates on file-level diffs. No language-specific hooks needed.
- **Patterns** (`analyze/patterns.py`): Works on abstract graph structure. Existing detectors work across all languages.
- **Store** (`index/store.py`): Language stored as metadata on chunks. No schema changes needed.

---

## 2. Shared Infrastructure Changes

### 2.1 File Extension Mapping (`src/archex/acquire/discovery.py`)

Add to `EXTENSION_MAP`:

| Release | Extensions |
|---------|-----------|
| v0.5.4 | `.java` → java, `.kt` / `.kts` → kotlin, `.cs` → csharp, `.swift` → swift |
| v0.5.5 | `.dart` → dart, `.zig` → zig, `.ex` / `.exs` → elixir |
| v0.5.6 | `.c` / `.h` → c, `.cpp` / `.cxx` / `.cc` / `.hpp` / `.hxx` / `.hh` → cpp |

### 2.2 Default Ignores (`src/archex/acquire/discovery.py`)

Add to `DEFAULT_IGNORES`:

```
"target/",        # Java/Kotlin (Maven/Gradle)
"bin/",            # C#/.NET
"obj/",            # C#/.NET
".build/",         # Swift (SwiftPM)
".dart_tool/",     # Dart
"zig-cache/",      # Zig
"zig-out/",        # Zig
"_build/",         # Elixir (Mix)
"deps/",           # Elixir (Mix)
```

### 2.3 JVM Shared Helpers (`src/archex/parse/adapters/_jvm_helpers.py`) — NEW

Shared between Java and Kotlin adapters:
- `resolve_jvm_package(package_name, file_map)` — maps `com.example.Foo` to `src/main/java/com/example/Foo.java`
- `detect_jvm_convention(file_map)` — detects Maven (`src/main/java`) vs Gradle (`app/src/main/java`) vs flat layout
- `map_jvm_visibility(modifier)` — maps JVM visibility keywords to `Visibility` enum
- `build_package_from_path(file_path, convention)` — reverse: infer package from file path

### 2.4 Dependencies (`pyproject.toml`)

v0.5.4 additions:
```
"tree-sitter-java>=0.23",
"tree-sitter-kotlin>=1.1",
"tree-sitter-c-sharp>=0.23",
```
Swift: use `tree-sitter-language-pack` fallback (standalone `tree-sitter-swift` 0.0.1 is unreliable).

v0.5.5 additions:
```
"tree-sitter-zig>=1.1",
```
Plus `tree-sitter-language-pack>=0.13` as optional dependency (for Dart, Elixir, and Swift fallback).
Test `tree-sitter-elixir>=0.3.4` standalone first — if compatible with tree-sitter>=0.23, add directly.

v0.5.6 additions:
```
"tree-sitter-c>=0.23",
"tree-sitter-cpp>=0.23",
```

> **Grammar Availability (verified March 2026):**
>
> | Package | PyPI Version | Status |
> |---------|-------------|--------|
> | `tree-sitter-java` | 0.23.5 | Compatible |
> | `tree-sitter-kotlin` | 1.1.0 | Compatible |
> | `tree-sitter-c-sharp` | 0.23.1 | Compatible |
> | `tree-sitter-c` | 0.24.1 | Compatible |
> | `tree-sitter-cpp` | 0.23.4 | Compatible |
> | `tree-sitter-zig` | 1.1.2 | Compatible |
> | `tree-sitter-swift` | 0.0.1 | **RISK** — early release, ARM Linux only, compat issues with tree-sitter 0.23+ |
> | `tree-sitter-elixir` | 0.3.4 | **UNCERTAIN** — pre-0.23 convention, needs testing |
> | `tree-sitter-dart` | N/A | **NOT ON PyPI** — grammar exists on GitHub but no standalone package |
>
> **Mitigation for Swift/Dart/Elixir:** `tree-sitter-language-pack>=0.13` bundles 165+ grammars including all three. Uses different API (`get_language("swift")` vs `Language(tree_sitter_swift.language())`). Engine needs a fallback loader path if standalone packages unavailable. See §2.5 for loader strategy.

### 2.5 Tree-Sitter Engine (`src/archex/parse/engine.py`)

`_LANGUAGE_LOADERS` maps language IDs to `(module_name, function_name)` tuples for lazy grammar loading. **Must add an entry per new language** — this is separate from adapter registration.

**Language ID convention:** The `_LANGUAGE_LOADERS` key, `EXTENSION_MAP` value, and adapter `language_id` property must all use the same string. For C#, use `"csharp"` (not `"c_sharp"`) — the loader value points to the actual module name `tree_sitter_c_sharp`.

v0.5.4 additions:
```python
"java": ("tree_sitter_java", "language"),
"kotlin": ("tree_sitter_kotlin", "language"),
"csharp": ("tree_sitter_c_sharp", "language"),
"swift": ("tree_sitter_swift", "language"),       # ⚠ see fallback below
```

v0.5.5 additions:
```python
"dart": ("tree_sitter_dart", "language"),          # ⚠ no standalone PyPI pkg
"zig": ("tree_sitter_zig", "language"),
"elixir": ("tree_sitter_elixir", "language"),      # ⚠ needs compat testing
```

v0.5.6 additions:
```python
"c": ("tree_sitter_c", "language"),
"cpp": ("tree_sitter_cpp", "language"),
```

**Fallback loader for Swift/Dart/Elixir:**

The engine's `get_language()` currently assumes standalone grammar packages. For languages where standalone packages are unavailable or broken, add a fallback path that tries `tree-sitter-language-pack`:

```python
# In get_language(), after the primary ImportError:
try:
    from tree_sitter_language_pack import get_language as _pack_get_language
    lang = _pack_get_language(language_id)
except ImportError:
    raise ParseError(f"tree-sitter grammar for {language_id!r} not installed")
```

This requires adding `tree-sitter-language-pack>=0.13` as an optional dependency:
```toml
[project.optional-dependencies]
language-pack = ["tree-sitter-language-pack>=0.13"]
```

**Decision needed:** Whether to use `tree-sitter-language-pack` for all grammars (uniform) or only as fallback for Swift/Dart/Elixir (minimal dependency). Recommendation: fallback-only — keeps the standalone-first approach for languages with good PyPI packages.

### 2.6 Adapter Registration (`src/archex/parse/adapters/__init__.py`)

Add imports and registrations for each new adapter per release. Pattern:

```python
from archex.parse.adapters.java import JavaAdapter
default_adapter_registry.register("java", JavaAdapter)
```

---

## 3. v0.5.4 — JVM + .NET + Apple (Java, Kotlin, C#, Swift)

### 3.1 Java Adapter (`src/archex/parse/adapters/java.py`)

**Properties:**
- `language_id`: `"java"`
- `file_extensions`: `[".java"]`
- `tree_sitter_name`: `"java"`

**extract_symbols:**
- Classes, interfaces, enums, records → `SymbolKind.CLASS` / `INTERFACE` / `ENUM` / `TYPE`
- Methods → `SymbolKind.METHOD` with parent set to enclosing class
- Fields, constants (`static final`) → `SymbolKind.VARIABLE` / `CONSTANT`
- Annotations (`@interface`) → `SymbolKind.TYPE`
- Inner classes: `qualified_name` = `OuterClass.InnerClass`
- Constructors: treat as `SymbolKind.METHOD` with name matching class name

**parse_imports:**
- `import pkg.Class` → `ImportStatement(module="pkg.Class")`
- `import pkg.*` → `ImportStatement(module="pkg.*")`
- `import static pkg.Class.method` → `ImportStatement(module="pkg.Class.method", symbols=["method"])`

**resolve_import:**
- Use `_jvm_helpers.resolve_jvm_package()` to map `com.example.Foo` → search for `Foo.java` in matching directory structure
- Handle Maven/Gradle `src/main/java` convention

**detect_entry_points:**
- Scan for `public static void main(String` pattern
- Detect `@SpringBootApplication` annotation
- Detect `@Test` (JUnit) in test files

**classify_visibility:**
- Parse access modifier from source: `public` → PUBLIC, `protected` → INTERNAL, `private` → PRIVATE
- No modifier (package-private) → INTERNAL

**Estimated size:** ~400 lines

### 3.2 Kotlin Adapter (`src/archex/parse/adapters/kotlin.py`)

**Properties:**
- `language_id`: `"kotlin"`
- `file_extensions`: `[".kt", ".kts"]`
- `tree_sitter_name`: `"kotlin"`

**extract_symbols:**
- Classes, objects, data classes, sealed classes/interfaces → CLASS / TYPE
- Functions (top-level and member) → FUNCTION / METHOD
- Properties (`val`/`var`) → VARIABLE
- Extension functions: `qualified_name` = `ReceiverType.methodName`, parent = receiver type
- Companion objects: `qualified_name` = `ClassName.Companion`
- Typealiases → TYPE
- Sealed hierarchies: parent set to sealed base

**parse_imports:**
- `import pkg.Class` → standard
- `import pkg.*` → wildcard
- `import pkg.func as alias` → with alias

**resolve_import:**
- Use shared `_jvm_helpers` — same directory convention as Java
- Cross-language: Kotlin files in same package resolve to Java files and vice versa

**detect_entry_points:**
- `fun main()` or `fun main(args: Array<String>)` at top level
- `@SpringBootApplication`, `@Test`, `@Composable`

**classify_visibility:**
- `public` (default, no modifier) → PUBLIC
- `internal` → INTERNAL
- `protected` → INTERNAL
- `private` → PRIVATE

**Estimated size:** ~350 lines

### 3.3 C# Adapter (`src/archex/parse/adapters/csharp.py`)

**Properties:**
- `language_id`: `"csharp"`
- `file_extensions`: `[".cs"]`
- `tree_sitter_name`: `"c_sharp"`

**extract_symbols:**
- Classes, structs, interfaces, enums, records → CLASS / TYPE / INTERFACE / ENUM
- Methods → METHOD with parent
- Properties → VARIABLE with parent
- Fields → VARIABLE / CONSTANT
- Events, delegates → TYPE
- `qualified_name` = `Namespace.Class.Method` (extract namespace from tree)

**parse_imports:**
- `using Namespace` → `ImportStatement(module="Namespace")`
- `using static Namespace.Class` → with symbols
- `global using` → same as using but flagged

**resolve_import:**
- Map `using Namespace` → search `.cs` files with matching `namespace` declaration in source
- `.csproj` references not needed for MVP (internal resolution sufficient)

**detect_entry_points:**
- `static void Main` / `static async Task Main`
- Top-level statements (C# 9+): files without class wrapper
- `[Fact]` / `[Test]` / `[TestMethod]` annotations

**classify_visibility:**
- `public` → PUBLIC
- `internal` → INTERNAL
- `protected` → INTERNAL
- `private` → PRIVATE
- `protected internal` → INTERNAL
- `private protected` → PRIVATE

**Estimated size:** ~400 lines

### 3.4 Swift Adapter (`src/archex/parse/adapters/swift.py`)

**Properties:**
- `language_id`: `"swift"`
- `file_extensions`: `[".swift"]`
- `tree_sitter_name`: `"swift"`

**extract_symbols:**
- Classes, structs, enums, protocols → CLASS / TYPE / ENUM / INTERFACE
- Functions → FUNCTION / METHOD
- Properties → VARIABLE
- Extensions: `qualified_name` = `ExtendedType.method`, parent = extended type
- Actors → CLASS
- Subscripts → METHOD

**parse_imports:**
- `import Module` → `ImportStatement(module="Module")`
- `@testable import Module` → with decorator info

**resolve_import:**
- Module-level imports — map internal modules via directory structure
- `Package.swift` parsing for target membership (best-effort)
- External frameworks (Foundation, UIKit) → None (external)

**detect_entry_points:**
- `@main` struct/class
- `@UIApplicationMain` / `@NSApplicationMain`
- `XCTestCase` subclasses

**classify_visibility:**
- `public` / `open` → PUBLIC
- `internal` (default, no modifier) → INTERNAL
- `fileprivate` / `private` → PRIVATE

**Estimated size:** ~350 lines

### 3.5 Test Fixtures (v0.5.4)

Create 4 fixture directories:

**`tests/fixtures/java_simple/`** (5 files):
- `Main.java` — entry point with `public static void main`
- `models/User.java` — class with fields, methods, constructors, inner class
- `services/UserService.java` — interface + implementation, DI pattern
- `utils/StringUtils.java` — utility class with static methods, constants
- `enums/Status.java` — enum with fields and methods

**`tests/fixtures/kotlin_simple/`** (5 files):
- `Main.kt` — top-level `fun main()`, top-level functions
- `models/User.kt` — data class, sealed class hierarchy
- `services/UserService.kt` — class with companion object, extension functions
- `utils/Extensions.kt` — extension functions, typealiases
- `config/AppConfig.kt` — object declaration, properties

**`tests/fixtures/csharp_simple/`** (5 files):
- `Program.cs` — entry point with `Main` or top-level statements
- `Models/User.cs` — class, record, struct, enum
- `Services/UserService.cs` — interface + implementation, properties
- `Utils/StringExtensions.cs` — static class, extension methods
- `Events/EventHandler.cs` — events, delegates

**`tests/fixtures/swift_simple/`** (5 files):
- `main.swift` — `@main` struct, top-level functions
- `Models/User.swift` — struct, enum, protocol
- `Services/UserService.swift` — class, extension, actor
- `Utils/Extensions.swift` — protocol extensions, typealiases
- `Views/ContentView.swift` — SwiftUI struct with property wrappers

### 3.6 Test Suites (v0.5.4)

Each adapter gets `tests/test_parse/test_<language>.py` with:

1. **Symbol extraction** (~10 tests per adapter):
   - All symbol kinds present in language
   - Inner/nested types
   - Qualified name correctness
   - Parent chain correctness
   - Signature extraction

2. **Import parsing** (~5 tests per adapter):
   - All import styles (standard, wildcard, static, aliased)
   - Line number accuracy

3. **Import resolution** (~5 tests per adapter):
   - Internal imports resolve to correct files
   - External imports return None
   - Convention detection (Maven/Gradle for JVM)

4. **Visibility** (~5 tests per adapter):
   - All access modifiers mapped correctly
   - Default visibility (no modifier) handled

5. **Entry points** (~3 tests per adapter):
   - Language-specific entry patterns detected
   - Non-entry files excluded

6. **Round-trip** (~5 tests per adapter):
   - parse → chunk → store → retrieve produces correct content
   - Stable symbol IDs generated correctly
   - Delta indexing with new-language files

7. **Cross-language** (Java + Kotlin only, ~5 tests):
   - Mixed-language project resolves imports correctly
   - Kotlin importing Java class and vice versa

**Total: ~200 new tests**

---

## 4. v0.5.5 — Mobile + BEAM + Next-Gen Systems (Dart, Zig, Elixir)

### 4.1 Dart Adapter (`src/archex/parse/adapters/dart.py`)

**Properties:**
- `language_id`: `"dart"`
- `file_extensions`: `[".dart"]`
- `tree_sitter_name`: `"dart"`

**extract_symbols:**
- Classes, mixins, extensions, enums → CLASS / TYPE / ENUM
- Functions, methods → FUNCTION / METHOD
- Properties (getters/setters) → VARIABLE
- Typedefs → TYPE
- Factory constructors → METHOD

**parse_imports:**
- `import 'package:...'` → package import
- `import '...' as prefix` → aliased
- `import '...' show/hide` → selective
- `part` / `part of` → treat as separate files (simpler approach)
- `export` → re-export

**resolve_import:**
- `package:` scheme → map via directory convention (`lib/`)
- Relative imports → direct file path mapping

**detect_entry_points:**
- `void main()` at top level
- `runApp()` call (Flutter)

**classify_visibility:**
- `_` prefix → PRIVATE
- Everything else → PUBLIC (Dart has no protected/internal)

**Estimated size:** ~300 lines

### 4.2 Zig Adapter (`src/archex/parse/adapters/zig.py`)

**Properties:**
- `language_id`: `"zig"`
- `file_extensions`: `[".zig"]`
- `tree_sitter_name`: `"zig"`

**extract_symbols:**
- Functions (`fn`) → FUNCTION
- Structs → TYPE (Zig structs serve as classes)
- Enums → ENUM
- Unions → TYPE
- Error sets → TYPE
- Constants (`const`) → CONSTANT
- Variables (`var`) → VARIABLE
- Test blocks (`test "..."`) → FUNCTION with name = test description
- Struct member functions → METHOD with parent

**parse_imports:**
- `@import("...")` → ImportStatement
- `const std = @import("std")` → with alias

**resolve_import:**
- `@import` of relative path → resolve relative to source file
- `@import("std")` → None (external)
- `build.zig` for package paths (best-effort)

**detect_entry_points:**
- `pub fn main()` at top level
- `export fn` (C ABI export)
- `test "..."` blocks

**classify_visibility:**
- `pub` keyword → PUBLIC
- No `pub` → PRIVATE

**Estimated size:** ~250 lines (simplest adapter — no inheritance, no exceptions, no macros)

### 4.3 Elixir Adapter (`src/archex/parse/adapters/elixir.py`)

**Properties:**
- `language_id`: `"elixir"`
- `file_extensions`: `[".ex", ".exs"]`
- `tree_sitter_name`: `"elixir"`

**extract_symbols:**
- Modules (`defmodule`) → MODULE
- Functions (`def`) → FUNCTION (public)
- Private functions (`defp`) → FUNCTION (private visibility)
- Macros (`defmacro`) → FUNCTION
- Structs (`defstruct`) → TYPE
- Protocols (`defprotocol`) → INTERFACE
- Protocol implementations (`defimpl`) → CLASS
- Callbacks (`@callback`) → FUNCTION

**parse_imports:**
- `alias Module.Name` → aliased import
- `import Module` → wildcard import
- `use Module` → macro-based import
- `require Module` → compile-time import

**resolve_import:**
- Module names → file paths via Elixir convention: `MyApp.Accounts.User` → `lib/my_app/accounts/user.ex`
- `mix.exs` deps → None (external)

**detect_entry_points:**
- `Application.start/2` callback
- `Plug.Router` / `Phoenix.Router` modules
- `GenServer.init/1` callback

**classify_visibility:**
- `def` → PUBLIC
- `defp` → PRIVATE
- Module-level only (no protected/internal in Elixir)

**Estimated size:** ~300 lines

### 4.4 Test Fixtures (v0.5.5)

**`tests/fixtures/dart_simple/`** (5 files):
- `main.dart` — entry point with `void main()`, `runApp()`
- `models/user.dart` — class, mixin, enum, factory constructor
- `services/api_service.dart` — class with async methods
- `widgets/home_page.dart` — StatefulWidget, State class
- `utils/extensions.dart` — extension methods, typedefs

**`tests/fixtures/zig_simple/`** (5 files):
- `main.zig` — `pub fn main()`, imports, constants
- `types.zig` — structs, enums, unions, error sets
- `utils.zig` — helper functions, pub/private
- `tests.zig` — test blocks
- `build.zig` — build configuration (not parsed, but present for convention)

**`tests/fixtures/elixir_simple/`** (5 files):
- `lib/app.ex` — Application module with start/2
- `lib/accounts/user.ex` — module with struct, functions, defp
- `lib/services/user_service.ex` — GenServer module
- `lib/web/router.ex` — Phoenix-style router
- `test/accounts/user_test.exs` — ExUnit test module

### 4.5 Test Suites (v0.5.5)

Same categories as v0.5.4 per adapter. **~150 new tests total** (~50 per adapter).

---

## 5. v0.5.6 — C/C++ (Hard Mode)

### 5.1 C Family Shared Helpers (`src/archex/parse/adapters/_c_family_helpers.py`) — NEW

Shared between C and C++ adapters:
- `parse_include(node, source)` — extract `#include "..."` / `#include <...>` paths
- `resolve_include(path, include_dirs, file_map)` — resolve include to file path
- `detect_include_dirs(file_map)` — heuristic detection from common layouts
- `load_compile_commands(project_root)` — parse `compile_commands.json` for include paths
- `classify_c_visibility(symbol, source)` — `static` = PRIVATE, else PUBLIC (file-level)

### 5.2 C Adapter (`src/archex/parse/adapters/c_lang.py`)

**Properties:**
- `language_id`: `"c"`
- `file_extensions`: `[".c", ".h"]`
- `tree_sitter_name`: `"c"`

**extract_symbols:**
- Functions → FUNCTION
- Structs, unions, enums → TYPE / ENUM
- Typedefs → TYPE
- Global variables → VARIABLE
- Macro definitions (`#define`) → CONSTANT (name only, no body analysis)
- Function declarations in headers → FUNCTION (forward declaration)

**parse_imports:**
- `#include "path.h"` → ImportStatement with is_relative=True
- `#include <sys/types.h>` → ImportStatement with is_relative=False

**resolve_import:**
- Quoted includes: resolve relative to source file, then search include dirs
- Angle-bracket includes: search include dirs only
- `compile_commands.json` for build-system include paths

**classify_visibility:**
- `static` functions/variables → PRIVATE (file-scoped)
- Everything else → PUBLIC

**Estimated size:** ~600 lines

### 5.3 C++ Adapter (`src/archex/parse/adapters/cpp.py`)

**Properties:**
- `language_id`: `"cpp"`
- `file_extensions`: `[".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"]`
- `tree_sitter_name`: `"cpp"`

**extract_symbols:**
- Everything from C adapter, plus:
- Classes → CLASS
- Namespaces → MODULE (qualified names use `::`)
- Templates → TYPE with signature including template params
- Methods (virtual, override, const) → METHOD
- Constructors/destructors → METHOD
- Operators → METHOD

**Additional complexity:**
- Namespace nesting: `qualified_name` uses `::` separator
- Anonymous namespaces → PRIVATE visibility
- Template parameter extraction (signature-level only, no instantiation analysis)

**Estimated size:** ~600 lines

### 5.4 Known Limitations (documented in README)

- Preprocessor macros: `#define` captured as constants, conditional compilation blocks not analyzed
- Header resolution: requires `compile_commands.json` for accuracy; heuristic fallback may miss paths
- Templates: signature-level only — template instantiation and metaprogramming not tracked
- C++20 modules: not supported (too early in adoption)

---

## 6. Implementation Order & Dependencies

### Phase A: Shared Infrastructure (all releases)

```
1. EXTENSION_MAP updates         → discovery.py           (per release)
2. DEFAULT_IGNORES additions     → discovery.py           (per release)
3. pyproject.toml dependencies   → pyproject.toml         (per release)
4. _LANGUAGE_LOADERS entries     → parse/engine.py        (per release)
5. Adapter registration          → adapters/__init__.py   (per release)
```

### Phase B: v0.5.4 Implementation Order

```
1. _jvm_helpers.py              → shared JVM module (Java + Kotlin prereq)
2. Java adapter + tests         → java.py, test_java.py, fixtures/java_simple/
3. Kotlin adapter + tests       → kotlin.py, test_kotlin.py, fixtures/kotlin_simple/
4. Java ↔ Kotlin cross-lang     → cross-language resolution tests
5. C# adapter + tests           → csharp.py, test_csharp.py, fixtures/csharp_simple/
6. Swift adapter + tests        → swift.py, test_swift.py, fixtures/swift_simple/
7. Integration tests            → test_integration.py additions
```

### Phase C: v0.5.5 Implementation Order

```
1. Dart adapter + tests         → dart.py, test_dart.py, fixtures/dart_simple/
2. Zig adapter + tests          → zig.py, test_zig.py, fixtures/zig_simple/
3. Elixir adapter + tests       → elixir.py, test_elixir.py, fixtures/elixir_simple/
4. Integration tests            → test_integration.py additions
```

### Phase D: v0.5.6 Implementation Order

```
1. _c_family_helpers.py         → shared C/C++ module
2. C adapter + tests            → c_lang.py, test_c.py, fixtures/c_simple/
3. C++ adapter + tests          → cpp.py, test_cpp.py, fixtures/cpp_simple/
4. compile_commands.json support → include resolution
5. Integration tests            → test_integration.py additions
```

---

## 7. File Manifest

### New Files

| File | Release | Est. Lines |
|------|---------|-----------|
| `src/archex/parse/adapters/_jvm_helpers.py` | v0.5.4 | ~150 |
| `src/archex/parse/adapters/java.py` | v0.5.4 | ~400 |
| `src/archex/parse/adapters/kotlin.py` | v0.5.4 | ~350 |
| `src/archex/parse/adapters/csharp.py` | v0.5.4 | ~400 |
| `src/archex/parse/adapters/swift.py` | v0.5.4 | ~350 |
| `src/archex/parse/adapters/dart.py` | v0.5.5 | ~300 |
| `src/archex/parse/adapters/zig.py` | v0.5.5 | ~250 |
| `src/archex/parse/adapters/elixir.py` | v0.5.5 | ~300 |
| `src/archex/parse/adapters/_c_family_helpers.py` | v0.5.6 | ~200 |
| `src/archex/parse/adapters/c_lang.py` | v0.5.6 | ~600 |
| `src/archex/parse/adapters/cpp.py` | v0.5.6 | ~600 |
| `tests/parse/adapters/test_java.py` | v0.5.4 | ~300 |
| `tests/parse/adapters/test_kotlin.py` | v0.5.4 | ~300 |
| `tests/parse/adapters/test_csharp.py` | v0.5.4 | ~300 |
| `tests/parse/adapters/test_swift.py` | v0.5.4 | ~300 |
| `tests/parse/adapters/test_jvm_cross.py` | v0.5.4 | ~100 |
| `tests/parse/adapters/test_dart.py` | v0.5.5 | ~250 |
| `tests/parse/adapters/test_zig.py` | v0.5.5 | ~250 |
| `tests/parse/adapters/test_elixir.py` | v0.5.5 | ~250 |
| `tests/parse/adapters/test_c.py` | v0.5.6 | ~300 |
| `tests/parse/adapters/test_cpp.py` | v0.5.6 | ~300 |
| Test fixtures (9 languages × 5 files) | All | ~45 files |

### Modified Files

| File | Change |
|------|--------|
| `src/archex/acquire/discovery.py` | Add EXTENSION_MAP entries + DEFAULT_IGNORES |
| `src/archex/parse/adapters/__init__.py` | Import + register new adapters |
| `src/archex/parse/engine.py` | Add entries to `_LANGUAGE_LOADERS` dict for each new language |
| `pyproject.toml` | Add tree-sitter grammar dependencies |
| `tests/test_integration.py` | Add integration tests for new languages |

---

## 8. Quality Gates

### Per Adapter

- [ ] All SymbolKind variants used correctly for the language
- [ ] qualified_name populated for all symbols (needed for stable IDs)
- [ ] parent field set for nested/member symbols
- [ ] Visibility mapping covers all access modifiers + default
- [ ] Import parsing handles all syntax variants
- [ ] Internal imports resolve; external imports return None
- [ ] Entry point detection finds language-specific patterns
- [ ] `uv run ruff check .` clean
- [ ] `uv run pyright` clean (strict mode)
- [ ] `uv run pytest` all passing

### Per Release

- [ ] All new tests pass
- [ ] Existing tests unaffected (no regressions)
- [ ] Coverage ≥ 88% (v0.5.4), ≥ 87% (v0.5.5), ≥ 86% (v0.5.6)
- [ ] Round-trip test: parse → chunk → store → retrieve for each language
- [ ] Delta indexing works for new-language files

---

## 9. Open Decisions

| Decision | Options | Recommendation |
|----------|---------|---------------|
| Java/Kotlin Spring patterns (v0.5.4) | Ship in v0.5.4 vs defer | Defer — ship core adapter first, add patterns as follow-up |
| Kotlin coroutine depth | Signature-only vs Flow graph | Signature-only (`suspend` detected as decorator) |
| Swift result builders | Detect as annotated symbols vs deep analysis | Detect as symbols with annotations |
| Dart `part`/`part of` | Merge into one ParsedFile vs separate | Keep separate (simpler, consistent with how files are discovered) |
| C/C++ `compile_commands.json` | Require vs heuristic vs both | Both with fallback — heuristic default, compile_commands when available |
| C++ template analysis | Signature-only vs instantiation | Signature-only (safe, complete) |
| Language-specific patterns entry points | Namespaced vs flat group | Same flat `archex.pattern_detectors` group (simpler) |
| Header file ambiguity (`.h`) | Default to C vs detect | Detect: if `.cpp`/`.cc` files exist in same project → treat `.h` as cpp; otherwise → c |
| Chunker `_format_import()` | Python-style for all langs vs language-aware | Defer — cosmetic issue only, Python-style is functionally harmless for BM25 search |
| Grammar loading strategy | Standalone packages vs `tree-sitter-language-pack` for all | Standalone-first with language-pack fallback for Swift/Dart/Elixir only |
| `tree-sitter-language-pack` as required or optional dep | Required (simpler) vs optional (lighter) | Optional — keeps install light for users who only index Tier 1 languages |

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| tree-sitter grammar incompatibility | Adapter fails to parse | Pin grammar versions; test against grammar before implementation |
| Grammar package not on PyPI (Dart) | Can't add as dependency | Use `tree-sitter-language-pack>=0.13` as optional dep with fallback loader in engine.py |
| Grammar compat issues (Swift, Elixir) | Adapter fails at parse time | Swift: language-pack fallback; Elixir: test standalone first, fall back to language-pack |
| JVM cross-language resolution edge cases | Incorrect import resolution | Start with simple cases, document known limitations |
| C/C++ include resolution accuracy | Missing or wrong dependencies in graph | Document as known limitation; compile_commands.json improves accuracy |
| Test fixture complexity | Incomplete coverage | Start with minimal fixtures, add edge cases as bugs surface |
| Performance with 13 languages | Slower discovery/parsing | Extension map is O(1); adapters only instantiated for detected languages |
