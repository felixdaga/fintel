"""Structural invariants. These fail the build, so the layout can't rot.

Convention alone is what let the previous attempt drift; each check here maps to
a defect that actually shipped.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import fintel

ROOT = Path(fintel.__file__).parent

# docs/architecture.md §5. Imports flow one way only.
LAYERS: dict[str, int] = {
    "models": 0,
    "utils": 1,
    "pit": 2,  # strictly below market: every source clamps through it
    "market": 3,
    "environment": 4,
    "agents": 5,
    "strategy": 6,
    "evaluate": 7,
    "scoring": 8,
    "report": 9,
    "cli": 10,
}

# scoring/ and pit/ read artifacts; they must never reach into orchestration.
FORBIDDEN: set[tuple[str, str]] = {("scoring", "evaluate"), ("pit", "evaluate")}


def _modules() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _package_of(path: Path) -> str:
    rel = path.relative_to(ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _imported_fintel_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[0] == "fintel" and len(parts) > 1:
                out.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "fintel" and len(parts) > 1:
                    out.add(parts[1])
    return out


def test_layer_ladder_is_respected():
    violations = []
    for path in _modules():
        src = _package_of(path)
        if src not in LAYERS:
            continue
        for dst in _imported_fintel_packages(path):
            if dst not in LAYERS or dst == src:
                continue
            if LAYERS[dst] >= LAYERS[src]:
                violations.append(
                    f"{path.relative_to(ROOT)}: {src}(L{LAYERS[src]}) -> {dst}(L{LAYERS[dst]})"
                )
    assert not violations, "layer violations:\n" + "\n".join(violations)


def test_forbidden_edges():
    violations = []
    for path in _modules():
        src = _package_of(path)
        for dst in _imported_fintel_packages(path):
            if (src, dst) in FORBIDDEN:
                violations.append(f"{path.relative_to(ROOT)}: {src} -> {dst}")
    assert not violations, "forbidden edges:\n" + "\n".join(violations)


def test_models_import_no_logic():
    for path in _modules():
        if _package_of(path) != "models":
            continue
        leaked = _imported_fintel_packages(path) - {"models"}
        assert not leaked, f"{path.relative_to(ROOT)} imports logic: {sorted(leaked)}"


def test_no_cross_package_private_imports():
    violations = []
    for path in _modules():
        src = _package_of(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
                continue
            parts = node.module.split(".")
            if parts[0] != "fintel" or len(parts) < 3 or parts[1] == src:
                continue
            if parts[-1].startswith("_"):
                violations.append(f"{path.relative_to(ROOT)} -> {node.module}")
    assert not violations, "cross-package private imports:\n" + "\n".join(violations)


def test_every_module_imports_cleanly():
    failures = []
    for mod in pkgutil.walk_packages(fintel.__path__, "fintel."):
        try:
            importlib.import_module(mod.name)
        except ImportError as exc:  # optional extras are allowed to be absent
            if "No module named" in str(exc) and "fintel" not in str(exc):
                continue
            failures.append(f"{mod.name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "import failures:\n" + "\n".join(failures)


def _imports_symbol(path: Path, module: str, attr: str) -> bool:
    """True if `path` reaches `module.attr`, written either way."""
    tree = ast.parse(path.read_text(), filename=str(path))
    full = f"{module}.{attr}"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == full:
                return True
            if node.module == module and any(a.name == attr for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name == full or a.name.startswith(full + ".") for a in node.names):
                return True
    return False


# `market.realized` is the only module allowed to read past the decision date.
# Anything the agent can reach must not be able to.
AGENT_FACING_PACKAGES = {"environment", "agents", "pit"}


def test_agent_facing_layers_cannot_read_realized_prices():
    violations = [
        str(path.relative_to(ROOT))
        for path in _modules()
        if _package_of(path) in AGENT_FACING_PACKAGES
        and _imports_symbol(path, "fintel.market", "realized")
    ]
    assert not violations, "these import the unclamped scoring price path:\n" + "\n".join(
        violations
    )


def test_the_realized_guard_can_actually_detect_a_violation(tmp_path):
    """A guard nobody has seen fail is a guard nobody knows works."""
    offender = tmp_path / "offender.py"
    offender.write_text("from fintel.market.realized import PriceLookup\n")
    assert _imports_symbol(offender, "fintel.market", "realized")
    offender.write_text("from fintel.market import realized\n")
    assert _imports_symbol(offender, "fintel.market", "realized")
    offender.write_text("from fintel.market.data import store\n")
    assert not _imports_symbol(offender, "fintel.market", "realized")


def test_no_legacy_imports():
    """Nothing outside _legacy/ may import it. The staging dir must stay a
    one-way harvest, and its emptying is the progress meter."""
    violations = []
    for path in _modules():
        if "_legacy" in path.parts:
            continue
        if "_legacy" in _imported_fintel_packages(path):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, "imports from _legacy:\n" + "\n".join(violations)


def _registries(mod) -> dict[str, dict[str, str]]:
    """Upper-case `{name: "module:Attr"}` maps — a factory may keep several
    (universes, schedules, sources) rather than one blessed BUILTINS."""
    out = {}
    for attr in dir(mod):
        if not attr.isupper():
            continue
        value = getattr(mod, attr)
        if (
            isinstance(value, dict)
            and value
            and all(
                isinstance(k, str) and isinstance(v, str) and ":" in v for k, v in value.items()
            )
        ):
            out[attr] = value
    return out


@pytest.mark.parametrize("package", sorted(LAYERS))
def test_factory_registries_resolve(package: str):
    """Every `module:Attr` string a factory advertises must actually import.
    This is the check that would have caught the dangling `delorean.config`."""
    try:
        mod = importlib.import_module(f"fintel.{package}.factory")
    except ModuleNotFoundError:
        pytest.skip(f"fintel.{package} has no factory")
    from fintel.utils.import_path import resolve

    registries = _registries(mod)
    assert registries, f"fintel.{package}.factory advertises no name registry"
    for name, registry in registries.items():
        for key, target in registry.items():
            assert resolve(target) is not None, f"{package}.{name}[{key}] -> {target}"


def test_market_registers_its_schedule_builtins():
    """Guard the guard: if a registry stops being discoverable the check above
    silently degrades to a skip."""
    from fintel.market import factory

    assert set(_registries(factory)) == {"SCHEDULES"}
