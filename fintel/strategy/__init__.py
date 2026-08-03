"""L6. Load a strategy package, validate it, freeze it.

A strategy *package* is an external directory: `strategy.toml` plus a mission
prompt and an output schema. This module is the platform's only interface to
one — it does not contain investment logic, which lives in the package and in
the scoring layer. Three composable steps:

  * `load`    — parse the manifest. No I/O beyond the file.
  * `preflight` — every reason the package cannot run, before any cost.
  * `build_lock` — freeze the package's identity for replay comparison.

Composed as `load_and_prepare`, which is what the CLI and evaluate call.
"""

from __future__ import annotations

from pathlib import Path

from fintel.market import catalog
from fintel.models.strategy import StrategyPaths
from fintel.strategy.load import ManifestError, PackageNotFound, load
from fintel.strategy.lock import StrategyLock, build_lock, read_lock
from fintel.strategy.preflight import PreflightError, PreflightResult, preflight

__all__ = [
    "ManifestError",
    "PackageNotFound",
    "PreflightError",
    "PreflightResult",
    "StrategyLock",
    "StrategyPaths",
    "build_lock",
    "load",
    "load_and_prepare",
    "preflight",
    "read_lock",
]


def load_and_prepare(
    package_dir: str | Path,
    *,
    env: dict[str, str] | None = None,
    write_lock: bool = True,
) -> tuple[StrategyPaths, PreflightResult, StrategyLock]:
    """Load, preflight, and lock in one call. The normal entry point for a run.

    `write_lock=True` writes `strategy.lock` into the package root so a later
    `fintel report` can read identity without re-deriving. Set False for a
    read-only inspection (e.g. `--list-strategy`).
    """
    paths = load(package_dir)
    result = preflight(paths, env=env)
    lock = build_lock(
        paths,
        catalog_sources=tuple(sorted(s.name for s in catalog.sources())),
        catalog_kinds=tuple(sorted(catalog.kinds())),
    )
    if write_lock:
        lock.write(paths.lock)
    return paths, result, lock
