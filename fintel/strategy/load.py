"""Load a strategy package: a directory with a `strategy.toml`.

One entry point, `load`, returns `StrategyPaths` (the directory plus the parsed
manifest). It does *not* preflight or lock — those are separate, composable
steps, so a tool that only needs the manifest (e.g. `--list-strategy`) doesn't
pay for validation, and a re-lock after a mission edit doesn't re-validate.

TOML is read with `tomllib` (stdlib, 3.11+). A package may ship its own data
sources as `module:Callable`; loading does not import them, so a package
referencing a source whose module isn't installed still loads — it only fails
at preflight or build, where the error is actionable.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from fintel.models.strategy import MANIFEST_NAME, StrategyManifest, StrategyPaths


class PackageNotFound(FileNotFoundError):
    pass


class ManifestError(ValueError):
    """The manifest is missing, unreadable, or fails schema validation."""


def load(package_dir: str | Path) -> StrategyPaths:
    """Load and parse a strategy package. Raises `PackageNotFound` if the
    directory is absent, `ManifestError` if the manifest is missing or invalid."""
    root = Path(package_dir)
    if not root.is_dir():
        raise PackageNotFound(f"strategy package not found: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestError(f"no {MANIFEST_NAME} in {root}")

    try:
        raw = tomllib.loads(manifest_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"cannot parse {manifest_path}: {exc}") from exc

    try:
        manifest = StrategyManifest(**raw)
    except Exception as exc:
        raise ManifestError(f"invalid manifest in {manifest_path}: {exc}") from exc

    return StrategyPaths(root=root, manifest=manifest)
