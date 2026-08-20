"""The lock: what makes two runs of one package the same package.

A strategy package is a directory with files a human can edit — a mission prompt,
an optional alpha view, an output schema, a `strategy.toml`. Two runs that
disagree on any of those are not the same run, and the platform's whole purpose
is to compare agents on the *same* strategy. The lock pins what that sameness
means, written once per package load and read by `fintel report` so it needs no
`--strategy` flag.

The lock is a digest of *contents*, not paths: the same mission text at a
different filename is the same mission. It is also the only place a strategy's
data bindings are frozen alongside the catalog state they were checked against,
so a binding that passed preflight against catalog version A and would fail
against version B is detectable rather than silently re-validated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fintel.models.strategy import StrategyPaths
from fintel.strategy.views import AlphaViewLibrary

LOCK_VERSION = 1


def _sha(text: str | bytes) -> str:
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()[:16]


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _sha(path.read_bytes())


@dataclass(frozen=True)
class StrategyLock:
    """The frozen identity of one package load."""

    name: str
    strategy_digest: str  # hash of the manifest text
    mission_digest: str | None
    output_schema_digest: str | None
    alpha_view_digest: str | None  # standing file + every dated note; None if none
    catalog_sources: tuple[str, ...]  # source names available at preflight
    catalog_kinds: tuple[str, ...]
    created_at: str
    schema_version: int = LOCK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "strategy_digest": self.strategy_digest,
            "mission_digest": self.mission_digest,
            "output_schema_digest": self.output_schema_digest,
            "alpha_view_digest": self.alpha_view_digest,
            "catalog_sources": list(self.catalog_sources),
            "catalog_kinds": list(self.catalog_kinds),
            "created_at": self.created_at,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def build_lock(
    paths: StrategyPaths,
    *,
    catalog_sources: tuple[str, ...],
    catalog_kinds: tuple[str, ...],
    now: datetime | None = None,
) -> StrategyLock:
    """Build a lock from a loaded package. Reads the manifest text and the
    mission/output/alpha-view files for digests; missing optional files are None."""
    manifest_text = paths.manifest_file.read_text()
    return StrategyLock(
        name=paths.manifest.name,
        strategy_digest=_sha(manifest_text),
        mission_digest=_file_digest(paths.mission),
        output_schema_digest=_file_digest(paths.output_schema),
        alpha_view_digest=AlphaViewLibrary.load(paths).digest,
        catalog_sources=catalog_sources,
        catalog_kinds=catalog_kinds,
        created_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
    )


def read_lock(path: Path) -> StrategyLock:
    raw = json.loads(path.read_text())
    return StrategyLock(
        schema_version=int(raw["schema_version"]),
        name=raw["name"],
        strategy_digest=raw["strategy_digest"],
        mission_digest=raw.get("mission_digest"),
        output_schema_digest=raw.get("output_schema_digest"),
        alpha_view_digest=raw.get("alpha_view_digest"),
        catalog_sources=tuple(raw.get("catalog_sources", [])),
        catalog_kinds=tuple(raw.get("catalog_kinds", [])),
        created_at=raw["created_at"],
    )
