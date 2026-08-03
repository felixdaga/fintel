"""`strategy.toml` — the package's mission, its data needs, and how it's judged."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fintel.models.common import DecisionScope
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef

MANIFEST_NAME = "strategy.toml"
LOCK_NAME = "strategy.lock"


class DecisionSpec(BaseModel):
    """`scope` decides what one agent invocation owns, hence the fan-out."""

    model_config = ConfigDict(extra="forbid")

    scope: DecisionScope = "single_name"
    schedule: ScheduleRef


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kpi: str  # builtin name or module:Class
    # The signal: how the decision's metrics become THE signal. Resolved by the
    # evaluation layer (`evaluate/signals.py`); the platform owns the mechanics
    # around it (transform, ensemble, holdings), the strategy owns what the
    # signal *is*. Defaults to the view's score (single-name).
    signal: str = "single_name"
    params: dict = Field(default_factory=dict)
    transform: str = "rank_range"
    horizons: list[int] = Field(default_factory=lambda: [1, 2, 3])
    metric_key: str = "icir"


class StrategyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str
    description: str = ""

    universe: UniverseRef
    decision: DecisionSpec
    data: list[DataBinding] = Field(default_factory=list)
    scoring: ScoringSpec

    mission_file: str = "mission.md"
    output_schema_file: str = "output_schema.json"
    cache_dir: str = "cache"

    @model_validator(mode="after")
    def _unique_kinds(self) -> StrategyManifest:
        kinds = [b.kind for b in self.data]
        dupes = {k for k in kinds if kinds.count(k) > 1}
        if dupes:
            raise ValueError(f"duplicate data kinds: {sorted(dupes)}")
        return self

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(b.kind for b in self.data)


@dataclass(frozen=True)
class StrategyPaths:
    root: Path
    manifest: StrategyManifest

    @property
    def manifest_file(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def mission(self) -> Path:
        return self.root / self.manifest.mission_file

    @property
    def output_schema(self) -> Path:
        return self.root / self.manifest.output_schema_file

    @property
    def cache_dir(self) -> Path:
        return self.root / self.manifest.cache_dir

    @property
    def lock(self) -> Path:
        return self.root / LOCK_NAME
