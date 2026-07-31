"""One of K repeats: the frozen effective config, its lock, and its result."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fintel.models.agent import AgentSpec
from fintel.models.common import DecisionScope, Status, Symbol
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.models.strategy import ScoringSpec
from fintel.models.trace import Usage
from fintel.models.trial import TrialResult


class StrategyRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    digest: str | None = None


class RunConfig(BaseModel):
    """The effective world after package defaults + job overrides.

    Self-describing on purpose: a finished or crashed run is reproducible from
    this file alone, without re-deriving any default.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    job_id: str
    k_index: int
    k_repeats: int
    created_at: str

    strategy: StrategyRef
    agent: AgentSpec
    scope: DecisionScope
    universe: UniverseRef
    universe_symbols: list[Symbol]
    schedule: ScheduleRef
    schedule_dates: list[str]
    data: list[DataBinding]
    scoring: ScoringSpec


class RunLock(BaseModel):
    """Digests for replay comparison. `fintel report` reads identity from here,
    which is why it needs no `--strategy`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    fingerprint: str
    code_version: str
    strategy_digest: str | None = None
    cache_digest: str | None = None
    model: dict = Field(default_factory=dict)
    prompts: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    job_id: str
    k_index: int
    status: Status
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int = 0
    n_trials: int = 0
    n_decisions: int = 0
    n_views: int = 0
    trials: list[TrialResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
