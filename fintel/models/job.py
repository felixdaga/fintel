"""One `fintel backtest` invocation: package × agent × market × K repeats."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fintel.models.agent import AgentSpec
from fintel.models.common import Status
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.models.trace import Usage


class JobConfig(BaseModel):
    """What the caller asked for. Overrides are None when the package's own
    declaration should stand."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    job_id: str
    strategy: str
    agent: AgentSpec

    k_repeats: int = Field(default=1, ge=1)
    max_concurrent: int = Field(default=1, ge=1)
    output_root: str = "runs"
    seed: int | None = None
    dry_run: bool = False

    universe: UniverseRef | None = None
    schedule: ScheduleRef | None = None
    data: list[DataBinding] | None = None

    @property
    def peak_concurrent(self) -> int:
        return self.k_repeats * self.max_concurrent


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    k_index: int
    dir: str
    status: Status
    n_views: int = 0
    error: str | None = None


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    job_id: str
    strategy: str
    agent: str
    k_repeats: int
    status: Status
    runs: list[RunSummary] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    usage: Usage = Field(default_factory=Usage)
