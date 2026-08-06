"""One `fintel simulation` invocation: package × agent × market × K repeats."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fintel.models.agent import AgentSpec
from fintel.models.common import HealthStatus, Status
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.models.trace import Usage

# Sentinel: "auto" — resolve at runtime to the natural fan-out width for that
# level. None lets a single config express "all parallel" without hard-coding a
# number that has to track the universe size or K.
AUTO = None


class JobConfig(BaseModel):
    """What the caller asked for. Overrides are None when the package's own
    declaration should stand.

    Concurrency is two nested axes, plus an optional flat pool:

      · `max_concurrent`  — across K repeats (parallel runs). Auto = K, so all
        repeats run at once — the "3 runs parallel" case.
      · `cell_concurrency` — across cells within one trial (concurrent
        tickers). Auto = universe size at each date, so all tickers run at once
        — the "10 tickers concurrent" case. Safe regardless of memory: memory
        writes happen after the whole trial completes, never per-cell.
      · `trial_concurrency` — across dates within a run. Default 1 (sequential)
        because a date's session carries the prior date's portfolio + memory.
        The memory guard in `run_run` forces this to 1 when memory is on, so a
        misconfigured job can't race on shared state.
      · `shared_concurrency` — flat pool across *all* (date, ticker) cells in a
        run. Keeps N cells in flight and rolls to the next date as slots free.
        When set, it replaces the nested cell × trial fan-out. Blocked when
        memory or feedback couples dates (independent cells only).

    Peak in-flight sessions without shared = resolved `max_concurrent` ×
    resolved `cell_concurrency` (e.g. 3 × 10 = 30). With shared =
    `max_concurrent` × `shared_concurrency`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    job_id: str
    strategy: str
    agent: AgentSpec

    k_repeats: int = Field(default=1, ge=1)
    max_concurrent: int | None = Field(default=AUTO, ge=1)
    cell_concurrency: int | None = Field(default=AUTO, ge=1)
    trial_concurrency: int = Field(default=1, ge=1)
    shared_concurrency: int | None = Field(default=None, ge=1)
    output_root: str = "runs"
    seed: int | None = None
    dry_run: bool = False

    # Warm the on-disk cache once before K fan-out so OpenClaw's parallel MCP
    # tool calls hit a hot cache (ms) instead of cold Massive fills (>60s →
    # host MCP timeout -32001). Runtime-only kinds (web_search) are skipped.
    prefetch: bool = True
    prefetch_workers: int = 8

    universe: UniverseRef | None = None
    schedule: ScheduleRef | None = None
    data: list[DataBinding] | None = None

    def resolve_run_concurrency(self) -> int:
        """K repeats in flight. Auto = all of them."""
        return self.max_concurrent if self.max_concurrent is not None else self.k_repeats

    def resolve_cell_concurrency(self, universe_size: int) -> int:
        """Cells in flight within one trial. Auto = the whole universe, so
        every ticker decides at once. Capped to `universe_size` even when set
        explicitly, since there are no more cells than tickers."""
        if self.cell_concurrency is None:
            return max(1, universe_size)
        return max(1, min(self.cell_concurrency, universe_size))

    @property
    def peak_concurrent(self) -> int:
        """An upper bound on simultaneous sessions, for capacity planning. The
        real peak depends on per-date universe size, which isn't known until
        runtime, so this uses K for the cell side when auto."""
        run_n = self.resolve_run_concurrency()
        if self.shared_concurrency is not None:
            return run_n * self.shared_concurrency
        cell_n = self.cell_concurrency if self.cell_concurrency is not None else self.k_repeats
        return run_n * max(1, cell_n)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    k_index: int
    dir: str
    status: Status
    n_views: int = 0
    error: str | None = None
    health: HealthStatus = "ok"


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
    health: HealthStatus = "ok"
    health_issues: list[str] = Field(default_factory=list)
