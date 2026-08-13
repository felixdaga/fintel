"""Models for the evaluation layer — read-only analytics over finished runs.

The evaluation layer consumes finished run artifacts (decisions, cells) and
produces analytics. These models are the in-memory shape the reader (`evaluate/
read.py`) builds and the analytics modules (`signals`, `kpi`, `holdings`,
`behaviour`, `variance`) operate on. Nothing here does I/O.

The abstraction (see docs/architecture.md §1): the strategy defines **the
signal** (`signal_fn(views) -> signal`) and **the KPI** (`kpi_fn(signal, prices)
-> metric`); the platform owns only the mechanics — ensemble, holdings, returns,
stochasticity, rendering. These models carry no opinion about how a signal or a
KPI is computed.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field

from fintel.models.common import Symbol
from fintel.models.decision import View


class CellBehaviour(BaseModel):
    """Per-cell behaviour for the L1 stochasticity layer.

    `has_trace=False` marks an agent with no tool-call trace (scripted / constant
    baselines). The behaviour layer no-ops on those rather than reporting zeros
    that would read as "perfectly stable".
    """

    cell: str
    decision_date: str
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_reads: int = 0
    outcome: str = ""
    has_trace: bool = False


class RunData(BaseModel):
    """One repeat's loaded artifacts — the reader builds this from a run dir.

    `views_by_date` carries the agent's decisions; `behaviour_by_date` carries
    the per-cell process stats. The universe is the resolved set of symbols the
    run actually scored (post cell-fan-out), per decision date.
    """

    run_id: str
    k_index: int
    decision_dates: list[Date] = Field(default_factory=list)
    universe: list[Symbol] = Field(default_factory=list)
    views_by_date: dict[Date, dict[Symbol, View]] = Field(default_factory=dict)
    behaviour_by_date: dict[Date, dict[Symbol, CellBehaviour]] = Field(default_factory=dict)

    def universe_at(self, decision_date: Date) -> list[Symbol]:
        """The symbols scored on a given date, in stable order."""
        return sorted(self.views_by_date.get(decision_date, {}).keys())


class Signals(BaseModel):
    """The signal series for a job.

    `per_run` is one signal map per repeat (each `dict[Date, dict[Symbol,
    float]]`); `ensemble` is the cell-mean across repeats. This is the
    platform-mechanics output the KPI and holdings layers consume — it carries
    no strategy opinion about *what* the signal is, only its values.
    """

    per_run: list[dict[Date, dict[Symbol, float]]] = Field(default_factory=list)
    ensemble: dict[Date, dict[Symbol, float]] = Field(default_factory=dict)
    decision_dates: list[Date] = Field(default_factory=list)
    universe: list[Symbol] = Field(default_factory=list)


class ReportPayload(BaseModel):
    """The full evaluation result for one job, written to
    `runs/<job>/report/report.json` and rendered to `report.md`.

    The layers are the strategy-defined KPI (`kpi`), the always-on
    stochasticity layers (`behaviour`, `variance`), the opt-in holdings
    (`holdings`, None when not requested), and the opt-in agent-on-agent
    evaluation (`agent_eval`, None when the pack declares no `[eval]` section).
    `meta` carries the run identity.
    """

    job_id: str
    k_repeats: int
    strategy: str
    signal: str
    transform: str
    kpi: str
    metric_key: str = "icir"
    horizons: list[int] = Field(default_factory=list)
    decision_dates: list[Date] = Field(default_factory=list)
    universe: list[Symbol] = Field(default_factory=list)
    kpi_result: dict = Field(default_factory=dict)
    behaviour: dict = Field(default_factory=dict)
    variance: dict = Field(default_factory=dict)
    holdings: dict | None = None
    agent_eval: dict | None = None
    meta: dict = Field(default_factory=dict)
