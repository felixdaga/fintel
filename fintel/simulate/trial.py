"""One decision date: fan out its cells, reduce them into one decision.

The trial owns the single-writer rule for `decision.json`. Every cell writes its
own `cells/<cell>.json`; nothing writes `decision.json` until *all* cells on the
date are done. The old layout had every symbol write into one
`decisions/<date>.json`, so concurrent cells overwrote each other and a run
could finish with views missing and no error.

Cells run sequentially by default. A strategy with a thread-safe agent may set
`cell_concurrency > 1`; a non-thread-safe one (the LLM host, which accumulates
trace on `self`) must stay at 1, which is why it's the default and not derived
from the job's `max_concurrent`.
"""

from __future__ import annotations

import time
from datetime import date as Date
from pathlib import Path

from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig, cells_for
from fintel.market.data.base import DataSource
from fintel.market.settings import MarketConfig
from fintel.models.agent import AgentSpec
from fintel.models.decision import AgentResponse
from fintel.models.paths import TrialPaths
from fintel.models.trial import CellResult, TrialConfig, TrialResult
from fintel.simulate.cell import CellOutcome, run_cell
from fintel.environment.progress import NullProgress, Progress
from fintel.simulate.artifacts import write_decision, write_trial_result
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_decision, reduce_trial


def run_trial(
    *,
    trial_config: TrialConfig,
    sources: dict[str, DataSource],
    agent_spec: AgentSpec,
    runtime: RuntimeConfig,
    paths: TrialPaths,
    cell_concurrency: int = 1,
    mission_text: str = "",
    output_schema_text: str = "",
    market_config: MarketConfig | None = None,
    progress: Progress | None = None,
) -> TrialResult:
    """Fan out cells for one decision date, reduce, write the decision once."""
    progress = progress or NullProgress()
    started = time.perf_counter()
    started_at = _now_iso()

    cells = cells_for(
        run_id=trial_config.run_id,
        decision_date=trial_config.decision_date,
        symbols=list(trial_config.universe),
        scope=trial_config.scope,
    )
    progress.emit(
        "trial_start",
        decision_date=trial_config.decision_date.isoformat(),
        n_symbols=len(trial_config.universe),
        n_cells=len(cells),
    )

    if not cells:
        result = reduce_trial(
            trial_config.decision_date, [], started_at=started_at, finished_at=started_at
        )
        write_trial_result(paths.result, result)
        progress.emit(
            "trial_done",
            decision_date=trial_config.decision_date.isoformat(),
            status=result.status,
            n_views=0,
        )
        return result

    def _run_one(cell: Cell) -> CellOutcome | None:
        try:
            return run_cell(
                cell=cell,
                sources=sources,
                universe=list(trial_config.universe),
                agent_spec=agent_spec,
                runtime=runtime,
                cell_path=paths.cell(cell.name),
                mission_text=mission_text,
                output_schema_text=output_schema_text,
                market_config=market_config,
                progress=progress,
            )
        except Exception:
            # `invoke` already classifies adapter failures; an exception here is
            # a platform bug (e.g. a bad session path). Record the cell as
            # failed rather than aborting the date.
            return CellOutcome(
                result=CellResult(
                    cell=cell.name,
                    symbols=list(cell.symbols),
                    status="failed",
                    error="cell executor raised",
                    health="broken",
                    health_issues=["cell executor raised"],
                ),
                response=AgentResponse(views={}, outcome="crashed", detail="cell executor raised"),
            )

    outcomes = map_parallel(_run_one, cells, bound=cell_concurrency)
    cell_results: list[CellResult] = []
    responses: list[tuple[str, AgentResponse]] = []
    for cell, outcome in zip(cells, outcomes):
        if outcome is None:
            cell_results.append(
                CellResult(
                    cell=cell.name,
                    symbols=list(cell.symbols),
                    status="failed",
                    error="cell returned no outcome",
                    health="broken",
                    health_issues=["cell returned no outcome"],
                )
            )
            continue
        cell_results.append(outcome.result)
        responses.append((cell.name, outcome.response))

    # The fan-in: one writer, after all cells are done.
    decision = reduce_decision(responses)
    write_decision(paths.decision, decision)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result = reduce_trial(
        trial_config.decision_date,
        cell_results,
        started_at=started_at,
        finished_at=_now_iso(),
        duration_ms=elapsed_ms,
    )
    write_trial_result(paths.result, result)
    progress.emit(
        "trial_done",
        decision_date=trial_config.decision_date.isoformat(),
        status=result.status,
        n_views=result.n_views,
        health=result.health,
    )
    return result


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
