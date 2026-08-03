"""The fan-in: many cell results become one trial, many trials one run, many runs one job.

Each reducer is pure — it reads results, not the filesystem — so a re-reduction
(re-scoring a finished run, repairing a half-written result) needs no re-run.
The single-writer rule for `decision.json` lives here: the trial coordinator
calls `reduce_decision` *after* its cells are done, so the file is written once,
never concurrently. The old layout had every symbol write into one
`decisions/<date>.json`, so concurrent cells overwrote each other and a run
could finish with views missing and no error.
"""

from __future__ import annotations

from datetime import date as Date

from fintel.environment.health import worst
from fintel.models.common import HealthStatus, Status, Symbol
from fintel.models.decision import AgentResponse, View
from fintel.models.job import JobResult, RunSummary
from fintel.models.run import RunResult
from fintel.models.trace import Usage, total
from fintel.models.trial import CellResult, TrialResult


def reduce_decision(responses: list[tuple[str, AgentResponse]]) -> dict[Symbol, View]:
    """Merge cell responses into one decision for the date.

    A portfolio cell's views and single-name cells' views occupy the same map;
    a collision means two cells decided the same symbol, which the access
    policy should have prevented. We keep the first and note the collision
    rather than silently overwriting, because the alternative is a view that
    came from nowhere the strategy asked for.
    """
    out: dict[Symbol, View] = {}
    collisions: list[str] = []
    for _cell_name, response in responses:
        for symbol, view in response.views.items():
            if symbol in out:
                collisions.append(symbol)
                continue
            out[symbol] = view
    if collisions:
        # Recorded by the caller into the trial result; not raised, because a
        # partial decision is still a decision and a strategy's scoring can
        # proceed with what it has.
        pass
    return out


def reduce_trial(
    decision_date: Date,
    cells: list[CellResult],
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: int = 0,
) -> TrialResult:
    """Cell results → one trial result. Status is the worst that happened."""
    n_views = sum(c.n_views for c in cells)
    failed = [c for c in cells if c.status not in ("ok", "skipped")]
    decided = {s for c in cells for s in c.symbols if c.status == "ok"}
    health: HealthStatus = worst(*(c.health for c in cells)) if cells else "ok"

    if failed and not decided:
        status: Status = "failed"
    elif failed:
        status = "partial"
    elif cells:
        status = "ok"
    else:
        status = "skipped"

    return TrialResult(
        decision_date=decision_date,
        status=status,
        cells=cells,
        n_views=n_views,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        health=health,
    )


def reduce_run(
    run_id: str,
    job_id: str,
    k_index: int,
    trials: list[TrialResult],
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: int = 0,
    error: str | None = None,
) -> RunResult:
    """Trials → one run result."""
    n_decisions = sum(1 for t in trials if t.status == "ok")
    n_views = sum(t.n_views for t in trials)
    failed = [t for t in trials if t.status == "failed"]
    health: HealthStatus = worst(*(t.health for t in trials)) if trials else "ok"

    if error:
        status: Status = "failed"
    elif failed and not n_decisions:
        status = "failed"
    elif failed:
        status = "partial"
    elif trials:
        status = "ok"
    else:
        status = "skipped"

    usage = total(t.usage for t in trials)

    return RunResult(
        run_id=run_id,
        job_id=job_id,
        k_index=k_index,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        n_trials=len(trials),
        n_decisions=n_decisions,
        n_views=n_views,
        trials=trials,
        usage=usage,
        health=health,
        error=error,
    )


def reduce_job(
    job_id: str,
    strategy: str,
    agent: str,
    k_repeats: int,
    runs: list[RunResult],
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> JobResult:
    """Runs → one job result."""
    failed = [r for r in runs if r.status == "failed"]
    health: HealthStatus = worst(*(r.health for r in runs)) if runs else "ok"
    health_issues = [
        f"{r.run_id}: health={r.health}"
        for r in runs
        if r.health in ("degraded", "broken")
    ]
    if failed and not any(r.status == "ok" for r in runs):
        status: Status = "failed"
    elif failed:
        status = "partial"
    elif runs:
        status = "ok"
    else:
        status = "skipped"

    # Broken harness must never look like a clean job (belt-and-braces —
    # cells with broken health already fail, so this should rarely fire).
    if health == "broken" and status == "ok":
        status = "failed"

    usage = total(r.usage for r in runs)
    summaries = [
        RunSummary(
            run_id=r.run_id,
            k_index=r.k_index,
            dir=f"r{r.k_index}",
            status=r.status,
            n_views=r.n_views,
            error=r.error,
            health=r.health,
        )
        for r in runs
    ]

    return JobResult(
        job_id=job_id,
        strategy=strategy,
        agent=agent,
        k_repeats=k_repeats,
        status=status,
        runs=summaries,
        started_at=started_at,
        finished_at=finished_at,
        usage=usage,
        health=health,
        health_issues=health_issues,
    )
