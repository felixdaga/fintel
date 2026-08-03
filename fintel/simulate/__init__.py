"""L7. Owns the fan-out: Job → Run → Trial → Cell.

The execution hierarchy. Each level builds the next level's config, fans out
with bounded concurrency, and reduces the results into one record — written
once, after its children are done, never concurrently. A failure at any level
is a recorded outcome, not a lost run: `agents.invoke` made that true at the
cell, and the same policy is carried up by `map_parallel` and the reducers.

Public entry point is `run_job`. The lower levels (`run_run`, `run_trial`,
`run_cell`) are exposed for tests and for a future `--only-run` / `--only-trial`
debugging surface.
"""

from __future__ import annotations

from fintel.simulate.artifacts import (
    write_cell,
    write_decision,
    write_echo,
    write_job_result,
    write_run_result,
    write_trial_result,
)
from fintel.simulate.cell import CellOutcome, build_agent, expect_tools, run_cell
from fintel.simulate.job import run_job
from fintel.environment.progress import NullProgress
from fintel.simulate.queue import map_parallel
from fintel.simulate.record import reduce_decision, reduce_job, reduce_run, reduce_trial
from fintel.simulate.run import run_run
from fintel.simulate.store import read_json, read_model, write_json, write_model
from fintel.simulate.trial import run_trial

__all__ = [
    "CellOutcome",
    "NullProgress",
    "build_agent",
    "expect_tools",
    "map_parallel",
    "read_json",
    "read_model",
    "reduce_decision",
    "reduce_job",
    "reduce_run",
    "reduce_trial",
    "run_cell",
    "run_job",
    "run_run",
    "run_trial",
    "write_cell",
    "write_decision",
    "write_echo",
    "write_json",
    "write_job_result",
    "write_model",
    "write_run_result",
    "write_trial_result",
]
