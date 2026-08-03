"""Semantic artifact writers — one place that knows WHAT each level persists.

`store.py` is the raw I/O layer (atomic write_json / write_model); this is the
layer above it: `write_cell`, `write_decision`, `write_trial_result`, … — each
binds a model to its path so a caller writes `write_cell(paths.cell(name),
record)` instead of `write_json(path, record.model_dump(mode="json"))`. The
shape of every artifact is declared once here, so the CLI's reader layer
(`cli/present.py`) and a re-run can point at the same schema rather than
re-deriving it at each call site.

Single-writer rules are still enforced by the callers (the trial writes
`decision.json` only after all cells; the job writes `result.json` once).
This module just makes each write a one-liner that can't drop the
`model_dump(mode="json")` or swap a path.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from fintel.models.common import Symbol
from fintel.models.decision import View
from fintel.models.job import JobResult
from fintel.models.run import RunResult
from fintel.models.trial import CellRecord, TrialResult
from fintel.simulate.store import write_json, write_model


def write_cell(cell_path: Path, record: CellRecord) -> None:
    write_model(cell_path, record)


def write_decision(decision_path: Path, views: dict[Symbol, View]) -> None:
    """One writer, after all cells on the date are done (the trial owns this)."""
    write_json(decision_path, {s: v.model_dump(mode="json") for s, v in views.items()})


def write_trial_result(result_path: Path, result: TrialResult) -> None:
    write_model(result_path, result)


def write_run_result(result_path: Path, result: RunResult) -> None:
    write_model(result_path, result)


def write_job_result(result_path: Path, result: JobResult) -> None:
    write_model(result_path, result)


def write_run_config(config_path: Path, config: BaseModel) -> None:
    write_model(config_path, config)


def write_run_lock(lock_path: Path, lock: BaseModel) -> None:
    write_model(lock_path, lock)


def write_job_config(config_path: Path, config: BaseModel) -> None:
    write_model(config_path, config)


def write_echo(echo_path: Path, echo: dict) -> None:
    write_json(echo_path, echo)


def write_fingerprint(fingerprint_path: Path, fingerprint) -> None:
    write_json(fingerprint_path, fingerprint.to_dict())


def write_health(health_path: Path, health: dict) -> None:
    write_json(health_path, health)


def write_prefetch(prefetch_path: Path, result) -> None:
    write_json(prefetch_path, result.to_dict())
