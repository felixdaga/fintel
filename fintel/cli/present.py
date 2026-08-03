"""The CLI's reader layer — typed loads + formatters for finished artifacts.

The simulation layer writes artifacts (`artifacts.py`); this is the matching
read side for the CLI. It loads each artifact into its model (so a corrupt or
truncated file is caught here, not mid-format) and renders a one-line summary
or a full block. Centralizing it means `cli/runs`, `cli/health` and
`cli/simulation` no longer each open-code `json.loads(...read_text())` — one
owner per artifact shape, paired with the writer in `artifacts.py`.

Failures degrade: a corrupt result.json shows as `status=corrupt` in the list
rather than crashing the whole listing, because a browse command should never
abort on one bad job.
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.models.job import JobResult
from fintel.models.run import RunResult
from fintel.models.trial import CellRecord
from fintel.simulate.store import read_json


def load_job_result(job_root: Path) -> JobResult | None:
    path = job_root / "result.json"
    if not path.is_file():
        return None
    try:
        return JobResult.model_validate(read_json(path))
    except Exception:
        return None


def load_run_result(run_root: Path) -> RunResult | None:
    path = run_root / "result.json"
    if not path.is_file():
        return None
    try:
        return RunResult.model_validate(read_json(path))
    except Exception:
        return None


def load_health(job_root: Path) -> dict | None:
    path = job_root / "health.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def load_cell_record(cell_path: Path) -> CellRecord | None:
    if not cell_path.is_file():
        return None
    try:
        return CellRecord.model_validate(read_json(cell_path))
    except Exception:
        return None


def load_decision(trial_dir: Path) -> dict | None:
    path = trial_dir / "decision.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def job_summary_line(job_root: Path) -> str:
    """One line for `fintel runs list`: name, status, health. A missing or
    corrupt result falls back to health.json, then to '?', so the listing
    never aborts on one bad job."""
    status = "?"
    health = "?"
    result = load_job_result(job_root)
    if result is not None:
        status = result.status
        health = result.health
    else:
        h = load_health(job_root)
        if h is not None:
            health = h.get("status", health)
            if status == "?":
                status = "corrupt"
    return f"{job_root.name:48}  status={status:8}  health={health}"


def print_job_artifacts(job_root: Path) -> None:
    """`fintel runs show`: dump the raw result/health/config JSON verbatim."""
    for name in ("result.json", "health.json", "config.json"):
        path = job_root / name
        if path.is_file():
            print(f"== {name}")
            print(path.read_text())


def print_decision_block(trial_dir: Path) -> None:
    """The per-date decision view used at the end of a simulation run."""
    decision = load_decision(trial_dir)
    if decision is None:
        return
    print(f"== decision {trial_dir.name}: {list(decision)}", flush=True)
    for sym, view in decision.items():
        print(
            f"   {sym}: score={view.get('score')}  "
            f"rationale={str(view.get('rationale', ''))[:80]}",
            flush=True,
        )


def print_json_block(name: str, obj: object) -> None:
    print(f"== {name}")
    print(json.dumps(obj, indent=2, default=str))
