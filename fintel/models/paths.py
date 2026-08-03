"""On-disk layout. Path conventions only — no reading, no parsing.

runs/<job_id>/config.json result.json job.log
              r1/config.json lock.json fingerprint.json result.json run.log memory.jsonl
                 trials/<date>/decision.json result.json
                               cells/<cell>.json      one writer each
                               trace/<cell>.jsonl     one writer each
              report/report.md report.json

Anything a cell writes is named after that cell. The old layout had every symbol
on a decision date write into one `decisions/<date>.json`, so concurrent cells
overwrote each other and a run could finish with views missing and no error.
`decision.json` still exists, but it is a *reduction* — written once by the trial
coordinator after its cells are done, never concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fintel.models import ids


@dataclass(frozen=True)
class TrialPaths:
    root: Path

    @property
    def decision(self) -> Path:
        """The reduced decision for this date. Single writer, after the fan-in."""
        return self.root / "decision.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def cells_dir(self) -> Path:
        return self.root / "cells"

    def cell(self, cell: str) -> Path:
        """One cell's own output. No other cell may write here."""
        return self.cells_dir / f"{cell}.json"

    def cell_files(self) -> list[Path]:
        if not self.cells_dir.is_dir():
            return []
        return sorted(self.cells_dir.glob("*.json"))

    @property
    def trace_dir(self) -> Path:
        return self.root / "trace"

    def trace(self, cell: str) -> Path:
        return self.trace_dir / f"{cell}.jsonl"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def lock(self) -> Path:
        return self.root / "lock.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def fingerprint(self) -> Path:
        """The agent's identity for this run: model, channel, prompt hash. The
        strategy's own identity is `lock`; together they pin what produced it."""
        return self.root / "fingerprint.json"

    @property
    def echo(self) -> Path:
        """The run echo: every input gathered before any cell runs. See
        `environment/echo.py` — printed at run start and persisted here."""
        return self.root / "echo.json"

    @property
    def log(self) -> Path:
        return self.root / "run.log"

    @property
    def memory(self) -> Path:
        return self.root / "memory.jsonl"

    @property
    def trials_dir(self) -> Path:
        return self.root / "trials"

    def trial(self, decision_date: date) -> TrialPaths:
        return TrialPaths(root=self.trials_dir / ids.trial_id(decision_date))

    def trial_dirs(self) -> list[Path]:
        if not self.trials_dir.is_dir():
            return []
        return sorted(p for p in self.trials_dir.iterdir() if p.is_dir())


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @classmethod
    def under(cls, output_root: str | Path, job_id: str) -> JobPaths:
        return cls(root=Path(output_root) / job_id)

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def log(self) -> Path:
        return self.root / "job.log"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    def run(self, k_index: int) -> RunPaths:
        return RunPaths(root=self.root / f"r{k_index}")

    def run_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            (
                p
                for p in self.root.iterdir()
                if p.is_dir()
                and p.name.startswith("r")
                and p.name[1:].isdigit()  # r1, r2, … — not "report" or "result"
            ),
            key=lambda p: int(p.name[1:]),
        )
