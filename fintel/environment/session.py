"""Per-cell filesystem isolation.

The path is derived from the cell, not discovered. The old code located a cell's
working directory by walking process ancestry for the CLI's pid and reading
`cli-pids/<pid>/bundle.json`; under a gateway that spawns the server itself the
walk finds nothing and every cell shares one directory. Concurrency was actually
saved by a pool of per-slot directories, which meant the pid scheme was doing
nothing while appearing to be the mechanism.

Two rules follow from that:

  * the path is a pure function of the cell, so two cells cannot collide and a
    reader never has to guess which cell a directory belongs to;
  * reuse is refused rather than silently accepted, because a non-empty directory
    means either a colliding cell or a stale one, and both produced real bugs.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from fintel.environment.cell import Cell

# A handshake file, so a subprocess reads its cell identity from one named place
# instead of inferring it. Absent means "not set up yet" and must fail loudly.
CELL_FILE = "cell.json"
RESULT_FILE = "result.json"
TRACE_FILE = "access.jsonl"
ENV_SESSION_DIR = "FINTEL_SESSION_DIR"


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionDir:
    """The only directory a cell's agent may write to."""

    root: Path
    cell: Cell

    @property
    def path(self) -> Path:
        return self.root / self.cell.run_id / self.cell.trial / self.cell.name

    @property
    def cell_file(self) -> Path:
        return self.path / CELL_FILE

    @property
    def result(self) -> Path:
        return self.path / RESULT_FILE

    @property
    def trace(self) -> Path:
        return self.path / TRACE_FILE

    def create(self, *, reset: bool = False) -> Path:
        """Make a directory this cell owns exclusively.

        A populated directory is a collision or a leftover. `reset=True` says
        "I know, discard it"; the default refuses, because silently inheriting
        another cell's files is how a stale decision date gets served.
        """
        if self.path.exists() and any(self.path.iterdir()):
            if not reset:
                raise SessionError(
                    f"session dir for {self.cell.describe()} already has contents: "
                    f"{self.path}. Pass reset=True to discard it, or use a fresh "
                    f"run_id — reusing it risks serving another cell's state."
                )
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cell_file.write_text(
            json.dumps(
                {
                    "run_id": self.cell.run_id,
                    "decision_date": self.cell.decision_date.isoformat(),
                    "symbols": list(self.cell.symbols),
                    "scope": self.cell.scope,
                    "cell": self.cell.name,
                },
                indent=2,
            )
        )
        return self.path

    def env(self) -> dict[str, str]:
        """What a subprocess needs to find its own session. Never credentials —
        a secret written here would land in a run artifact."""
        return {ENV_SESSION_DIR: str(self.path)}

    def discard(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)


def read_cell(session_path: str | Path) -> dict:
    """Load a cell's identity from its directory.

    Fails loudly when absent. The old server treated a missing bundle as
    "register the tools anyway and work it out later", then cached whatever it
    eventually found for the lifetime of the process.
    """
    path = Path(session_path) / CELL_FILE
    if not path.is_file():
        raise SessionError(
            f"no {CELL_FILE} in {session_path}; the session was not set up, or "
            f"{ENV_SESSION_DIR} points at the wrong directory"
        )
    return json.loads(path.read_text())


def session_dir_from_env() -> Path:
    raw = os.environ.get(ENV_SESSION_DIR, "").strip()
    if not raw:
        raise SessionError(
            f"{ENV_SESSION_DIR} is not set; a cell's working directory is passed "
            f"explicitly, never inferred from the process tree"
        )
    return Path(raw)
