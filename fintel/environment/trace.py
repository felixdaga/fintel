"""What the agent actually saw, as an append-only record.

The point is auditability after the fact: given a decision, a reviewer must be
able to establish which questions were asked, which were refused, and which came
back empty versus failed. Recording only successful reads would make a run that
half-failed look like a run against a quiet company.

Line-per-event JSONL, flushed on write, so a crashed cell keeps everything up to
the crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fintel.environment.access import Reading
from fintel.environment.cell import Cell

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AccessLog:
    """Append-only. `path=None` keeps the record in memory, for tests and dry runs."""

    cell: Cell
    path: Path | None = None
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.append(
            "cell_opened",
            run_id=self.cell.run_id,
            decision_date=self.cell.decision_date.isoformat(),
            symbols=list(self.cell.symbols),
            scope=self.cell.scope,
        )

    def append(self, event: str, **fields) -> dict:
        record = {"ts": _now(), "event": event, **fields}
        self.events.append(record)
        if self.path is not None:
            try:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
            except OSError as exc:
                # Losing the trace must not lose the run, but it must be visible.
                logger.warning("could not write access log %s: %s", self.path, exc)
        return record

    def record(self, reading: Reading) -> dict:
        return self.append("read", **reading.record())

    def denied(self, what: str, detail: str) -> dict:
        return self.append("denied", what=what, detail=detail)

    def submitted(self, *, n_views: int, dropped: list[str]) -> dict:
        return self.append("submitted", n_views=n_views, dropped=dropped)

    # ── reading the record back ──────────────────────────────────────────────

    @property
    def reads(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "read"]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for read in self.reads:
            status = read.get("status", "unknown")
            out[status] = out.get(status, 0) + 1
        return out

    def kinds_used(self) -> tuple[str, ...]:
        return tuple(sorted({r["kind"] for r in self.reads if r.get("kind")}))

    def summary(self) -> dict:
        """Enough for a cell record to be honest about data quality."""
        counts = self.counts()
        return {
            "n_reads": len(self.reads),
            "by_status": counts,
            "kinds_used": list(self.kinds_used()),
            "degraded": bool(counts.get("failed") or counts.get("denied")),
        }


def load(path: str | Path) -> list[dict]:
    """Read a trace back. Tolerates a truncated final line from a killed cell."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed trace line in %s", path)
    return out
