"""Event timeline — a curated, PIT-clamped chronology shipped with a package.

A geopol event package ships an ``event.md`` (or ``event.json``) file containing
a dated chronology of the event. This source reads that file and returns only
the entries whose ``entry_date`` is strictly before the decision date — the same
PIT discipline every other source enforces, applied to a curated one-off
database that is reused across all runs.

File format (Markdown, one entry per line starting with an ISO date):

    2018-03-22 — Trump signs Section 301 memorandum directing USTR to investigate
    Chinese trade practices. USTR probe announced.
    2018-04-03 — USTR publishes proposed $50B tariff list (1,333 Chinese products).
    ...

Lines without a leading date are treated as continuation of the previous entry
(so a single event can span multiple lines). Blank lines separate entries.

The source is ``subject="none"`` — the timeline is shared across all parties
(the event is the event). Both cells see the same timeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

from fintel.market.data.base import DataError
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[—\-:]\s*(.+)")


@dataclass
class EventTimeline:
    """Curated event chronology, PIT-clamped on entry date.

    Reads ``event_file`` (a path, resolved relative to CWD unless absolute) once
    on first fetch, caches the parsed entries in memory. Each fetch returns the
    entries with ``entry_date < decision_date`` and ``>= decision_date - lookback``.
    """

    event_file: str = "event.md"
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    name: str = "event_timeline"
    kinds: tuple[str, ...] = ("event_timeline",)
    _entries: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        path = Path(self.event_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise DataError(
                f"event_timeline: event file not found at {path}. "
                f"Pass event_file= in the [[data]] binding."
            )
        if path.suffix == ".json":
            self._load_json(path)
        else:
            self._load_md(path)
        self._loaded = True

    def _load_md(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        entries: list[dict[str, str]] = []
        current_date: str | None = None
        current_body: list[str] = []

        def _flush() -> None:
            nonlocal current_date, current_body
            if current_date is not None:
                entries.append(
                    {"entry_date": current_date, "body": "\n".join(current_body).strip()}
                )
            current_date = None
            current_body = []

        for line in text.splitlines():
            m = _DATE_RE.match(line.strip())
            if m:
                _flush()
                current_date = m.group(1)
                current_body = [m.group(2).strip()]
            elif line.strip() == "":
                _flush()
            elif current_date is not None:
                current_body.append(line.strip())
        _flush()
        self._entries = entries

    def _load_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise DataError(f"event_timeline: JSON file must be a list of entries")
        self._entries = [
            {"entry_date": str(e["entry_date"]), "body": str(e.get("body", ""))}
            for e in data
            if e.get("entry_date")
        ]

    def fetch(self, query: dict, cutoff: Cutoff) -> dict[str, Any]:
        decision_date = cutoff.decision_date
        lookback = int(query.get("lookback_days", self.lookback_days))
        since = decision_date - timedelta(days=lookback)
        through = decision_date - timedelta(days=1)

        visible = [
            e
            for e in self._entries
            if since.isoformat() <= e["entry_date"] <= through.isoformat()
        ]
        return {
            "as_of": decision_date.isoformat(),
            "lookback_days": lookback,
            "n_entries_total": len(self._entries),
            "n_entries_visible": len(visible),
            "entries": visible,
        }

    def warm(self, since: Date, through: Date) -> None:
        """No-op — the timeline is loaded once, no network fetch."""
        pass
