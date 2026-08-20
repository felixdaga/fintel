"""Alpha view: a strategy-owned thesis, composed into every agent's prompts.

The pack rubric lives in ``mission.md``. An optional **alpha view** is a
separate standing thesis (``alpha_view.md``) plus optional dated research
notes (``alpha_views/YYYY-MM-DD.md``). The strategy module loads the library
once, resolves it point-in-time against a decision date, and returns plain
strings. Simulate (L7) passes those strings into agents (L5); agents never
import this module.

Resolution (v1): standing file + the latest dated note with ``as_of`` ≤ the
decision date. Future-dated notes are stored (they are part of package
identity) but never shown. Accumulating every note up to the date is a
later knob; do not do it here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fintel.models.strategy import StrategyPaths

logger = logging.getLogger(__name__)

_NOTE_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
_ALPHA_VIEW_HEADING = "## Alpha view"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DatedNote:
    """One as-of research note. ``as_of`` is the first date the note is visible."""

    as_of: date
    text: str


@dataclass(frozen=True)
class AlphaViewLibrary:
    """The pack's standing thesis plus every dated note, unsorted of identity.

    ``notes`` is sorted by ``as_of`` ascending. ``digest`` hashes the whole
    library (not one date's resolution) so a run fingerprint is stable across
    the schedule; PIT clipping happens only at ``resolve``.
    """

    standing: str
    notes: tuple[DatedNote, ...]
    digest: str | None

    @classmethod
    def load(cls, paths: StrategyPaths) -> AlphaViewLibrary:
        standing = ""
        if paths.alpha_view.is_file():
            standing = paths.alpha_view.read_text()

        notes: list[DatedNote] = []
        notes_dir = paths.alpha_views_dir
        if notes_dir.is_dir():
            for path in sorted(notes_dir.iterdir()):
                note = _read_dated_note(path)
                if note is not None:
                    notes.append(note)
        frozen = tuple(notes)
        return cls(standing=standing, notes=frozen, digest=_library_digest(standing, frozen))

    def resolve(self, decision_date: date) -> str:
        """PIT: standing thesis + latest note with ``as_of`` ≤ ``decision_date``."""
        parts: list[str] = []
        if self.standing.strip():
            parts.append(self.standing.strip())
        latest: DatedNote | None = None
        for note in self.notes:
            if note.as_of <= decision_date:
                latest = note
            else:
                break
        if latest is not None and latest.text.strip():
            parts.append(f"### Research note ({latest.as_of.isoformat()})\n\n{latest.text.strip()}")
        return "\n\n".join(parts)


@dataclass(frozen=True)
class PackContext:
    """Mission, schema, names, and alpha-view library — loaded once per job."""

    mission_text: str
    output_schema_text: str
    company_names: dict[str, str]
    alpha_views: AlphaViewLibrary


def load_pack_context(paths: StrategyPaths) -> PackContext:
    """Read the pack files simulate threads into every cell.

    Mission and schema are the raw files. Alpha view is the *library*; each
    cell resolves it against that cell's decision date.
    """
    mission_text = paths.mission.read_text() if paths.mission.is_file() else ""
    output_schema_text = paths.output_schema.read_text() if paths.output_schema.is_file() else ""
    company_names: dict[str, str] = {}
    if paths.company_names.is_file():
        try:
            loaded = json.loads(paths.company_names.read_text())
            if isinstance(loaded, dict):
                company_names = {str(k): str(v) for k, v in loaded.items()}
        except (json.JSONDecodeError, ValueError):
            logger.warning("company_names.json is not valid JSON; ignored")
    return PackContext(
        mission_text=mission_text,
        output_schema_text=output_schema_text,
        company_names=company_names,
        alpha_views=AlphaViewLibrary.load(paths),
    )


def format_alpha_view_block(resolved: str) -> str:
    """Canonical ``## Alpha view`` block, or empty if there is nothing to show."""
    body = resolved.strip()
    if not body:
        return ""
    lines = body.splitlines()
    if lines and lines[0].lstrip().lower().startswith("## alpha view"):
        body = "\n".join(lines[1:]).strip()
    if not body:
        return ""
    return f"{_ALPHA_VIEW_HEADING}\n\n{body}"


def compose_mission(mission: str, resolved_view: str) -> str:
    """Append the formatted alpha-view block to the rubric. Identity if empty.

    Idempotent if the mission already contains the same block (a pack that
    still inlines Alpha View during migration will not double it).
    """
    block = format_alpha_view_block(resolved_view)
    if not block:
        return mission
    if block.strip() in mission:
        return mission
    if not mission.strip():
        return f"{block}\n"
    return f"{mission.rstrip()}\n\n{block}\n"


def apply_alpha_view(
    mission: str, library: AlphaViewLibrary, decision_date: date
) -> tuple[str, str]:
    """Resolve the library at ``decision_date``.

    Returns ``(composed_mission, alpha_view_block)``. Single-prompt agents
    consume the composed mission; multi-call agents also take the block so
    sub-agents that do not see ``mission.md`` still get the thesis.
    """
    resolved = library.resolve(decision_date)
    block = format_alpha_view_block(resolved)
    return compose_mission(mission, resolved), block


def _read_dated_note(path: Path) -> DatedNote | None:
    match = _NOTE_NAME.match(path.name)
    if not match or not path.is_file():
        return None
    try:
        as_of = date.fromisoformat(match.group(1))
    except ValueError:
        logger.warning("ignoring alpha-view note with invalid date name: %s", path.name)
        return None
    return DatedNote(as_of=as_of, text=path.read_text())


def _library_digest(standing: str, notes: tuple[DatedNote, ...]) -> str | None:
    if not standing.strip() and not notes:
        return None
    parts = [standing]
    for note in notes:
        parts.append(f"{note.as_of.isoformat()}\n{note.text}")
    return _sha("\n---\n".join(parts))
