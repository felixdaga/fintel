"""Alpha view: strategy-owned thesis, PIT dated notes, compose into prompts."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

from fintel.strategy import build_lock, load, load_and_prepare, read_lock
from fintel.strategy.views import (
    AlphaViewLibrary,
    apply_alpha_view,
    compose_mission,
    format_alpha_view_block,
    load_pack_context,
)

MISSION = "# Mission\nScore the names you are given."

_GOOD_MANIFEST = textwrap.dedent("""
    name = "test_stockrate"
    description = "a test package"

    [universe]
    symbols = ["AAPL", "MSFT"]

    [decision]
    scope = "single_name"
    schedule = { kind = "single_point", on = "2024-01-02" }

    [[data]]
    kind = "prices"
    source = "synthetic_prices"

    [scoring]
    kpi = "icir"
    horizons = [1, 2, 3]
""")


def _write_package(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.md").write_text(MISSION)
    (root / "output_schema.json").write_text("{}")
    (root / "strategy.toml").write_text(_GOOD_MANIFEST)
    return root


def _pkg_with_views(
    tmp_path: Path,
    *,
    standing: str | None = "Rate the business, not the narrative.",
    notes: dict[str, str] | None = None,
) -> Path:
    root = _write_package(tmp_path / "pkg")
    if standing is not None:
        (root / "alpha_view.md").write_text(standing)
    if notes:
        views_dir = root / "alpha_views"
        views_dir.mkdir()
        for name, body in notes.items():
            (views_dir / name).write_text(body)
    return root


def test_missing_alpha_view_is_empty_not_an_error(tmp_path):
    root = _write_package(tmp_path / "pkg")
    library = AlphaViewLibrary.load(load(root))
    assert library.standing == ""
    assert library.notes == ()
    assert library.digest is None
    assert library.resolve(date(2024, 1, 2)) == ""
    composed, block = apply_alpha_view(MISSION, library, date(2024, 1, 2))
    assert composed == MISSION
    assert block == ""


def test_standing_view_is_formatted_and_composed(tmp_path):
    root = _pkg_with_views(tmp_path)
    library = AlphaViewLibrary.load(load(root))
    composed, block = apply_alpha_view(MISSION, library, date(2024, 1, 2))
    assert block.startswith("## Alpha view\n")
    assert "Rate the business, not the narrative." in block
    assert composed.endswith(block + "\n") or block in composed
    assert MISSION.strip() in composed
    assert composed.index(MISSION.strip()) < composed.index("## Alpha view")


def test_dated_note_is_pit_clipped(tmp_path):
    root = _pkg_with_views(
        tmp_path,
        notes={
            "2024-01-01.md": "January note.",
            "2024-06-01.md": "June note must not leak.",
        },
    )
    library = AlphaViewLibrary.load(load(root))
    assert [n.as_of.isoformat() for n in library.notes] == ["2024-01-01", "2024-06-01"]

    jan = library.resolve(date(2024, 1, 2))
    assert "January note." in jan
    assert "June note" not in jan
    assert "### Research note (2024-01-01)" in jan

    before = library.resolve(date(2023, 12, 31))
    assert "January note" not in before
    assert "Rate the business" in before

    after = library.resolve(date(2024, 6, 1))
    assert "June note must not leak." in after
    assert "January note" not in after


def test_same_day_note_is_visible(tmp_path):
    root = _pkg_with_views(tmp_path, notes={"2024-01-02.md": "Decision-day note."})
    library = AlphaViewLibrary.load(load(root))
    assert "Decision-day note." in library.resolve(date(2024, 1, 2))


def test_readme_and_invalid_dates_in_views_dir_are_ignored(tmp_path):
    root = _pkg_with_views(
        tmp_path,
        notes={
            "README.md": "not a note",
            "2024-13-40.md": "impossible date",
            "note.txt": "wrong suffix",
            "2024-03-01.md": "March note.",
        },
    )
    library = AlphaViewLibrary.load(load(root))
    assert [n.as_of.isoformat() for n in library.notes] == ["2024-03-01"]
    assert "March note." in library.resolve(date(2024, 3, 15))
    assert "not a note" not in library.resolve(date(2024, 3, 15))


def test_compose_is_idempotent_when_mission_already_has_the_block():
    view = "Rate the business, not the narrative."
    block = format_alpha_view_block(view)
    mission = f"{MISSION}\n\n{block}\n"
    assert compose_mission(mission, view) == mission


def test_format_strips_a_leading_heading():
    block = format_alpha_view_block("## Alpha view\n\nAlready headed.")
    assert block.startswith("## Alpha view\n\nAlready headed.")
    assert block.count("## Alpha view") == 1


def test_lock_digest_covers_standing_and_dated_notes(tmp_path):
    from datetime import UTC, datetime

    now = datetime(2024, 1, 1, tzinfo=UTC)
    root = _pkg_with_views(tmp_path, notes={"2024-01-01.md": "January note."})
    paths = load(root)
    before = build_lock(paths, catalog_sources=(), catalog_kinds=(), now=now)
    assert before.alpha_view_digest is not None

    (root / "alpha_view.md").write_text("Edited standing thesis.")
    after_standing = build_lock(paths, catalog_sources=(), catalog_kinds=(), now=now)
    assert after_standing.alpha_view_digest != before.alpha_view_digest
    assert after_standing.mission_digest == before.mission_digest

    (root / "alpha_views" / "2024-01-01.md").write_text("Edited January note.")
    after_note = build_lock(paths, catalog_sources=(), catalog_kinds=(), now=now)
    assert after_note.alpha_view_digest != after_standing.alpha_view_digest


def test_lock_roundtrip_includes_alpha_view_digest(tmp_path):
    root = _pkg_with_views(tmp_path)
    paths, result, lock = load_and_prepare(root, env={}, write_lock=True)
    assert result.ok, result.problems
    assert lock.alpha_view_digest is not None
    reread = read_lock(root / "strategy.lock")
    assert reread.alpha_view_digest == lock.alpha_view_digest


def test_old_lock_without_alpha_view_digest_still_reads(tmp_path):
    import json

    path = tmp_path / "strategy.lock"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "legacy",
                "strategy_digest": "abc",
                "mission_digest": "def",
                "output_schema_digest": None,
                "catalog_sources": [],
                "catalog_kinds": [],
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        )
    )
    lock = read_lock(path)
    assert lock.alpha_view_digest is None


def test_load_pack_context_carries_the_library(tmp_path):
    root = _pkg_with_views(tmp_path)
    pack = load_pack_context(load(root))
    assert pack.mission_text == MISSION
    assert "Rate the business" in pack.alpha_views.standing
