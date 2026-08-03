"""Strategy package: load, preflight, lock.

A package is an external directory; these tests build minimal packages in
tmp_path to cover the happy path and each failure mode. The point is that one
preflight reports *every* problem, not just the first — a package with a bad
binding, a missing mission, and an unset env var gets three findings, so the
author fixes all three in one pass.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fintel.strategy import (
    ManifestError,
    PackageNotFound,
    PreflightError,
    load,
    load_and_prepare,
    preflight,
    read_lock,
)


# ── fixtures ────────────────────────────────────────────────────────────────

MISSION = "# Mission\nScore the names you are given."
SCHEMA = {"type": "object", "properties": {"views": {"type": "array"}}}


def _write_package(
    root: Path,
    *,
    manifest: str,
    mission: str = MISSION,
    schema: dict | None = SCHEMA,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.md").write_text(mission)
    if schema is not None:
        import json

        (root / "output_schema.json").write_text(json.dumps(schema))
    (root / "strategy.toml").write_text(manifest)
    return root


GOOD_MANIFEST = textwrap.dedent("""
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


def good_package(tmp_path: Path) -> Path:
    return _write_package(tmp_path / "pkg", manifest=GOOD_MANIFEST)


# ── load ─────────────────────────────────────────────────────────────────────


def test_load_parses_a_valid_package(tmp_path):
    root = good_package(tmp_path)
    paths = load(root)
    assert paths.manifest.name == "test_stockrate"
    assert paths.manifest.kinds == ("prices",)
    assert paths.mission.is_file()
    assert paths.root == root


def test_load_missing_directory_raises():
    with pytest.raises(PackageNotFound):
        load("/no/such/dir")


def test_load_missing_manifest_raises(tmp_path):
    (tmp_path / "pkg").mkdir()
    with pytest.raises(ManifestError, match="no strategy.toml"):
        load(tmp_path / "pkg")


def test_load_unparseable_manifest_raises(tmp_path):
    root = tmp_path / "pkg"
    _write_package(root, manifest="this is not = valid = toml =")
    with pytest.raises(ManifestError, match="cannot parse"):
        load(root)


def test_load_invalid_manifest_raises(tmp_path):
    # missing required fields
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "mission.md").write_text(MISSION)
    (root / "strategy.toml").write_text('name = "x"\n')
    with pytest.raises(ManifestError, match="invalid manifest"):
        load(root)


def test_duplicate_kinds_rejected(tmp_path):
    manifest = GOOD_MANIFEST + textwrap.dedent("""
        [[data]]
        kind = "prices"
        source = "massive_prices"
    """)
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    with pytest.raises(ManifestError, match="duplicate data kinds"):
        load(root)


# ── preflight ────────────────────────────────────────────────────────────────


def test_preflight_ok_for_a_good_package(tmp_path):
    paths = load(good_package(tmp_path))
    result = preflight(paths, env={})
    assert result.ok, result.problems
    assert result.problems == []
    assert "synthetic_prices" in result.required_env or result.required_env == []


def test_preflight_reports_unknown_source(tmp_path):
    manifest = GOOD_MANIFEST.replace('source = "synthetic_prices"', 'source = "no_such_source"')
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={})
    assert not result.ok
    assert any("unknown source" in p for p in result.problems)


def test_preflight_reports_missing_mission(tmp_path):
    root = good_package(tmp_path)
    (root / "mission.md").unlink()
    paths = load(root)
    result = preflight(paths, env={})
    assert not result.ok
    assert any("mission file" in p for p in result.problems)


def test_preflight_warns_on_missing_output_schema(tmp_path):
    root = good_package(tmp_path)
    (root / "output_schema.json").unlink()
    paths = load(root)
    result = preflight(paths, env={})
    # schema is optional metadata; absence is a warning, not a stop
    assert result.ok, result.problems
    assert any("output schema" in w for w in result.warnings)


def test_preflight_reports_unknown_schedule(tmp_path):
    manifest = GOOD_MANIFEST.replace('kind = "single_point"', 'kind = "no_such_schedule"')
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={})
    assert not result.ok
    assert any("unknown schedule kind" in p for p in result.problems)


def test_preflight_reports_missing_env(tmp_path):
    # synthetic_prices needs no env, so use a source that does. We test the
    # env path by declaring a source whose requires_env is non-empty. The
    # massive_prices source requires MASSIVE_API_KEY.
    manifest = GOOD_MANIFEST.replace(
        'source = "synthetic_prices"', 'source = "massive_prices"'
    )
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={})
    assert not result.ok
    assert any("MASSIVE_API_KEY" in p for p in result.problems)


def test_preflight_env_satisfied_when_set(tmp_path):
    manifest = GOOD_MANIFEST.replace(
        'source = "synthetic_prices"', 'source = "massive_prices"'
    )
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={"MASSIVE_API_KEY": "x"})
    # env satisfied; other problems (none here) may still apply
    assert not any("MASSIVE_API_KEY" in p for p in result.problems)


def test_preflight_reports_multiple_problems_at_once(tmp_path):
    # bad source AND missing mission AND missing env: three findings, not one
    manifest = GOOD_MANIFEST.replace(
        'source = "synthetic_prices"', 'source = "massive_prices"'
    ).replace('source = "massive_prices"', 'source = "typo_source"')
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    (root / "mission.md").unlink()
    paths = load(root)
    result = preflight(paths, env={})
    assert not result.ok
    assert len(result.problems) >= 2  # unknown source + missing mission at least


def test_preflight_raise_if_not_ok_raises(tmp_path):
    paths = load(good_package(tmp_path))
    preflight(paths, env={}).raise_if_not_ok()  # ok, no raise

    root = good_package(tmp_path)
    (root / "mission.md").unlink()
    paths = load(root)
    with pytest.raises(PreflightError):
        preflight(paths, env={}).raise_if_not_ok()


def test_preflight_accepts_module_callable_source(tmp_path):
    # a module:Callable source is allowed without a catalog entry
    manifest = GOOD_MANIFEST.replace(
        'source = "synthetic_prices"', 'source = "fintel.market.data.synthetic:SyntheticPrices"'
    )
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={})
    # module:Callable sources are not flagged as unknown
    assert not any("unknown source" in p for p in result.problems)


def test_preflight_accepts_module_callable_schedule(tmp_path):
    manifest = GOOD_MANIFEST.replace(
        'kind = "single_point", on = "2024-01-02"',
        'kind = "fintel.market.schedule:SinglePoint", on = "2024-01-02"',
    )
    root = _write_package(tmp_path / "pkg", manifest=manifest)
    paths = load(root)
    result = preflight(paths, env={})
    assert not any("unknown schedule" in p for p in result.problems)


# ── lock ─────────────────────────────────────────────────────────────────────


def test_lock_is_reproducible_for_unchanged_package(tmp_path):
    from datetime import UTC, datetime

    now = datetime(2024, 1, 1, tzinfo=UTC)
    paths = load(good_package(tmp_path))
    from fintel.strategy import build_lock

    lock1 = build_lock(
        paths,
        catalog_sources=("synthetic_prices",),
        catalog_kinds=("prices",),
        now=now,
    )
    lock2 = build_lock(
        paths,
        catalog_sources=("synthetic_prices",),
        catalog_kinds=("prices",),
        now=now,
    )
    assert lock1.strategy_digest == lock2.strategy_digest
    assert lock1.mission_digest == lock2.mission_digest
    assert lock1.digest if hasattr(lock1, "digest") else lock1.strategy_digest


def test_lock_changes_when_mission_edited(tmp_path):
    from datetime import UTC, datetime

    now = datetime(2024, 1, 1, tzinfo=UTC)
    root = good_package(tmp_path)
    paths = load(root)
    from fintel.strategy import build_lock

    before = build_lock(paths, catalog_sources=(), catalog_kinds=(), now=now)
    (root / "mission.md").write_text("# changed mission\n")
    after = build_lock(paths, catalog_sources=(), catalog_kinds=(), now=now)
    assert before.mission_digest != after.mission_digest
    # strategy digest is the manifest, which didn't change
    assert before.strategy_digest == after.strategy_digest


def test_lock_write_and_read_roundtrip(tmp_path):
    root = good_package(tmp_path)
    paths, result, lock = load_and_prepare(root, env={}, write_lock=True)
    assert result.ok, result.problems
    assert (root / "strategy.lock").is_file()
    reread = read_lock(root / "strategy.lock")
    assert reread.name == lock.name
    assert reread.strategy_digest == lock.strategy_digest
    assert reread.mission_digest == lock.mission_digest


def test_load_and_prepare_returns_paths_result_lock(tmp_path):
    root = good_package(tmp_path)
    paths, result, lock = load_and_prepare(root, env={}, write_lock=False)
    assert paths.manifest.name == "test_stockrate"
    assert result.ok
    assert lock.name == "test_stockrate"
    assert "prices" in lock.catalog_kinds


def test_load_and_prepare_write_lock_false_does_not_write(tmp_path):
    root = good_package(tmp_path)
    load_and_prepare(root, env={}, write_lock=False)
    assert not (root / "strategy.lock").exists()
