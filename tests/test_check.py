"""Pack/agent surface inspection: wired vs default vs issues."""

from __future__ import annotations

import json
import textwrap
from argparse import Namespace
from pathlib import Path

from fintel.agents.contract import PACK_CONTEXT_FIELDS, inspect_agent
from fintel.cli.check import _cross_issues, run_check
from fintel.cli.main import build_parser, main
from fintel.strategy.inspect import PACK_FEATURES, inspect_pack


def _write_package(root: Path, *, extra_toml: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.md").write_text("# Mission\nScore the names.\n")
    (root / "output_schema.json").write_text("{}")
    (root / "strategy.toml").write_text(
        textwrap.dedent(
            """
            name = "test_stockrate"
            description = "a test package"

            [universe]
            symbols = ["AAPL"]

            [decision]
            scope = "single_name"
            schedule = { kind = "single_point", on = "2024-01-02" }

            [[data]]
            kind = "prices"
            source = "synthetic_prices"

            [scoring]
            kpi = "icir"
            """
        )
        + extra_toml
    )
    return root


def test_pack_features_list_is_unique_and_covers_alpha_view():
    ids = [f.id for f in PACK_FEATURES]
    assert ids == list(dict.fromkeys(ids))
    assert "mission" in ids
    assert "alpha_view" in ids
    assert "eval" in ids
    assert "ablation" in ids


def test_inspect_pack_reports_optional_features_as_default(tmp_path):
    root = _write_package(tmp_path / "pkg")
    report = inspect_pack(root, env={})
    by_id = {f.id: f for f in report.features}
    assert by_id["mission"].status == "wired"
    assert by_id["output_schema"].status == "wired"
    assert by_id["alpha_view"].status == "default"
    assert by_id["alpha_views"].status == "default"
    assert by_id["company_names"].status == "default"
    assert by_id["eval"].status == "default"
    assert by_id["ablation"].status == "default"
    assert report.has_alpha_view is False
    assert report.problems == ()
    scoring = {f.id: f.status for f in report.scoring_defaults}
    assert scoring["scoring.signal"] == "default"


def test_inspect_pack_wires_alpha_view_and_dated_notes(tmp_path):
    root = _write_package(tmp_path / "pkg")
    (root / "alpha_view.md").write_text("Rate the business, not the narrative.")
    (root / "alpha_views").mkdir()
    (root / "alpha_views" / "2024-01-01.md").write_text("January note.")
    report = inspect_pack(root, env={})
    by_id = {f.id: f for f in report.features}
    assert by_id["alpha_view"].status == "wired"
    assert by_id["alpha_views"].status == "wired"
    assert "2024-01-01" in by_id["alpha_views"].detail
    assert report.has_alpha_view is True
    assert report.alpha_view_notes == ("2024-01-01",)


def test_inspect_agent_constant_accepts_pack_context():
    surface = inspect_agent("constant")
    assert surface.known
    assert surface.pit_enforcement == "access"
    assert surface.subagents is False
    for field in PACK_CONTEXT_FIELDS:
        assert surface.accepts[field]
    assert surface.issues == ()
    assert surface.preflight == ()


def test_inspect_agent_optimized_declares_subagents():
    surface = inspect_agent("optimized")
    assert surface.subagents is True
    assert surface.accepts["alpha_view_text"] is True
    assert surface.accepts["company_names"] is True


def test_build_agent_keeps_mission_when_adapter_omits_company_names():
    """company_names.json must not TypeError-drop mission_text on constant."""
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.simulate.cell import build_agent

    agent = build_agent(
        AgentSpec(name="constant", model=ModelSpec()),
        mission_text="score it",
        alpha_view_text="## Alpha view\n\nRate the business.",
        company_names={"AAPL": "Apple"},
    )
    assert agent.mission_text == "score it"
    assert agent.alpha_view_text == "## Alpha view\n\nRate the business."


def test_inspect_agent_unknown():
    surface = inspect_agent("not-an-agent")
    assert surface.known is False
    assert surface.issues


def test_cross_issues_flag_ablation_on_the_wrong_agent(tmp_path):
    root = _write_package(
        tmp_path / "pkg",
        extra_toml='[ablation]\nsearch_query = "AAPL earnings"',
    )
    pack = inspect_pack(root, env={})
    assert pack.ablation_search_query
    constant = inspect_agent("constant")
    issues = _cross_issues(pack, constant)
    assert any("search_query" in item and "llm" in item for item in issues)
    llm = inspect_agent("llm")
    assert _cross_issues(pack, llm) == []


def test_check_cli_pack_and_constant_agent(tmp_path, capsys):
    root = _write_package(tmp_path / "pkg")
    code = main(["check", str(root), "--agent", "constant", "--no-bootstrap"])
    out = capsys.readouterr().out
    assert code == 0
    assert "test_stockrate" in out
    assert "alpha_view" in out
    assert "constant" in out
    assert "issues    none" in out


def test_check_cli_json(tmp_path, capsys):
    root = _write_package(tmp_path / "pkg")
    args = Namespace(
        package=str(root),
        agent=["constant"],
        all_agents=False,
        json=True,
        no_bootstrap=True,
    )
    code = run_check(args)
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["pack"]["name"] == "test_stockrate"
    assert payload["agents"][0]["name"] == "constant"
    assert payload["issues"] == []


def test_check_parser_accepts_repeatable_agents():
    args = build_parser().parse_args(
        ["check", "packages/demo", "--agent", "optimized", "--agent", "llm"]
    )
    assert args.command == "check"
    assert args.agent == ["optimized", "llm"]


def test_check_missing_package_exits_2(capsys):
    code = main(["check", "/no/such/pack", "--no-bootstrap"])
    assert code == 2
    assert "not found" in capsys.readouterr().out
