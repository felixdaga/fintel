"""CLI surface: entry point parses and dispatches without running a job."""

from __future__ import annotations

from fintel.cli.main import build_parser, main


def test_simulation_parser_accepts_openclaw_shape():
    parser = build_parser()
    args = parser.parse_args(
        [
            "simulation",
            "packages/systematic_stockrate_djia_weekly",
            "--agent",
            "openclaw",
            "--agent-opt",
            "profile=default",
            "--agent-opt",
            "agent_id=main",
            "--universe",
            "AAPL,MSFT",
            "--dates",
            "2025-01-02",
            "--cell-concurrency",
            "1",
            "--job-id",
            "cli-test",
        ]
    )
    assert args.command == "simulation"
    assert args.agent == "openclaw"
    assert args.agent_opt == ["profile=default", "agent_id=main"]
    assert args.universe == "AAPL,MSFT"
    assert args.dates == "2025-01-02"


def test_main_missing_command_exits():
    import pytest

    with pytest.raises(SystemExit):
        main([])
