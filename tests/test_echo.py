"""The run echo — every input gathered before any cell runs.

Pins that `run_job` emits a `run_echo` nerve event (terminal + run.log) carrying
the full input surface summary (agent, universe, schedule, data, tools with
their *exact* schemas, the injected prompt, the PIT policy, the fingerprint),
and that the terminal renderer is scannable. The fingerprint digest is sealed
into `r1/config.json` (not a sibling file). The echo is the 'never go blind'
signal: if a run misbehaves, the log echo + config tell you what world the
agent was acting on.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fintel.environment.echo import build_echo, render_echo
from fintel.environment.nerve import Nerve


def test_run_job_emits_echo_and_seals_fingerprint_in_config(tmp_path):
    from fintel.market.settings import MarketConfig
    from fintel.simulate.job import run_job

    package = _write_package(tmp_path / "pkg")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")
    run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))

    run_root = tmp_path / "runs" / "test-job-001" / "r1"
    assert not (run_root / "echo.json").exists()
    assert not (run_root / "lock.json").exists()
    assert not (run_root / "fingerprint.json").exists()

    cfg = json.loads((run_root / "config.json").read_text())
    assert cfg["agent"]["name"] == "constant"
    assert cfg["fingerprint"]["digest"]
    assert cfg["fingerprint"]["agent_name"] == "constant"
    assert cfg["fingerprint"]["data_kinds"] == ["prices"]

    log = [json.loads(l) for l in (run_root / "run.log").read_text().splitlines() if l.strip()]
    echo_ev = next(e for e in log if e.get("event") == "run_echo")
    block = echo_ev["echo"]
    assert "== run echo" in block
    assert "constant" in block
    assert "get_prices" in block
    assert "fingerprint:" in block


def test_echo_records_package_supplied_source_without_catalog_entry(tmp_path):
    """`build_echo` directly: one tool entry per bound kind, never dropped, and
    the composed instruction names the decision date and universe."""
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
    from fintel.models.run import RunConfig, StrategyRef
    from fintel.models.strategy import ScoringSpec

    rc = RunConfig(
        run_id="r1",
        job_id="j1",
        k_index=1,
        k_repeats=1,
        created_at="2025-01-01T00:00:00Z",
        strategy=StrategyRef(name="s", path="p"),
        agent=AgentSpec(name="constant", model=ModelSpec()),
        scope="single_name",
        universe=UniverseRef(symbols=["AAPL"]),
        universe_symbols=["AAPL"],
        schedule=ScheduleRef(kind="single_point", on="2025-01-02"),
        schedule_dates=["2025-01-02"],
        data=[DataBinding(kind="prices", source="synthetic_prices")],
        scoring=ScoringSpec(kpi="direction"),
    )
    echo = build_echo(
        run_config=rc,
        strategy_description="desc",
        sources={},
        mission_text="m",
        output_schema_text="{}",
        fingerprint={"adapter_params": {"pit_enforcement": "access", "pit_deny": []}, "digest": "abc"},
    )
    assert echo["tools"][0]["kind"] == "prices"
    assert "Decision date:" in echo["prompt"]["composed_instruction"]


def test_render_echo_is_scannable_and_complete(tmp_path):
    """The terminal block shows agent, universe, schedule, data, tools (with
    params), prompt lengths, and the fingerprint — compact but complete."""
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
    from fintel.models.run import RunConfig, StrategyRef
    from fintel.models.strategy import ScoringSpec

    rc = RunConfig(
        run_id="r1",
        job_id="j1",
        k_index=1,
        k_repeats=1,
        created_at="2025-01-01T00:00:00Z",
        strategy=StrategyRef(name="s", path="p"),
        agent=AgentSpec(name="constant", model=ModelSpec()),
        scope="single_name",
        universe=UniverseRef(symbols=["AAPL"]),
        universe_symbols=["AAPL"],
        schedule=ScheduleRef(kind="single_point", on="2025-01-02"),
        schedule_dates=["2025-01-02"],
        data=[DataBinding(kind="prices", source="synthetic_prices")],
        scoring=ScoringSpec(kpi="direction"),
    )
    echo = build_echo(
        run_config=rc,
        strategy_description="desc",
        sources={},
        mission_text="mission text",
        output_schema_text='{"$schema": "x"}',
        fingerprint={"adapter_params": {"pit_enforcement": "access", "pit_deny": []}, "digest": "0123456789abcdef"},
    )
    block = render_echo(echo)
    assert "== run echo" in block
    assert "agent:" in block and "constant" in block
    assert "universe:" in block and "AAPL" in block
    assert "schedule:" in block and "2025-01-02" in block
    assert "tools:" in block and "get_prices" in block
    assert "prompt:" in block and "mission=" in block
    assert "fingerprint:" in block


def test_nerve_emits_run_echo_block_to_terminal_and_log(tmp_path):
    """`run_echo` carries a pre-rendered multi-line block; Nerve prints it
    verbatim to the terminal and records the event in run.log."""
    out = io.StringIO()
    nerve = Nerve(run_root=tmp_path, stream=out, verbose=True)
    block = "== run echo\n   agent: constant"
    nerve.emit("run_echo", echo=block)
    assert "== run echo" in out.getvalue()
    log = [json.loads(l) for l in (tmp_path / "run.log").read_text().splitlines() if l.strip()]
    assert log[0]["event"] == "run_echo"
    assert log[0]["echo"] == block


# ── fixtures ─────────────────────────────────────────────────────────────────
# Mirrors the helpers in test_simulate.py so this test is standalone.


def _write_package(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.md").write_text("mission text\n")
    (root / "output_schema.json").write_text('{"$schema": "https://example.com"}\n')
    (root / "strategy.toml").write_text(
        'name = "test"\n'
        'description = "test strategy"\n'
        "\n"
        "[universe]\nsymbols = [\"AAPL\"]\n"
        "\n"
        "[decision]\nscope = \"single_name\"\n"
        "schedule = { kind = \"single_point\", on = \"2025-01-02\" }\n"
        "\n"
        "[[data]]\nkind = \"prices\"\nsource = \"synthetic_prices\"\n"
        "\n"
        "[scoring]\nkpi = \"direction\"\n"
    )
    return root


def _job_config(package: Path, *, agent_name: str = "constant", k_repeats: int = 1):
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.job import JobConfig

    return JobConfig(
        job_id="test-job-001",
        strategy=str(package),
        agent=AgentSpec(name=agent_name, model=ModelSpec()),
        k_repeats=k_repeats,
        max_concurrent=1,
        output_root="",
    )
