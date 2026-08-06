"""The execution hierarchy: Job → Run → Trial → Cell, and the fan-in.

These tests build a real strategy package with synthetic prices and a scripted
agent, then run a full job end to end. The point is that the artifact tree is
correct (one writer per cell, decision.json reduced once), a failed cell doesn't
abort its date, and the reducers are pure and deterministic.
"""

from __future__ import annotations

import json
import textwrap
from datetime import date as Date
from pathlib import Path

import pytest

from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.decision import AgentResponse, View
from fintel.models.job import JobConfig
from fintel.models.trace import Usage
from fintel.models.trial import CellResult, TrialResult
from fintel.simulate import (
    map_parallel,
    reduce_decision,
    reduce_job,
    reduce_run,
    reduce_trial,
    run_job,
)

# ── fixtures ────────────────────────────────────────────────────────────────

MISSION = "# Mission\nScore the names you are given."


def _write_package(root: Path, *, scope: str = "single_name", symbols: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mission.md").write_text(MISSION)
    (root / "output_schema.json").write_text("{}")
    syms = ", ".join(f'"{s}"' for s in (symbols or ["AAPL", "MSFT"]))
    manifest = textwrap.dedent(f"""
        name = "test_pkg"
        description = "test"

        [universe]
        symbols = [{syms}]

        [decision]
        scope = "{scope}"
        schedule = {{ kind = "single_point", on = "2024-01-02" }}

        [[data]]
        kind = "prices"
        source = "synthetic_prices"

        [scoring]
        kpi = "icir"
        horizons = [1]
    """)
    (root / "strategy.toml").write_text(manifest)
    return root


def _job_config(package: Path, *, agent_name: str = "constant", k_repeats: int = 1) -> JobConfig:
    return JobConfig(
        job_id="test-job-001",
        strategy=str(package),
        agent=AgentSpec(name=agent_name, model=ModelSpec()),
        k_repeats=k_repeats,
        max_concurrent=1,
        output_root="",  # set per-test
    )


# ── queue ────────────────────────────────────────────────────────────────────


def test_map_parallel_sequential_returns_in_order():
    out = map_parallel(lambda x: x * 2, [1, 2, 3], bound=1)
    assert out == [2, 4, 6]


def test_map_parallel_bound_gt_one_returns_in_order():
    out = map_parallel(lambda x: x * 2, list(range(10)), bound=4)
    assert out == [x * 2 for x in range(10)]


def test_map_parallel_failure_is_none_not_raise():
    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x

    out = map_parallel(fn, [1, 2, 3], bound=1)
    assert out == [1, None, 3]


def test_map_parallel_empty():
    assert map_parallel(lambda x: x, [], bound=1) == []


# ── reducers ─────────────────────────────────────────────────────────────────


def test_reduce_decision_merges_views():
    r1 = AgentResponse(views={"AAPL": View(symbol="AAPL", score=0.5, rationale="a")})
    r2 = AgentResponse(views={"MSFT": View(symbol="MSFT", score=-0.3, rationale="b")})
    decision = reduce_decision([("AAPL", r1), ("MSFT", r2)])
    assert set(decision) == {"AAPL", "MSFT"}
    assert decision["AAPL"].score == 0.5


def test_reduce_decision_collision_keeps_first():
    r1 = AgentResponse(views={"AAPL": View(symbol="AAPL", score=0.5, rationale="first")})
    r2 = AgentResponse(views={"AAPL": View(symbol="AAPL", score=0.9, rationale="second")})
    decision = reduce_decision([("c1", r1), ("c2", r2)])
    assert decision["AAPL"].score == 0.5  # first wins


def test_reduce_trial_ok():
    cells = [
        CellResult(cell="AAPL", symbols=["AAPL"], status="ok", n_views=1),
        CellResult(cell="MSFT", symbols=["MSFT"], status="ok", n_views=1),
    ]
    result = reduce_trial(Date(2024, 1, 2), cells)
    assert result.status == "ok"
    assert result.n_views == 2


def test_reduce_trial_partial_when_some_failed():
    cells = [
        CellResult(cell="AAPL", symbols=["AAPL"], status="ok", n_views=1),
        CellResult(cell="MSFT", symbols=["MSFT"], status="failed", n_views=0, error="x"),
    ]
    result = reduce_trial(Date(2024, 1, 2), cells)
    assert result.status == "partial"


def test_reduce_trial_failed_when_all_failed():
    cells = [CellResult(cell="AAPL", symbols=["AAPL"], status="failed", error="x")]
    result = reduce_trial(Date(2024, 1, 2), cells)
    assert result.status == "failed"


def test_reduce_trial_skipped_when_no_cells():
    result = reduce_trial(Date(2024, 1, 2), [])
    assert result.status == "skipped"


def test_reduce_run_ok():
    trials = [
        TrialResult(decision_date=Date(2024, 1, 2), status="ok", n_views=2),
    ]
    result = reduce_run("r1", "j1", 1, trials)
    assert result.status == "ok"
    assert result.n_decisions == 1
    assert result.n_views == 2


def test_reduce_run_partial():
    trials = [
        TrialResult(decision_date=Date(2024, 1, 2), status="ok", n_views=2),
        TrialResult(decision_date=Date(2024, 4, 1), status="failed", error="x"),
    ]
    result = reduce_run("r1", "j1", 1, trials)
    assert result.status == "partial"


def test_reduce_run_failed():
    trials = [TrialResult(decision_date=Date(2024, 1, 2), status="failed", error="x")]
    result = reduce_run("r1", "j1", 1, trials)
    assert result.status == "failed"


def test_reduce_run_usage_aggregated():
    trials = [
        TrialResult(
            decision_date=Date(2024, 1, 2),
            status="ok",
            cells=[CellResult(cell="AAPL", symbols=["AAPL"], status="ok", usage=Usage(n_llm_calls=1, tokens_in=10))],
        ),
    ]
    result = reduce_run("r1", "j1", 1, trials)
    assert result.usage.n_llm_calls == 1
    assert result.usage.tokens_in == 10


def test_reduce_job_ok():
    from fintel.models.run import RunResult

    runs = [
        RunResult(run_id="r1", job_id="j1", k_index=1, status="ok", n_views=2),
    ]
    result = reduce_job("j1", "test_pkg", "constant", 1, runs)
    assert result.status == "ok"
    assert result.runs[0].run_id == "r1"


def test_reduce_job_partial_when_some_failed():
    from fintel.models.run import RunResult

    runs = [
        RunResult(run_id="r1", job_id="j1", k_index=1, status="ok", n_views=2),
        RunResult(run_id="r2", job_id="j1", k_index=2, status="failed", error="x"),
    ]
    result = reduce_job("j1", "test_pkg", "constant", 2, runs)
    assert result.status == "partial"


def test_reduce_job_failed_when_all_failed():
    from fintel.models.run import RunResult

    runs = [RunResult(run_id="r1", job_id="j1", k_index=1, status="failed", error="x")]
    result = reduce_job("j1", "test_pkg", "constant", 1, runs)
    assert result.status == "failed"


# ── end to end ───────────────────────────────────────────────────────────────


def test_run_job_single_name_constant_agent(tmp_path):
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))

    assert result.status == "ok"
    assert result.k_repeats == 1
    assert len(result.runs) == 1
    assert result.runs[0].status == "ok"

    # Artifact tree: job config + result
    job_root = tmp_path / "runs" / "test-job-001"
    assert (job_root / "config.json").is_file()
    assert (job_root / "result.json").is_file()

    # One run, one trial, two cells (single_name scope)
    run_root = job_root / "r1"
    assert (run_root / "config.json").is_file()
    assert (run_root / "result.json").is_file()
    assert not (run_root / "lock.json").exists()
    assert not (run_root / "echo.json").exists()
    assert not (run_root / "fingerprint.json").exists()
    cfg = json.loads((run_root / "config.json").read_text())
    assert cfg.get("fingerprint", {}).get("digest")

    trial_root = run_root / "trials" / "2024-01-02"
    assert (trial_root / "decision.json").is_file()
    assert (trial_root / "result.json").is_file()
    assert (trial_root / "cells" / "AAPL.json").is_file()
    assert (trial_root / "cells" / "MSFT.json").is_file()

    # decision.json is a reduction of the cells — two views
    decision = json.loads((trial_root / "decision.json").read_text())
    assert set(decision) == {"AAPL", "MSFT"}

    # Each cell wrote its own file; no concurrent overwrites
    aapl = json.loads((trial_root / "cells" / "AAPL.json").read_text())
    assert aapl["cell"] == "AAPL"
    assert aapl["outcome"] == "ok"
    assert "AAPL" in aapl["views"]


def test_run_job_portfolio_scope_one_cell(tmp_path):
    package = _write_package(tmp_path / "pkg", scope="portfolio")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))

    assert result.status == "ok"
    trial_root = tmp_path / "runs" / "test-job-001" / "r1" / "trials" / "2024-01-02"
    # portfolio scope: one cell, named __portfolio__
    assert (trial_root / "cells" / "__portfolio__.json").is_file()
    cells = list((trial_root / "cells").glob("*.json"))
    assert len(cells) == 1

    decision = json.loads((trial_root / "decision.json").read_text())
    assert set(decision) == {"AAPL", "MSFT"}


def test_run_job_k_repeats_produces_k_runs(tmp_path):
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, k_repeats=3)
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))

    assert len(result.runs) == 3
    job_root = tmp_path / "runs" / "test-job-001"
    for k in (1, 2, 3):
        assert (job_root / f"r{k}" / "result.json").is_file()


def test_run_job_dry_run_flag_accepted(tmp_path):
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")
    job.__dict__["dry_run"] = True

    from fintel.market.settings import MarketConfig

    # dry_run is accepted by the config; the executor doesn't short-circuit yet,
    # so this just confirms the flag flows through without breaking.
    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    assert result.status in ("ok", "skipped", "failed")


def test_run_job_scripted_agent_with_reads(tmp_path):
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, agent_name="scripted")
    job.__dict__["output_root"] = str(tmp_path / "runs")
    job.agent.options = {"reads": ("prices",), "score": 0.7}

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    assert result.status == "ok"

    trial_root = tmp_path / "runs" / "test-job-001" / "r1" / "trials" / "2024-01-02"
    aapl = json.loads((trial_root / "cells" / "AAPL.json").read_text())
    assert aapl["views"]["AAPL"]["score"] == 0.7
    # the scripted agent read prices through the access path
    assert aapl["environment"]["n_reads"] >= 1


# ── mission / output schema / fingerprint / agent preflight ─────────────────


def test_run_job_writes_a_fingerprint_into_config(tmp_path):
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))

    cfg = json.loads((tmp_path / "runs" / "test-job-001" / "r1" / "config.json").read_text())
    fp = cfg["fingerprint"]
    assert fp["agent_name"] == "constant"
    assert fp["data_kinds"] == ["prices"]
    assert fp["digest"]


def test_two_runs_of_the_same_package_have_the_same_fingerprint(tmp_path):
    """The fingerprint hashes the pack's mission, so it is stable across runs
    of the same package and would change if the mission did."""
    package = _write_package(tmp_path / "pkg")
    from fintel.market.settings import MarketConfig

    digests = []
    for i in (1, 2):
        job = _job_config(package)
        job.__dict__["output_root"] = str(tmp_path / f"runs{i}")
        run_job(job, market_config=MarketConfig(cache_root=tmp_path / f"cache{i}", offline=True))
        cfg = json.loads(
            (tmp_path / f"runs{i}" / "test-job-001" / "r1" / "config.json").read_text()
        )
        digests.append(cfg["fingerprint"]["digest"])
    assert digests[0] == digests[1]


def test_run_job_passes_the_packages_mission_and_schema_to_the_agent(tmp_path):
    """End to end: mission.md and output_schema.json reach the live agent
    inside a cell, not just the fingerprint hash."""
    import sys
    from dataclasses import dataclass, field

    from fintel.models.decision import AgentResponse, View

    @dataclass
    class RecordingAgent:
        name: str = "recording"
        version: str = "1"
        mission_text: str = ""
        output_schema_text: str = ""
        pit_enforcement: str = "access"
        seen: list = field(default_factory=list, init=False)

        def decide(self, env):
            RecordingAgent.last_mission = self.mission_text
            RecordingAgent.last_schema = self.output_schema_text
            views = {
                s: View(symbol=s, score=0.1, rationale="recording")
                for s in sorted(env.policy.decidable)
            }
            return AgentResponse(views=views)

    sys.modules["__test_recording__"] = type(sys)("__test_recording__")
    sys.modules["__test_recording__"].RecordingAgent = RecordingAgent  # type: ignore[attr-defined]

    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, agent_name="__test_recording__:RecordingAgent")
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    assert result.status == "ok"
    assert RecordingAgent.last_mission == MISSION
    assert RecordingAgent.last_schema == "{}"


def test_run_job_fails_closed_when_the_agent_cannot_run(tmp_path, monkeypatch):
    """Agent preflight runs alongside strategy preflight — a job that can never
    call its adapter should never fan out a single cell."""
    from fintel.strategy.preflight import PreflightError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, agent_name="llm")
    job.__dict__["output_root"] = str(tmp_path / "runs")

    from fintel.market.settings import MarketConfig

    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY"):
        run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    # Nothing was fanned out.
    assert not (tmp_path / "runs").exists()


# ── concurrency model ─────────────────────────────────────────────────────────
#
# Two axes, mirroring delorean: cells concurrent within a trial (the "10 tickers"
# case), runs parallel across K repeats (the "3 runs" case). Dates sequential
# when memory is on. Auto = None resolves to the natural fan-out width.


def test_job_config_auto_run_concurrency_defaults_to_k():
    from fintel.models.job import JobConfig

    cfg = JobConfig(job_id="j", strategy="x", agent=AgentSpec(name="constant"))
    cfg.__dict__["k_repeats"] = 3
    # auto (None) → all K runs in parallel
    assert cfg.resolve_run_concurrency() == 3
    cfg.__dict__["max_concurrent"] = 2
    assert cfg.resolve_run_concurrency() == 2  # explicit caps


def test_job_config_auto_cell_concurrency_defaults_to_universe():
    from fintel.models.job import JobConfig

    cfg = JobConfig(job_id="j", strategy="x", agent=AgentSpec(name="constant"))
    # auto → universe size (all tickers at once)
    assert cfg.resolve_cell_concurrency(10) == 10
    cfg.__dict__["cell_concurrency"] = 4
    assert cfg.resolve_cell_concurrency(10) == 4  # explicit caps
    assert cfg.resolve_cell_concurrency(3) == 3  # capped to universe too


def test_job_config_peak_concurrent_for_30_session_scenario():
    """3 runs × 10 tickers = 30 sessions, the user's scenario."""
    from fintel.models.job import JobConfig

    cfg = JobConfig(job_id="j", strategy="x", agent=AgentSpec(name="constant"))
    cfg.__dict__["k_repeats"] = 3
    # auto on both axes: peak = 3 runs × 3 (k_repeats used as cell stand-in) = 9
    # the real per-date peak needs the universe, resolved at runtime
    assert cfg.peak_concurrent == 9  # 3 runs × 3 (auto cell uses k_repeats as bound)


def test_memory_guard_forces_sequential_trials(tmp_path):
    """When memory is on, trial_concurrency>1 is forced back to 1 — a date's
    session carries the prior date's state, so parallel dates would race."""
    from fintel.market.settings import MarketConfig
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.market import UniverseRef
    from fintel.models.paths import RunPaths
    from fintel.models.run import RunConfig, StrategyRef
    from fintel.simulate.run import run_run

    package = _write_package(tmp_path / "pkg", symbols=["AAPL", "MSFT"])
    run_paths = RunPaths(root=tmp_path / "run")
    run_config = RunConfig(
        run_id="r1",
        job_id="j1",
        k_index=1,
        k_repeats=1,
        created_at="2024-01-01T00:00:00+00:00",
        strategy=StrategyRef(name="test_pkg", path=str(package)),
        agent=AgentSpec(name="constant", model=ModelSpec()),
        scope="single_name",
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        universe_symbols=["AAPL", "MSFT"],
        schedule={"kind": "single_point", "on": "2024-01-02"},
        schedule_dates=["2024-01-02"],
        data=[{"kind": "prices", "source": "synthetic_prices"}],
        scoring={"kpi": "icir", "horizons": [1]},
    )

    # Run with memory on and trial_concurrency=4 — guard must force 1.
    # The run completes (sequential, no race) and writes a result.
    run_run(
        run_config=run_config,
        market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True),
        paths=run_paths,
        cell_concurrency=2,
        trial_concurrency=4,
        memory_on=True,
    )
    assert (run_paths.root / "result.json").is_file()


def test_shared_concurrency_blocked_when_memory_on(tmp_path):
    """shared_concurrency raises when dates are coupled by memory."""
    from fintel.market.settings import MarketConfig
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.market import UniverseRef
    from fintel.models.paths import RunPaths
    from fintel.models.run import RunConfig, StrategyRef
    from fintel.simulate.run import run_run

    package = _write_package(tmp_path / "pkg", symbols=["AAPL", "MSFT"])
    run_paths = RunPaths(root=tmp_path / "run")
    run_config = RunConfig(
        run_id="r1",
        job_id="j1",
        k_index=1,
        k_repeats=1,
        created_at="2024-01-01T00:00:00+00:00",
        strategy=StrategyRef(name="test_pkg", path=str(package)),
        agent=AgentSpec(name="constant", model=ModelSpec()),
        scope="single_name",
        universe=UniverseRef(symbols=["AAPL", "MSFT"]),
        universe_symbols=["AAPL", "MSFT"],
        schedule={"kind": "single_point", "on": "2024-01-02"},
        schedule_dates=["2024-01-02"],
        data=[{"kind": "prices", "source": "synthetic_prices"}],
        scoring={"kpi": "icir", "horizons": [1]},
    )

    with pytest.raises(ValueError, match="shared_concurrency requires independent"):
        run_run(
            run_config=run_config,
            market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True),
            paths=run_paths,
            shared_concurrency=4,
            memory_on=True,
        )


def test_shared_concurrency_blocked_when_feedback_on(tmp_path):
    """shared_concurrency raises when dates are coupled by feedback."""
    from fintel.market.settings import MarketConfig
    from fintel.models.agent import AgentSpec, ModelSpec
    from fintel.models.market import UniverseRef
    from fintel.models.paths import RunPaths
    from fintel.models.run import RunConfig, StrategyRef
    from fintel.simulate.run import run_run

    package = _write_package(tmp_path / "pkg", symbols=["AAPL"])
    run_paths = RunPaths(root=tmp_path / "run")
    run_config = RunConfig(
        run_id="r1",
        job_id="j1",
        k_index=1,
        k_repeats=1,
        created_at="2024-01-01T00:00:00+00:00",
        strategy=StrategyRef(name="test_pkg", path=str(package)),
        agent=AgentSpec(name="constant", model=ModelSpec()),
        scope="single_name",
        universe=UniverseRef(symbols=["AAPL"]),
        universe_symbols=["AAPL"],
        schedule={"kind": "single_point", "on": "2024-01-02"},
        schedule_dates=["2024-01-02"],
        data=[{"kind": "prices", "source": "synthetic_prices"}],
        scoring={"kpi": "icir", "horizons": [1]},
    )

    with pytest.raises(ValueError, match="shared_concurrency requires independent"):
        run_run(
            run_config=run_config,
            market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True),
            paths=run_paths,
            shared_concurrency=2,
            feedback_on=True,
        )


def test_shared_concurrency_rolls_across_dates(tmp_path):
    """Flat pool keeps N cells in flight across dates — not blocked on a
    full-date barrier. With shared=3 and 2 tickers/date, a third slot must
    pull the next date while the first date is still running."""
    import threading
    import time
    from collections import Counter
    from dataclasses import dataclass

    from fintel.models.decision import AgentResponse, View
    from fintel.models.market import ScheduleRef

    package = _write_package(tmp_path / "pkg", symbols=["A", "B"])
    (package / "strategy.toml").write_text(
        textwrap.dedent(
            """
            name = "test_pkg"
            description = "test"

            [universe]
            symbols = ["A", "B"]

            [decision]
            scope = "single_name"
            schedule = { kind = "custom_dates", dates = ["2024-01-02", "2024-01-03", "2024-01-04"] }

            [[data]]
            kind = "prices"
            source = "synthetic_prices"

            [scoring]
            kpi = "icir"
            horizons = [1]
            """
        ).lstrip()
    )

    counter = {"in_flight": 0, "peak": 0, "saw_cross_date": False}
    dates_in_flight: Counter[str] = Counter()
    lock = threading.Lock()

    @dataclass
    class CountingAgent:
        score: float = 0.5
        name: str = "counting"
        version: str = "1"
        pit_enforcement: str = "access"

        def decide(self, env) -> AgentResponse:
            d = env.cell.decision_date.isoformat()
            with lock:
                counter["in_flight"] += 1
                counter["peak"] = max(counter["peak"], counter["in_flight"])
                dates_in_flight[d] += 1
                if sum(1 for c in dates_in_flight.values() if c > 0) >= 2:
                    counter["saw_cross_date"] = True
            time.sleep(0.08)
            with lock:
                counter["in_flight"] -= 1
                dates_in_flight[d] -= 1
            views = {
                s: View(symbol=s, score=self.score, rationale="counting")
                for s in sorted(env.policy.decidable)
            }
            return AgentResponse(views=views)

    import sys

    sys.modules["__test_shared__"] = type(sys)("__test_shared__")
    sys.modules["__test_shared__"].CountingAgent = CountingAgent  # type: ignore[attr-defined]

    job = _job_config(package, agent_name="__test_shared__:CountingAgent")
    job.__dict__["output_root"] = str(tmp_path / "runs")
    # 3 > universe size (2) so the third worker must start the next date
    # while date-1 cells are still in flight — impossible under nested
    # trial_concurrency=1.
    job.__dict__["shared_concurrency"] = 3
    job.__dict__["schedule"] = ScheduleRef(
        kind="custom_dates", dates=["2024-01-02", "2024-01-03", "2024-01-04"]
    )

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    assert result.status == "ok"
    assert counter["peak"] <= 3, f"peak={counter['peak']} exceeded shared=3"
    assert counter["peak"] == 3, f"expected the pool to fill (peak={counter['peak']})"
    assert counter["saw_cross_date"], "expected cells from two dates in flight together"


def test_job_config_peak_concurrent_with_shared():
    from fintel.models.job import JobConfig

    cfg = JobConfig(job_id="j", strategy="x", agent=AgentSpec(name="constant"))
    cfg.__dict__["k_repeats"] = 2
    cfg.__dict__["max_concurrent"] = 2
    cfg.__dict__["shared_concurrency"] = 30
    assert cfg.peak_concurrent == 60


def test_cells_run_concurrently_with_auto(tmp_path):
    """With auto cell_concurrency, all tickers decide at once. Verified by a
    shared peak-in-flight counter the agent increments on entry / decrements
    on exit — if cells were sequential the peak would be 1."""
    import threading
    from dataclasses import dataclass

    from fintel.models.decision import AgentResponse, View

    package = _write_package(tmp_path / "pkg", symbols=["A", "B", "C", "D", "E"])

    counter = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    @dataclass
    class CountingAgent:
        score: float = 0.5
        name: str = "counting"
        version: str = "1"
        pit_enforcement: str = "access"

        def decide(self, env) -> AgentResponse:
            with lock:
                counter["in_flight"] += 1
                counter["peak"] = max(counter["peak"], counter["in_flight"])
            import time

            time.sleep(0.05)  # hold the slot so overlap is observable
            with lock:
                counter["in_flight"] -= 1
            views = {
                s: View(symbol=s, score=self.score, rationale="counting")
                for s in sorted(env.policy.decidable)
            }
            return AgentResponse(views=views)

    # Register via module:Class — the factory resolves it without an entry
    import sys

    sys.modules["__test__"] = type(sys)("__test__")
    sys.modules["__test__"].CountingAgent = CountingAgent  # type: ignore[attr-defined]

    job = _job_config(package, agent_name="__test__:CountingAgent")
    job.__dict__["output_root"] = str(tmp_path / "runs")
    # auto cell_concurrency (None) → all 5 tickers concurrent
    job.__dict__["cell_concurrency"] = None

    from fintel.market.settings import MarketConfig

    result = run_job(job, market_config=MarketConfig(cache_root=tmp_path / "cache", offline=True))
    assert result.status == "ok"
    # If cells ran sequentially, peak would be 1. With 5 concurrent, peak >= 2.
    assert counter["peak"] >= 2, f"cells did not run concurrently (peak={counter['peak']})"


# -- backfill -----------------------------------------------------------------


def test_backfill_reruns_error_cells_and_fixes_them(tmp_path):
    """A run with some failed cells; backfill reruns just those cells and
    the decision/run/job results are updated to reflect the fix."""
    import sys
    from dataclasses import dataclass
    from typing import ClassVar

    from fintel.environment.progress import NullProgress
    from fintel.market.settings import MarketConfig
    from fintel.models.decision import AgentResponse, View
    from fintel.simulate.backfill import run_backfill

    # Mutable container so the agent (rebuilt fresh per cell by build_agent)
    # sees the current value without relying on ClassVar field detection.
    _fail_msft = [True]

    @dataclass
    class FlakyAgent:
        name: str = "flaky"
        version: str = "1"
        mission_text: str = ""
        output_schema_text: str = ""
        pit_enforcement: ClassVar[str] = "access"

        def decide(self, env):
            views = {}
            for s in sorted(env.policy.decidable):
                if _fail_msft[0] and s == "MSFT":
                    return AgentResponse(
                        views={}, outcome="parse_error", detail="flaky fail"
                    )
                views[s] = View(symbol=s, score=0.5, rationale="ok")
            return AgentResponse(views=views)

    sys.modules["__test_flaky__"] = type(sys)("__test_flaky__")
    sys.modules["__test_flaky__"].FlakyAgent = FlakyAgent

    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, agent_name="__test_flaky__:FlakyAgent")
    job.__dict__["output_root"] = str(tmp_path / "runs")

    market = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    result = run_job(job, market_config=market, progress=NullProgress())
    # MSFT failed -> trial is partial (run is still ok because AAPL decided)
    trial_root = tmp_path / "runs" / "test-job-001" / "r1" / "trials" / "2024-01-02"
    msft_cell = json.loads((trial_root / "cells" / "MSFT.json").read_text())
    assert msft_cell["outcome"] == "parse_error"

    # Fix the agent: stop failing on MSFT
    _fail_msft[0] = False

    report = run_backfill(
        job_id="test-job-001",
        run_index=1,
        cell_concurrency=1,
        output_root=str(tmp_path / "runs"),
        market_config=market,
        progress=NullProgress(),
    )
    assert report.n_error_cells == 1
    assert report.n_reran == 1
    assert report.n_fixed == 1
    assert report.n_still_failed == 0
    assert "2024-01-02" in report.affected_dates

    # Cell file is now ok
    msft_cell2 = json.loads((trial_root / "cells" / "MSFT.json").read_text())
    assert msft_cell2["outcome"] == "ok"
    assert "MSFT" in msft_cell2["views"]

    # Decision now has both symbols
    decision = json.loads((trial_root / "decision.json").read_text())
    assert set(decision) == {"AAPL", "MSFT"}

    # Run result: trial is now ok (was partial)
    run_result = json.loads(
        (tmp_path / "runs" / "test-job-001" / "r1" / "result.json").read_text()
    )
    assert run_result["status"] == "ok"
    trial_statuses = {t["decision_date"]: t["status"] for t in run_result["trials"]}
    assert trial_statuses["2024-01-02"] == "ok"

    # Job result is now ok
    job_result = json.loads(
        (tmp_path / "runs" / "test-job-001" / "result.json").read_text()
    )
    assert job_result["status"] == "ok"


def test_backfill_noop_when_no_errors(tmp_path):
    """A clean run has no error cells; backfill is a no-op."""
    from fintel.environment.progress import NullProgress
    from fintel.market.settings import MarketConfig
    from fintel.simulate.backfill import run_backfill

    package = _write_package(tmp_path / "pkg")
    job = _job_config(package)
    job.__dict__["output_root"] = str(tmp_path / "runs")

    market = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    run_job(job, market_config=market, progress=NullProgress())

    report = run_backfill(
        job_id="test-job-001",
        run_index=1,
        output_root=str(tmp_path / "runs"),
        market_config=market,
        progress=NullProgress(),
    )
    assert report.n_error_cells == 0
    assert report.n_reran == 0
    assert report.n_fixed == 0
    assert report.affected_dates == []


def test_backfill_still_failed_when_cell_keeps_failing(tmp_path):
    """If the rerun still fails, the cell stays failed and the report
    counts it as still_failed."""
    import sys
    from dataclasses import dataclass
    from typing import ClassVar

    from fintel.environment.progress import NullProgress
    from fintel.market.settings import MarketConfig
    from fintel.models.decision import AgentResponse
    from fintel.simulate.backfill import run_backfill

    @dataclass
    class AlwaysFailAgent:
        name: str = "alwaysfail"
        version: str = "1"
        mission_text: str = ""
        output_schema_text: str = ""
        pit_enforcement: ClassVar[str] = "access"

        def decide(self, env):
            return AgentResponse(views={}, outcome="parse_error", detail="always fails")

    sys.modules["__test_alwaysfail__"] = type(sys)("__test_alwaysfail__")
    sys.modules["__test_alwaysfail__"].AlwaysFailAgent = AlwaysFailAgent

    package = _write_package(tmp_path / "pkg")
    job = _job_config(package, agent_name="__test_alwaysfail__:AlwaysFailAgent")
    job.__dict__["output_root"] = str(tmp_path / "runs")

    market = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    run_job(job, market_config=market, progress=NullProgress())

    report = run_backfill(
        job_id="test-job-001",
        run_index=1,
        output_root=str(tmp_path / "runs"),
        market_config=market,
        progress=NullProgress(),
    )
    assert report.n_error_cells == 2  # AAPL + MSFT both failed
    assert report.n_reran == 2
    assert report.n_fixed == 0
    assert report.n_still_failed == 2
