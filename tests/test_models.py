from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from fintel.models import ids
from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.common import PORTFOLIO_CELL
from fintel.models.decision import AgentResponse, Decision, View
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.models.paths import JobPaths
from fintel.models.strategy import StrategyManifest, StrategyPaths
from fintel.models.trace import Usage
from fintel.models.trial import TrialConfig

# ── ids ──────────────────────────────────────────────────────────────────────


def test_job_id_is_slugged_and_unique():
    a = ids.new_job_id(strategy="Systematic StockRate DJIA", agent="openclaw")
    b = ids.new_job_id(strategy="Systematic StockRate DJIA", agent="openclaw")
    assert a != b
    assert a.startswith("systematic-stockrate-djia-openclaw-")
    assert " " not in a


def test_run_id_is_one_based():
    assert ids.run_id("job", 1) == "job-r1"
    with pytest.raises(ValueError):
        ids.run_id("job", 0)


def test_cell_id_falls_back_to_portfolio():
    assert ids.cell_id("AAPL") == "AAPL"
    assert ids.cell_id(None) == PORTFOLIO_CELL


# ── paths ────────────────────────────────────────────────────────────────────


def test_artifact_layout():
    job = JobPaths.under("runs", "j1")
    assert job.config == Path("runs/j1/config.json")
    run = job.run(1)
    assert run.root == Path("runs/j1/r1")
    assert run.config == Path("runs/j1/r1/config.json")
    assert run.result == Path("runs/j1/r1/result.json")
    assert run.log == Path("runs/j1/r1/run.log")
    trial = run.trial(date(2025, 1, 2))
    assert trial.decision == Path("runs/j1/r1/trials/2025-01-02/decision.json")
    assert trial.trace("AAPL") == Path("runs/j1/r1/trials/2025-01-02/trace/AAPL.jsonl")


def test_k_equals_one_still_nests_under_r1():
    assert JobPaths.under("runs", "j1").run(1).root.name == "r1"


# ── market refs keep unknown keys as params ──────────────────────────────────


def test_schedule_ref_collects_extras_as_params():
    ref = ScheduleRef(kind="custom_dates", dates=["2025-01-02"], start="2025-01-01")
    assert ref.kind == "custom_dates"
    assert ref.params == {"dates": ["2025-01-02"], "start": "2025-01-01"}


def test_universe_ref_requires_a_source():
    UniverseRef(preset="dow30")
    UniverseRef(symbols=["AAPL"])
    UniverseRef(source="pkg.mod:Custom")
    with pytest.raises(ValidationError):
        UniverseRef()


def test_data_binding_params():
    b = DataBinding(kind="prices", source="massive_prices", lookback_days=365)
    assert b.params == {"lookback_days": 365}


# ── decisions ────────────────────────────────────────────────────────────────


def test_view_key_must_match_symbol():
    with pytest.raises(ValidationError):
        AgentResponse(views={"AAPL": View(symbol="MSFT", score=0.1)})


def test_score_is_bounded():
    with pytest.raises(ValidationError):
        View(symbol="AAPL", score=1.5)


def test_source_type_is_open_for_novel_kinds():
    view = View(
        symbol="AAPL",
        score=0.2,
        sources_cited=[{"source_type": "options_chain", "source_id": "x"}],
    )
    assert view.sources_cited[0].source_type == "options_chain"


def test_decision_round_trips_through_json():
    d = Decision(
        run_id="r",
        decision_date=date(2025, 1, 2),
        universe=["AAPL"],
        agent_response=AgentResponse(views={"AAPL": View(symbol="AAPL", score=0.3)}),
        context_hash="abc",
    )
    assert Decision.model_validate_json(d.model_dump_json()) == d


# ── trial fan-out follows declared scope ─────────────────────────────────────


def test_single_name_scope_makes_one_cell_per_symbol():
    cfg = TrialConfig(
        run_id="r",
        decision_date=date(2025, 1, 2),
        universe=["AAPL", "MSFT"],
        scope="single_name",
    )
    assert [c.cell for c in cfg.cells()] == ["AAPL", "MSFT"]


def test_portfolio_scope_makes_one_cell():
    cfg = TrialConfig(
        run_id="r",
        decision_date=date(2025, 1, 2),
        universe=["AAPL", "MSFT"],
        scope="portfolio",
    )
    cells = cfg.cells()
    assert len(cells) == 1
    assert cells[0].is_portfolio
    assert cells[0].symbols == ["AAPL", "MSFT"]


# ── usage merges honestly ────────────────────────────────────────────────────


def test_unknown_cost_stays_none_rather_than_zero():
    assert Usage().merge(Usage()).cost_usd is None
    assert Usage(cost_usd=0.1).merge(Usage()).cost_usd == pytest.approx(0.1)


# ── manifest ─────────────────────────────────────────────────────────────────

MANIFEST = {
    "name": "demo",
    "universe": {"preset": "dow30"},
    "decision": {"scope": "single_name", "schedule": {"kind": "quarterly"}},
    "scoring": {"kpi": "single_name_ir"},
}


def test_manifest_minimal():
    m = StrategyManifest.model_validate(MANIFEST)
    assert m.decision.scope == "single_name"
    assert m.kinds == ()


def test_manifest_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        StrategyManifest.model_validate({**MANIFEST, "univrse": {"preset": "dow30"}})


def test_manifest_rejects_duplicate_kinds():
    with pytest.raises(ValidationError):
        StrategyManifest.model_validate(
            {
                **MANIFEST,
                "data": [
                    {"kind": "prices", "source": "a"},
                    {"kind": "prices", "source": "b"},
                ],
            }
        )


def test_strategy_paths_follow_the_manifest():
    m = StrategyManifest.model_validate({**MANIFEST, "mission_file": "brief.md"})
    paths = StrategyPaths(root=Path("/pkg"), manifest=m)
    assert paths.mission == Path("/pkg/brief.md")
    assert paths.cache_dir == Path("/pkg/cache")
    assert paths.lock == Path("/pkg/strategy.lock")


# ── agent spec ───────────────────────────────────────────────────────────────


def test_model_spec_noop_when_empty():
    assert ModelSpec().is_noop()
    assert ModelSpec().to_params() == {}
    assert not ModelSpec(temperature=0.0).is_noop()


def test_pinned_provider_disables_fallbacks_by_default():
    params = ModelSpec(provider=["xiaomi"]).to_params()
    assert params["provider"] == {"order": ["xiaomi"], "allow_fallbacks": False}


def test_agent_options_are_opaque():
    spec = AgentSpec(name="openclaw", options={"profile": "fintel", "anything": 1})
    assert spec.options["anything"] == 1
