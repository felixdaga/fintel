"""Phase 1 tests: the reader adapter + signal build, round-tripped against a
real finished job (`runs/par2-0001`). These verify the fintel-specific seam and
the platform mechanics (signal resolution, transform, ensemble) without any
strategy opinion about what the signal means.
"""

from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path

import pytest

from fintel.evaluate.read import load_job, load_run
from fintel.evaluate.signals import (
    ensemble_signal,
    resolve_signal,
    resolve_transform,
)
from fintel.evaluate.transforms import identity, rank_range, zscore
from fintel.models.common import Symbol

PAR2 = Path(__file__).resolve().parent.parent / "runs" / "par2-0001"


def _has_par2() -> bool:
    return (PAR2 / "r1" / "trials" / "2026-01-02" / "decision.json").is_file()


# --- transforms (pure math, no fixtures) --------------------------------------


def test_identity_is_a_copy():
    sig = {"AAPL": 0.3, "NVDA": -0.1}
    assert identity(sig) == sig
    assert identity(sig) is not sig  # a copy, not the same object


def test_rank_range_is_centered_and_bounded():
    sig = {"A": -1.0, "B": 0.0, "C": 1.0}
    out = rank_range(sig)
    assert out["A"] == pytest.approx(-0.5)
    assert out["C"] == pytest.approx(0.5)
    assert out["B"] == pytest.approx(0.0)


def test_rank_range_ties_share_average_rank():
    sig = {"A": 1.0, "B": 1.0, "C": 2.0}
    out = rank_range(sig)
    assert out["A"] == out["B"]  # tied -> same value
    assert out["C"] > out["A"]


def test_rank_range_single_name_is_zero():
    assert rank_range({"X": 5.0}) == {"X": 0.0}


def test_zscore_is_standardized():
    sig = {"A": 1.0, "B": 2.0, "C": 3.0}
    out = zscore(sig)
    vals = list(out.values())
    assert sum(vals) / len(vals) == pytest.approx(0.0, abs=1e-9)
    # population std of the z-scores is 1
    mu = sum(vals) / len(vals)
    std = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    assert std == pytest.approx(1.0)


def test_zscore_degenerate_spread_is_zeros():
    assert zscore({"A": 7.0, "B": 7.0}) == {"A": 0.0, "B": 0.0}


def test_resolve_signal_builtins_and_callable():
    fn = resolve_signal("single_name")
    out = fn({"AAPL": type("V", (), {"__getattribute__": lambda self, k: 0.3})()})
    # the builtin returns the score; verify it's callable and dict-shaped
    assert isinstance(out, dict)


def test_resolve_signal_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_signal("not_a_real_signal")


def test_resolve_transform_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_transform("not_a_real_transform")


# --- ensemble (platform mechanics) -------------------------------------------


def test_ensemble_cell_mean():
    per_run = [
        {Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.1}},
        {Date(2026, 1, 2): {"AAPL": 0.5, "NVDA": -0.3}},
    ]
    ens = ensemble_signal(per_run)
    assert ens[Date(2026, 1, 2)]["AAPL"] == pytest.approx(0.4)
    assert ens[Date(2026, 1, 2)]["NVDA"] == pytest.approx(-0.2)


def test_ensemble_missing_symbol_excluded_not_zero():
    """A symbol absent from one run is excluded from that run's mean, not
    treated as zero — a missing signal is not a neutral one."""
    per_run = [
        {Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.1}},
        {Date(2026, 1, 2): {"AAPL": 0.5}},  # NVDA absent in run 2
    ]
    ens = ensemble_signal(per_run)
    # AAPL averaged over 2 runs; NVDA only over run 1 (its sole value)
    assert ens[Date(2026, 1, 2)]["AAPL"] == pytest.approx(0.4)
    assert ens[Date(2026, 1, 2)]["NVDA"] == pytest.approx(-0.1)


def test_ensemble_empty_is_empty():
    assert ensemble_signal([]) == {}


# --- reader adapter against the real par2-0001 job ---------------------------


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_load_run_reads_decisions_and_dates():
    run = load_run(PAR2 / "r1")
    assert run.run_id == "r1"
    assert run.k_index == 1
    assert Date(2026, 1, 2) in run.decision_dates
    views = run.views_by_date[Date(2026, 1, 2)]
    assert "NVDA" in views and "AAPL" in views
    # scores are in [-1, 1]
    for v in views.values():
        assert -1.0 <= v.score <= 1.0


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_load_run_reads_behaviour():
    run = load_run(PAR2 / "r1")
    beh = run.behaviour_by_date[Date(2026, 1, 2)]
    # openclaw cells recorded reads
    assert any(b.has_trace for b in beh.values())
    assert all(b.n_reads >= 0 for b in beh.values())


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_load_job_loads_all_repeats():
    runs = load_job(PAR2)
    assert len(runs) == 2  # r1, r2
    assert {r.run_id for r in runs} == {"r1", "r2"}
    assert {r.k_index for r in runs} == {1, 2}


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_build_signals_round_trip():
    """The full Phase 1 pipeline: load -> signal -> transform -> ensemble."""
    from fintel.evaluate import build_signals

    runs = load_job(PAR2)
    sig = build_signals(runs, signal="single_name", transform="single_name")
    # ensemble has the one decision date, both tickers
    assert Date(2026, 1, 2) in sig.ensemble
    ens = sig.ensemble[Date(2026, 1, 2)]
    assert set(ens) == {"AAPL", "NVDA"}
    # ensemble is the cell-mean of the two runs' scores
    r1 = runs[0].views_by_date[Date(2026, 1, 2)]
    r2 = runs[1].views_by_date[Date(2026, 1, 2)]
    for sym in ("AAPL", "NVDA"):
        expected = (r1[sym].score + r2[sym].score) / 2
        assert ens[sym] == pytest.approx(expected)
    # per_run has one entry per repeat
    assert len(sig.per_run) == 2


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_build_signals_rank_range_transform():
    """A non-identity transform still flows through to the ensemble."""
    from fintel.evaluate import build_signals

    runs = load_job(PAR2)
    sig = build_signals(runs, signal="single_name", transform="rank_range")
    ens = sig.ensemble[Date(2026, 1, 2)]
    # rank_range maps to [-0.5, +0.5]; two names -> {-0.5, +0.5}
    assert min(ens.values()) == pytest.approx(-0.5)
    assert max(ens.values()) == pytest.approx(0.5)


# --- KPI protocol (Phase 2) ---------------------------------------------------


def test_resolve_kpi_builtins():
    from fintel.evaluate.kpi import resolve_kpi

    fn = resolve_kpi("single_name_ir")
    assert callable(fn)


def test_resolve_kpi_rejects_unknown():
    from fintel.evaluate.kpi import resolve_kpi

    with pytest.raises(ValueError):
        resolve_kpi("not_a_real_kpi")


def test_spearman_ic_perfect_monotone():
    from fintel.evaluate.kpi import _spearman_ic

    # signal ranks match return ranks exactly -> IC = 1
    sig = {"A": 1.0, "B": 2.0, "C": 3.0}
    fwd = {"A": 0.01, "B": 0.02, "C": 0.03}
    assert _spearman_ic(sig, fwd) == pytest.approx(1.0)


def test_spearman_ic_anti_monotone():
    from fintel.evaluate.kpi import _spearman_ic

    sig = {"A": 1.0, "B": 2.0, "C": 3.0}
    fwd = {"A": 0.03, "B": 0.02, "C": 0.01}
    assert _spearman_ic(sig, fwd) == pytest.approx(-1.0)


def test_spearman_ic_too_few_names_is_none():
    from fintel.evaluate.kpi import _spearman_ic

    assert _spearman_ic({"A": 0.3}, {"A": 0.1}) is None  # need >= 2


def test_single_name_ir_on_synthetic_grid():
    """A known signal/return grid: IC should be exactly 1.0 on every period."""
    from fintel.evaluate.kpi import single_name_ir

    # Build a fake price lookup over a 4-date grid where returns are monotone in
    # the signal each period.
    dates = [Date(2026, 1, 2), Date(2026, 4, 1), Date(2026, 7, 1), Date(2026, 10, 1)]
    # signal: A < B < C every date
    signal = {d: {"A": -0.5, "B": 0.0, "C": 0.5} for d in dates}
    prices = _FakePriceLookup(dates)
    out = single_name_ir(signal, prices, horizons=[1], params={})
    h1 = out["per_horizon"][1]
    assert h1["mean_ic"] == pytest.approx(1.0)
    assert h1["raw_icir"] is None  # zero std -> None (perfect IC, no dispersion)
    assert h1["n_periods"] == 3  # 4 dates, horizon 1 -> 3 forward periods


def test_single_name_ir_empty_when_no_forward_data():
    from fintel.evaluate.kpi import single_name_ir

    out = single_name_ir({}, _FakePriceLookup([]), horizons=[1], params={})
    assert out["per_horizon"][1]["n_periods"] == 0


class _FakePriceLookup:
    """A stand-in for PriceLookup over a synthetic monotone grid.

    Returns are increasing in the symbol's signal rank each period, so a
    signal that ranks A < B < C produces IC = 1.0.
    """

    _RANK = {"A": 0, "B": 1, "C": 2}

    def __init__(self, dates: list[Date]) -> None:
        # entry price cross-sectionally flat; exit price carries the rank bonus,
        # so forward return increases with symbol rank.
        self._dates = dates
        self._p0: dict[Date, float] = {d: 100.0 * (i + 1) for i, d in enumerate(dates)}
        self._p1: dict[tuple[Date, Symbol], float] = {}
        for i, d in enumerate(dates):
            for sym, r in self._RANK.items():
                self._p1[(d, sym)] = 100.0 * (i + 2) + 10.0 * r

    def forward_return(self, symbol: Symbol, start: Date, end: Date) -> float | None:
        p0 = self._p0.get(start)
        p1 = self._p1.get((end, symbol))
        if p0 is None or p1 is None:
            return None
        return p1 / p0 - 1.0


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_compute_kpi_round_trip():
    """The full Phase 2 pipeline over the real job: signal -> KPI."""
    from fintel.evaluate import build_signals
    from fintel.evaluate.kpi import compute
    from fintel.evaluate.prices import price_lookup_for

    runs = load_job(PAR2)
    sig = build_signals(runs, signal="single_name", transform="single_name")
    prices = price_lookup_for(PAR2)
    out = compute(
        sig, prices, kpi="single_name_ir", horizons=[1, 2, 3, 4], params={"metric_key": "icir"}
    )
    assert "ensemble" in out and "per_run" in out
    assert len(out["per_run"]) == 2
    # par2-0001 has a single decision date -> no forward periods at h=1
    # (need >= 2 dates), so IC is empty but the structure is well-formed.
    ens = out["ensemble"]
    assert ens["kpi"] == "single_name_ir"
    assert set(ens["per_horizon"]) == {1, 2, 3, 4}


# --- Phase 3: behaviour (L1) + variance (L2) ----------------------------------


def test_behaviour_no_op_when_no_traces():
    """A scripted/constant agent with no traces -> available=False, not zeros."""
    from fintel.evaluate.behaviour import analyse as analyse_behaviour
    from fintel.models.evaluate import CellBehaviour, RunData

    runs = [
        RunData(
            run_id="r1",
            k_index=1,
            decision_dates=[Date(2026, 1, 2)],
            universe=["AAPL"],
            behaviour_by_date={
                Date(2026, 1, 2): {
                    "AAPL": CellBehaviour(cell="AAPL", decision_date="2026-01-02", has_trace=False)
                }
            },
        )
    ] * 2
    out = analyse_behaviour(runs)
    assert out["available"] is False
    assert "no tool" in out["summary"]["note"]


def test_behaviour_dispersion_across_runs():
    from fintel.evaluate.behaviour import analyse as analyse_behaviour
    from fintel.models.evaluate import CellBehaviour, RunData

    runs = [
        RunData(
            run_id="r1",
            k_index=1,
            decision_dates=[Date(2026, 1, 2)],
            universe=["AAPL"],
            behaviour_by_date={
                Date(2026, 1, 2): {
                    "AAPL": CellBehaviour(
                        cell="AAPL",
                        decision_date="2026-01-02",
                        has_trace=True,
                        n_tool_calls=4,
                        n_reads=3,
                    )
                }
            },
        ),
        RunData(
            run_id="r2",
            k_index=2,
            decision_dates=[Date(2026, 1, 2)],
            universe=["AAPL"],
            behaviour_by_date={
                Date(2026, 1, 2): {
                    "AAPL": CellBehaviour(
                        cell="AAPL",
                        decision_date="2026-01-02",
                        has_trace=True,
                        n_tool_calls=6,
                        n_reads=3,
                    )
                }
            },
        ),
    ]
    out = analyse_behaviour(runs)
    assert out["available"] is True
    cell = out["per_cell"]["2026-01-02|AAPL"]
    assert cell["n_tool_calls_mean"] == 5.0
    assert cell["n_tool_calls_std"] > 0  # 4 vs 6 -> nonzero std


def test_variance_needs_two_runs():
    from fintel.evaluate.variance import analyse as analyse_var

    out = analyse_var([{}])  # one run
    assert out["available"] is False


def test_variance_dispersion_across_runs():
    from fintel.evaluate.variance import analyse as analyse_var

    per_run = [
        {Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.1}},
        {Date(2026, 1, 2): {"AAPL": 0.5, "NVDA": -0.3}},
    ]
    out = analyse_var(per_run)
    assert out["available"] is True
    aapl = out["per_cell"]["2026-01-02|AAPL"]
    assert aapl["score_mean"] == pytest.approx(0.4)
    assert aapl["score_std"] > 0
    assert aapl["sign_agreement"] is True  # both positive
    # ranks agree perfectly across the two runs -> mean_rank_corr = 1.0
    assert out["per_date"]["2026-01-02"]["mean_rank_corr"] == pytest.approx(1.0)


def test_variance_sign_disagreement_flagged():
    from fintel.evaluate.variance import analyse as analyse_var

    per_run = [
        {Date(2026, 1, 2): {"AAPL": 0.3}},
        {Date(2026, 1, 2): {"AAPL": -0.3}},
    ]
    out = analyse_var(per_run)
    assert out["per_cell"]["2026-01-02|AAPL"]["sign_agreement"] is False
    assert out["summary"]["min_sign_agreement"] is False


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_behaviour_and_variance_on_real_job():
    """L1 + L2 over the real par2-0001 job (openclaw, 2 repeats)."""
    from fintel.evaluate import build_signals
    from fintel.evaluate.behaviour import analyse as analyse_behaviour
    from fintel.evaluate.variance import analyse as analyse_var

    runs = load_job(PAR2)
    sig = build_signals(runs, signal="single_name", transform="single_name")
    b = analyse_behaviour(runs)
    v = analyse_var(sig.per_run)
    # openclaw recorded traces -> behaviour available
    assert b["available"] is True
    # 2 repeats -> variance available
    assert v["available"] is True
    assert v["summary"]["n_cells"] >= 2  # NVDA + AAPL


# --- Phase 4: holdings (default opt-in) ---------------------------------------


def test_active_weights_long_only_and_normalized():
    from fintel.evaluate.holdings import active_weights

    w = active_weights({"A": -0.5, "B": 0.0, "C": 0.5})
    assert all(v >= 0 for v in w.values())  # long-only
    assert sum(w.values()) == pytest.approx(1.0)  # normalized
    assert w["C"] > w["B"] > w["A"]  # signal order preserved


def test_active_weights_empty_signal():
    from fintel.evaluate.holdings import active_weights

    assert active_weights({}) == {}


def test_turnover_two_way():
    from fintel.evaluate.holdings import turnover

    # full reversal: 100% in A -> 100% in B = 2.0 two-way
    assert turnover({"A": 1.0}, {"B": 1.0}) == pytest.approx(2.0)
    # no change
    assert turnover({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}) == pytest.approx(0.0)


def test_holdings_opt_in_off_by_default():
    from fintel.evaluate.holdings import build
    from fintel.models.evaluate import Signals

    sig = Signals(per_run=[], ensemble={}, decision_dates=[], universe=[])
    assert build(sig, _FakePriceLookup([]), params={}) is None


def test_holdings_nav_gross_and_net():
    """Gross NAV rises with positive returns; net <= gross once cost is applied."""
    from fintel.evaluate.holdings import build
    from fintel.models.evaluate import Signals

    dates = [Date(2026, 1, 2), Date(2026, 4, 1), Date(2026, 7, 1)]
    # all-positive signal so weights are positive; fake returns all positive
    signal = {d: {"A": 0.5, "B": -0.5} for d in dates}
    sig = Signals(per_run=[signal], ensemble=signal, decision_dates=dates, universe=["A", "B"])
    out = build(sig, _FakePriceLookup(dates), params={"holdings": True, "cost_bps": 10.0})
    assert out is not None
    ens = out["ensemble"]
    assert ens["gross"][-1]["nav"] > 1.0  # positive returns compound up
    assert ens["net"][-1]["nav"] <= ens["gross"][-1]["nav"]  # cost drags net down
    assert ens["turnover_total"] >= 0.0


def test_holdings_first_rebalance_free():
    """The first rebalance incurs no cost (turnover_total only counts i>0)."""
    from fintel.evaluate.holdings import build
    from fintel.models.evaluate import Signals

    dates = [Date(2026, 1, 2), Date(2026, 4, 1)]
    signal = {dates[0]: {"A": 0.5}, dates[1]: {"A": 0.5}}
    sig = Signals(per_run=[signal], ensemble=signal, decision_dates=dates, universe=["A"])
    out = build(sig, _FakePriceLookup(dates), params={"holdings": True, "cost_bps": 100.0})
    # only one forward period (i=0, first rebalance) -> turnover_total == 0
    assert out["ensemble"]["turnover_total"] == 0.0


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_holdings_on_real_job_opt_in():
    from fintel.evaluate import build_signals
    from fintel.evaluate.holdings import build as build_holdings
    from fintel.evaluate.prices import price_lookup_for

    runs = load_job(PAR2)
    sig = build_signals(runs, signal="single_name", transform="single_name")
    prices = price_lookup_for(PAR2)
    # off by default
    assert build_holdings(sig, prices, params={}) is None
    # on when opted in
    out = build_holdings(sig, prices, params={"holdings": True})
    assert out is not None
    # single decision date -> NAV series is just the starting point (no forward period)
    assert out["ensemble"]["gross"][0]["nav"] == 1.0


# --- Phase 5: report pipeline + CLI -------------------------------------------


def _scoring(**over):
    from fintel.models.strategy import ScoringSpec

    base = dict(
        kpi="single_name_ir",
        signal="single_name",
        transform="single_name",
        horizons=[1, 2, 3, 4],
        metric_key="icir",
        params={},
    )
    base.update(over)
    return ScoringSpec(**base)


def test_report_payload_shape_synthetic():
    """The full pipeline over a synthetic 2-date job (so IC has a forward period)."""
    from fintel.evaluate.report import report

    dates = [Date(2026, 1, 2), Date(2026, 4, 1), Date(2026, 7, 1)]
    signal = {d: {"A": -0.5, "B": 0.0, "C": 0.5} for d in dates}
    # build a tiny job dir with decisions
    import tempfile

    from fintel.models.decision import View

    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "synth"
        for k in (1, 2):
            rdir = job / f"r{k}"
            for d in dates:
                tdir = rdir / "trials" / d.isoformat()
                tdir.mkdir(parents=True)
                views = {s: View(symbol=s, score=signal[d][s]) for s in signal[d]}
                (tdir / "decision.json").write_text(
                    json.dumps({s: v.model_dump(mode="json") for s, v in views.items()})
                )
        payload = report(
            job, scoring=_scoring(params={"holdings": True}), prices=_FakePriceLookup(dates)
        )
        assert payload.job_id == "synth"
        assert payload.kpi == "single_name_ir"
        assert payload.kpi_result["ensemble"]["kpi"] == "single_name_ir"
        # 3 dates, horizon 1 -> 2 forward periods -> IC computable
        assert payload.kpi_result["ensemble"]["per_horizon"][1]["n_periods"] == 2
        assert payload.holdings is not None  # opted in
        assert payload.behaviour.get("available") is False  # no traces in synth


@pytest.mark.skipif(not _has_par2(), reason="runs/par2-0001 not present")
def test_report_on_real_job():
    """The full pipeline over par2-0001, written to disk."""
    # copy par2-0001 to a temp dir so we don't pollute the real run with a report/
    import shutil
    import tempfile

    from fintel.evaluate.report import report, write_report

    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "par2-0001"
        shutil.copytree(PAR2, job)
        payload = report(job, scoring=_scoring(params={"holdings": True}))
        paths = write_report(payload, job)
        assert paths["json"].is_file()
        assert paths["markdown"].is_file()
        md = paths["markdown"].read_text()
        assert "fintel report" in md
        assert "KPI" in md
        assert "single_name_ir" in md


def test_cli_report_subparser_exists():
    """The `fintel report` subcommand is wired into the CLI parser."""
    from fintel.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["report", "some-job"])
    assert args.command == "report"
    assert args.job_id == "some-job"


# --- Phase 6: architecture conformance ----------------------------------------


def test_evaluate_never_imports_simulate():
    """The evaluation layer is read-only over finished runs; it must never
    reach into the simulation. This is the load-bearing guard that keeps a
    re-score from needing a re-run. (Also enforced in test_architecture.py.)"""
    import ast

    import fintel

    root = Path(fintel.__file__).parent / "evaluate"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.startswith("fintel.simulate"):
                    violations.append(f"{path.name}: imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("fintel.simulate"):
                        violations.append(f"{path.name}: imports {alias.name}")
    assert not violations, "evaluate/ imports simulate/:\n" + "\n".join(violations)


def test_evaluate_modules_import_cleanly():
    """Every evaluate module imports without error (the architecture test's
    walk-packages check, scoped to this layer)."""
    import importlib
    import pkgutil

    import fintel.evaluate

    failures: list[str] = []
    for mod in pkgutil.walk_packages(fintel.evaluate.__path__, "fintel.evaluate."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "evaluate import failures:\n" + "\n".join(failures)


# --- Extension seam: strategy-defined signal + KPI via module:Callable ----------


def test_custom_signal_callable_resolves_and_runs():
    """A strategy may define arbitrary signal construction (clamp, normalize,
    weight…) as a `module:Callable`. The platform resolves and calls it with
    no assumption about its internals."""
    from fintel.evaluate.signals import resolve_signal
    from fintel.models.decision import View

    fn = resolve_signal("tests.eval_custom_callables:truncate_normalize_signal")
    views = {
        "AAPL": View(symbol="AAPL", score=0.9),  # clamps to 0.5
        "NVDA": View(symbol="NVDA", score=-0.9),  # clamps to -0.5
    }
    out = fn(views)
    # clamped to [-0.5, 0.5]
    assert out["AAPL"] == pytest.approx(0.5)
    assert out["NVDA"] == pytest.approx(-0.5)
    # cross-sectionally normalized to unit sum-of-abs
    assert sum(abs(v) for v in out.values()) == pytest.approx(1.0)


def test_custom_kpi_callable_resolves_and_runs():
    """A strategy may define an arbitrary KPI (not just IC) as a
    `module:Callable`. Here a non-IR metric (mean absolute signal)."""
    from fintel.evaluate.kpi import resolve_kpi

    fn = resolve_kpi("tests.eval_custom_callables:mean_abs_score_kpi")
    signal = {Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.2}}
    out = fn(signal, _FakePriceLookup([]), horizons=[1], params={"metric_key": "mean_abs"})
    assert out["kpi"] == "mean_abs_score"
    assert out["mean_abs"] == pytest.approx(0.25)  # (0.3 + 0.2)/2
    assert out["n_cells"] == 2


def test_build_signals_with_custom_signal_callable():
    """The full signal build accepts a `module:Callable` signal, end to end."""
    from fintel.evaluate import build_signals
    from fintel.models.decision import View
    from fintel.models.evaluate import RunData

    runs = [
        RunData(
            run_id="r1",
            k_index=1,
            decision_dates=[Date(2026, 1, 2)],
            universe=["AAPL", "NVDA"],
            views_by_date={
                Date(2026, 1, 2): {
                    "AAPL": View(symbol="AAPL", score=0.9),
                    "NVDA": View(symbol="NVDA", score=-0.9),
                }
            },
        )
    ]
    sig = build_signals(
        runs,
        signal="tests.eval_custom_callables:truncate_normalize_signal",
        transform="identity",
    )
    ens = sig.ensemble[Date(2026, 1, 2)]
    assert ens["AAPL"] == pytest.approx(0.5)
    assert ens["NVDA"] == pytest.approx(-0.5)


def test_compute_with_custom_kpi_callable():
    """The KPI orchestration accepts a `module:Callable` KPI, end to end."""
    from fintel.evaluate.kpi import compute
    from fintel.models.evaluate import Signals

    sig = Signals(
        per_run=[{Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.2}}],
        ensemble={Date(2026, 1, 2): {"AAPL": 0.3, "NVDA": -0.2}},
        decision_dates=[Date(2026, 1, 2)],
        universe=["AAPL", "NVDA"],
    )
    out = compute(
        sig,
        _FakePriceLookup([]),
        kpi="tests.eval_custom_callables:mean_abs_score_kpi",
        horizons=[1],
        params={"metric_key": "mean_abs"},
    )
    assert out["ensemble"]["kpi"] == "mean_abs_score"
    assert out["ensemble"]["mean_abs"] == pytest.approx(0.25)


# --- Conformance: a dissimilar package needs NO platform change ---------------


def test_dissimilar_package_runs_end_to_end():
    """The §1 conformance test, applied to the evaluation layer: a second,
    dissimilar strategy (portfolio scope, rank_range transform, a custom
    module:Callable KPI, no `single_name` anywhere) runs through the full
    evaluation pipeline with zero edits to `fintel/evaluate/`.

    The only contract the package must satisfy is the decision shape:
    `decision.json` keyed by symbol -> `View` (score in [-1, 1]). Everything
    else the package owns (signal construction, KPI math) via builtins or
    `module:Callable`.
    """
    import tempfile

    from fintel.evaluate.report import report
    from fintel.models.decision import View
    from fintel.models.strategy import ScoringSpec

    dates = [Date(2026, 1, 2), Date(2026, 4, 1), Date(2026, 7, 1)]
    # portfolio-scope: one cell sees the whole universe each date
    universe = ["AAPL", "NVDA", "MSFT"]

    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "portf-0001"
        for k in (1, 2):
            rdir = job / f"r{k}"
            (rdir).mkdir(parents=True)
            # a dissimilar scoring spec: rank_range transform + custom KPI
            (rdir / "config.json").write_text(
                json.dumps(
                    {
                        "run_id": f"portf-0001-r{k}",
                        "schedule_dates": [d.isoformat() for d in dates],
                        "universe_symbols": universe,
                        "scoring": {
                            "signal": "single_name",  # score is the signal
                            "kpi": "tests.eval_custom_callables:mean_abs_score_kpi",
                            "transform": "rank_range",  # cross-sectional rank
                            "horizons": [1],
                            "metric_key": "mean_abs",
                            "params": {},
                        },
                    }
                )
            )
            for d in dates:
                tdir = rdir / "trials" / d.isoformat()
                tdir.mkdir(parents=True)
                views = {
                    s: View(symbol=s, score={(0): -0.4, (1): 0.1, (2): 0.6}[i])
                    for i, s in enumerate(universe)
                }
                (tdir / "decision.json").write_text(
                    json.dumps({s: v.model_dump(mode="json") for s, v in views.items()})
                )

        # The dissimilar scoring spec, built from the run config — no platform edit.
        scoring = ScoringSpec(
            signal="single_name",
            kpi="tests.eval_custom_callables:mean_abs_score_kpi",
            transform="rank_range",
            horizons=[1],
            metric_key="mean_abs",
        )
        payload = report(job, scoring=scoring, prices=_FakePriceLookup(dates))
        # the custom KPI ran (not single_name_ir), over the rank_range ensemble
        assert payload.kpi == "tests.eval_custom_callables:mean_abs_score_kpi"
        assert payload.kpi_result["ensemble"]["kpi"] == "mean_abs_score"
        # rank_range maps 3 names to {-0.5, 0, +0.5} each date -> mean_abs = (0.5+0+0.5)/3
        assert payload.kpi_result["ensemble"]["mean_abs"] == pytest.approx(1.0 / 3)
        # variance layer ran on the portfolio-scope signals
        assert payload.variance["available"] is True
        assert payload.universe == sorted(universe)
