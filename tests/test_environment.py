"""The environment: what one agent invocation may see, and what it may not.

Weighted towards the failures the old layer actually had — a stale cell served
from a cached bundle, a universe enforced only at submission, five disagreeing
PIT implementations, and empty results standing in for errors.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from fintel.environment import (
    AccessDenied,
    AccessLog,
    AccessPolicy,
    Cell,
    DataAccess,
    RuntimeConfig,
    build_environment,
    cells_for,
)
from fintel.environment import evidence as ev
from fintel.environment.access import Reading, _is_empty
from fintel.environment.factory import build_policy
from fintel.environment.session import CELL_FILE, SessionDir, SessionError, read_cell
from fintel.environment.tools import tool_name
from fintel.market.data.base import DataError
from fintel.pit import Cutoff
from tests import fixtures

DAY = date(2024, 6, 3)
UNIVERSE = ["AAPL", "MSFT", "NVDA"]


def a_cell(symbol: str = "AAPL", **kw) -> Cell:
    return Cell(run_id="job-r1", decision_date=DAY, symbols=(symbol,), **kw)


class Spy:
    """Records the cutoff it was handed, so tests can prove it wasn't chosen
    by the caller or defaulted somewhere downstream."""

    name = "spy"
    kinds = ("prices",)

    def __init__(self, data=None, raises=None):
        self.data = data if data is not None else [{"date": "2024-05-01"}]
        self.raises = raises
        self.calls: list[tuple[dict, Cutoff]] = []

    def fetch(self, query, cutoff):
        self.calls.append((dict(query), cutoff))
        if self.raises is not None:
            raise self.raises
        return self.data


def access_for(source=None, *, cell=None, kinds=("prices",), peers=()) -> DataAccess:
    cell = cell or a_cell()
    policy = AccessPolicy(
        kinds=frozenset(kinds), decidable=frozenset(cell.symbols), peers=frozenset(peers)
    )
    return DataAccess(cell=cell, sources={"prices": source or Spy()}, policy=policy)


# ── the cell ─────────────────────────────────────────────────────────────────


def test_the_cell_owns_the_cutoff():
    cell = a_cell()
    assert cell.cutoff == Cutoff(DAY)
    assert cell.cutoff.decision_date == DAY


def test_single_name_scope_decides_on_exactly_one_symbol():
    with pytest.raises(ValueError, match="exactly one symbol"):
        Cell(run_id="r", decision_date=DAY, symbols=("AAPL", "MSFT"))


def test_portfolio_scope_takes_the_whole_universe():
    cell = Cell(run_id="r", decision_date=DAY, symbols=tuple(UNIVERSE), scope="portfolio")
    assert cell.is_portfolio
    assert cell.name == "__portfolio__"


def test_a_cell_needs_an_identity():
    with pytest.raises(ValueError, match="run_id"):
        Cell(run_id="", decision_date=DAY, symbols=("AAPL",))
    with pytest.raises(ValueError, match="at least one symbol"):
        Cell(run_id="r", decision_date=DAY, symbols=())
    with pytest.raises(ValueError, match="duplicate"):
        Cell(run_id="r", decision_date=DAY, symbols=("AAPL", "AAPL"), scope="portfolio")


def test_cell_keys_are_unique_across_the_fan_out():
    cells = cells_for(run_id="r", decision_date=DAY, symbols=UNIVERSE, scope="single_name")
    assert len({c.key for c in cells}) == 3


def test_scope_drives_the_fan_out():
    single = cells_for(run_id="r", decision_date=DAY, symbols=UNIVERSE, scope="single_name")
    portfolio = cells_for(run_id="r", decision_date=DAY, symbols=UNIVERSE, scope="portfolio")
    assert len(single) == 3
    assert len(portfolio) == 1
    assert portfolio[0].symbols == tuple(UNIVERSE)


# ── PIT: one chokepoint, injected, unreachable by the agent ──────────────────


def test_the_cutoff_is_injected_from_the_cell():
    spy = Spy()
    access_for(spy).read("prices", symbol="AAPL")
    query, cutoff = spy.calls[0]
    assert cutoff == Cutoff(DAY)
    assert "cutoff" not in query and "as_of" not in query


def test_an_agent_cannot_supply_or_widen_the_cutoff():
    """The old server accepted an `as_of` it then clamped. Here there is no such
    parameter to clamp, so there is nothing to get the clamp wrong about."""
    spy = Spy()
    reading = access_for(spy).read("prices", symbol="AAPL", as_of="2030-01-01")
    # `as_of` is passed through as an ordinary query key the source ignores; the
    # authoritative cutoff is still the cell's.
    assert spy.calls[0][1] == Cutoff(DAY)
    assert reading.status == "ok"


def test_read_is_the_only_path_and_it_always_records():
    access = access_for()
    access.read("prices", symbol="AAPL")
    access.read("news", symbol="AAPL")
    assert [r.status for r in access.readings] == ["ok", "denied"]


# ── absence is not failure ───────────────────────────────────────────────────


def test_a_source_that_answers_nothing_is_empty():
    reading = access_for(Spy(data=[])).read("prices", symbol="AAPL")
    assert reading.status == "empty"
    assert "no prices available" in reading.payload()["detail"]


def test_a_source_that_breaks_is_failed_not_empty():
    """The distinction the old tools erased: a missing API key and a quiet
    company both returned []."""
    reading = access_for(Spy(raises=DataError("MASSIVE_API_KEY not set"))).read(
        "prices", symbol="AAPL"
    )
    assert reading.status == "failed"
    payload = reading.payload()
    assert payload["data"] is None
    assert "MASSIVE_API_KEY" in payload["error"]


def test_an_unexpected_exception_is_also_failed_not_empty():
    reading = access_for(Spy(raises=ZeroDivisionError("bad adapter"))).read("prices", symbol="AAPL")
    assert reading.status == "failed"
    assert "ZeroDivisionError" in reading.detail


def test_emptiness_detection_across_shapes():
    assert _is_empty(None) and _is_empty([]) and _is_empty("") and _is_empty({})
    assert _is_empty(pd.DataFrame())
    assert not _is_empty([1])
    assert not _is_empty(pd.DataFrame({"close": [1.0]}))
    # A computed kind whose values are all None is empty even though keys exist.
    assert _is_empty({"pe": None, "notes": ["no_filing"], "as_of": "2024-06-03"})
    assert not _is_empty({"pe": 20.0, "notes": []})


def test_the_summary_distinguishes_degraded_from_quiet():
    access = access_for(Spy(raises=DataError("down")))
    access.read("prices", symbol="AAPL")
    assert access.summary() == {"failed": 1}


# ── the universe is enforced on reads, not just submissions ──────────────────


def test_reading_outside_the_universe_is_denied():
    """The old server let data tools serve any symbol and only checked the
    universe at submit time."""
    reading = access_for().read("prices", symbol="TSLA")
    assert reading.status == "denied"
    assert "outside this cell's universe" in reading.detail


def test_peers_widen_reads_but_never_decisions():
    policy = build_policy(
        cell=a_cell(), kinds=("prices",), universe=UNIVERSE, peers=True
    )
    assert "MSFT" in policy.readable
    assert "MSFT" not in policy.decidable
    policy.check_symbol("MSFT")
    with pytest.raises(AccessDenied, match="readable but not decidable"):
        policy.check_decidable("MSFT")


def test_an_undeclared_kind_is_denied_with_the_declared_list():
    reading = access_for(kinds=("prices",)).read("news", symbol="AAPL")
    assert reading.status == "denied"
    assert "declared kinds: ['prices']" in reading.detail


def test_a_declared_but_unbound_kind_is_denied_not_silently_empty():
    cell = a_cell()
    access = DataAccess(
        cell=cell,
        sources={},
        policy=AccessPolicy(kinds=frozenset({"prices"}), decidable=frozenset(cell.symbols)),
    )
    reading = access.read("prices", symbol="AAPL")
    assert reading.status == "denied"
    assert "not bound to a source" in reading.detail


def test_oversized_lookbacks_are_trimmed_rather_than_refused():
    spy = Spy()
    # The cap is the strategy's per-kind lookback (lookback_caps), not a global
    # max_lookback_days. A caller may request less, never more.
    policy = AccessPolicy(
        kinds=frozenset({"prices"}),
        decidable=frozenset({"AAPL"}),
        lookback_caps=frozenset({("prices", 365)}),
    )
    DataAccess(cell=a_cell(), sources={"prices": spy}, policy=policy).read(
        "prices", symbol="AAPL", lookback_days=99999
    )
    assert spy.calls[0][0]["lookback_days"] == 365


def test_a_nonsense_lookback_becomes_the_minimum():
    spy = Spy()
    access_for(spy).read("prices", symbol="AAPL", lookback_days=0)
    assert spy.calls[0][0]["lookback_days"] == 1


# ── session isolation ────────────────────────────────────────────────────────


def test_the_session_path_is_derived_from_the_cell_not_discovered(tmp_path):
    """No pid walking: two cells cannot land in the same directory."""
    a = SessionDir(root=tmp_path, cell=a_cell("AAPL"))
    b = SessionDir(root=tmp_path, cell=a_cell("MSFT"))
    assert a.path != b.path
    assert a.path.parts[-3:] == ("job-r1", "2024-06-03", "AAPL")


def test_a_session_writes_its_cell_identity_for_a_subprocess_to_read(tmp_path):
    session = SessionDir(root=tmp_path, cell=a_cell())
    session.create()
    got = read_cell(session.path)
    assert got["decision_date"] == "2024-06-03"
    assert got["symbols"] == ["AAPL"]
    assert session.env() == {"FINTEL_SESSION_DIR": str(session.path)}


def test_a_populated_session_dir_is_refused_rather_than_inherited(tmp_path):
    """A reused directory means a colliding or stale cell. The old server
    happily served whichever bundle it had cached first."""
    session = SessionDir(root=tmp_path, cell=a_cell())
    session.create()
    (session.path / "leftover.json").write_text("{}")
    with pytest.raises(SessionError, match="already has contents"):
        session.create()
    session.create(reset=True)
    assert not (session.path / "leftover.json").exists()
    assert (session.path / CELL_FILE).is_file()


def test_a_missing_cell_file_fails_loudly(tmp_path):
    with pytest.raises(SessionError, match="was not set up"):
        read_cell(tmp_path)


def test_the_session_env_carries_no_credentials(tmp_path):
    """Nothing written into a run artifact may contain a secret."""
    session = SessionDir(root=tmp_path, cell=a_cell())
    session.create()
    assert list(session.env()) == ["FINTEL_SESSION_DIR"]
    written = (session.path / CELL_FILE).read_text().lower()
    for smell in ("api_key", "apikey", "token", "secret", "password"):
        assert smell not in written


# ── the access log ───────────────────────────────────────────────────────────


def test_the_log_records_every_read_including_the_refused_ones(tmp_path):
    cell = a_cell()
    log = AccessLog(cell=cell, path=tmp_path / "access.jsonl")
    access = DataAccess(
        cell=cell,
        sources={"prices": Spy()},
        policy=AccessPolicy(kinds=frozenset({"prices"}), decidable=frozenset({"AAPL"})),
        on_read=log.record,
    )
    access.read("prices", symbol="AAPL")
    access.read("prices", symbol="TSLA")
    access.read("news", symbol="AAPL")

    lines = [json.loads(x) for x in (tmp_path / "access.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "cell_opened"
    assert [x["status"] for x in lines if x["event"] == "read"] == ["ok", "denied", "denied"]
    assert log.summary()["degraded"] is True


def test_the_log_does_not_duplicate_the_cache(tmp_path):
    """It records what was asked and how it went, not another copy of the data."""
    log = AccessLog(cell=a_cell(), path=tmp_path / "a.jsonl")
    record = log.record(
        Reading(kind="prices", query={"symbol": "AAPL"}, status="ok", data=[1, 2, 3])
    )
    assert record["n"] == 3
    assert "data" not in record


def test_a_quiet_run_is_not_reported_as_degraded():
    log = AccessLog(cell=a_cell())
    log.record(Reading(kind="news", query={}, status="empty", data=[]))
    assert log.summary()["degraded"] is False
    assert log.summary()["by_status"] == {"empty": 1}


def test_a_truncated_trace_still_loads(tmp_path):
    from fintel.environment import trace

    path = tmp_path / "t.jsonl"
    path.write_text('{"event": "read"}\n{"event": "read"\n')
    assert len(trace.load(path)) == 1


# ── the tool surface is generated ────────────────────────────────────────────


def test_tool_names_read_naturally():
    assert tool_name("prices") == "get_prices"
    assert tool_name("web_search") == "web_search"


def test_only_declared_kinds_become_tools(tmp_path):
    env = an_environment(tmp_path, kinds=("prices", "fundamentals"))
    assert set(env.tools.names) == {"get_prices", "get_fundamentals"}
    assert "get_news" not in env.tools.names


def test_a_tool_schema_comes_from_the_catalog():
    from fintel.environment.tools import spec_for
    from fintel.market import catalog

    catalog.register_builtins()
    spec = spec_for("prices", "massive_prices", decision_date="2024-06-03")
    assert spec.name == "get_prices"
    assert spec.required == ("symbol",)
    assert "lookback_days" in spec.schema["properties"]
    # The field roster and PIT boundary come from the catalog, so tool help
    # cannot contradict the source the way the old hand-written docstrings did.
    assert "close" in spec.description
    assert "2024-06-03" in spec.description
    assert "cannot be passed, widened, or bypassed" in spec.description


def test_a_tool_schema_follows_whichever_source_is_bound(tmp_path):
    """Bind a different source and the tool's arguments change with it."""
    env = an_environment(tmp_path, kinds=("prices",))
    spec = next(s for s in env.tools.descriptors() if s.name == "get_prices")
    assert spec.required == ("symbol",)
    assert "price" in spec.schema["properties"]  # flat_prices' own knob


def test_a_strategy_owned_param_is_not_offered_to_the_agent(tmp_path):
    """window_days defines what a P/E means; two readings in one run must agree.
    filings_lookback_days is internal to the ratios source, not strategy-visible."""
    env = an_environment(tmp_path, kinds=("prices", "fundamentals", "ratios"))
    spec = next(s for s in env.tools.descriptors() if s.name == "get_ratios")
    assert "window_days" not in spec.schema["properties"]
    assert "filings_lookback_days" not in spec.schema["properties"]


def test_web_search_asks_for_a_query_not_a_symbol(tmp_path):
    env = an_environment(tmp_path, kinds=("web_search",))
    spec = next(s for s in env.tools.descriptors() if s.name == "web_search")
    assert spec.required == ("query",)


def test_calling_an_absent_tool_says_what_exists(tmp_path):
    env = an_environment(tmp_path, kinds=("prices",))
    out = env.tools.call("get_news", {"symbol": "AAPL"})
    assert out["status"] == "denied"
    assert "get_prices" in out["error"]


def test_a_tool_call_missing_its_subject_is_refused(tmp_path):
    env = an_environment(tmp_path, kinds=("prices",))
    assert env.tools.call("get_prices", {})["status"] == "denied"


def test_a_tool_rejects_arguments_it_does_not_accept(tmp_path):
    """Silently ignoring an argument teaches an agent the wrong lesson."""
    env = an_environment(tmp_path, kinds=("prices",))
    out = env.tools.call("get_prices", {"symbol": "AAPL", "as_of": "2030-01-01"})
    assert out["status"] == "denied"
    assert "as_of" in out["error"]


def test_a_tool_call_goes_through_the_same_chokepoint(tmp_path):
    env = an_environment(tmp_path, kinds=("prices",))
    out = env.tools.call("get_prices", {"symbol": "TSLA"})
    assert out["status"] == "denied"
    assert env.log.counts()["denied"] == 1


# ── the evidence pack ────────────────────────────────────────────────────────


def test_missing_values_render_as_na_never_zero():
    assert ev.number(None) == "n/a"
    assert ev.number(0) == "0"
    assert ev.number(float("nan")) == "n/a"
    assert ev.number(1.5e9) == "1.50B"
    assert ev.number(0.075) == "0.07" or ev.number(0.075) == "0.08"


def test_a_failed_section_says_so_instead_of_looking_quiet():
    """The old dossier dropped empty sections, so an unconfigured source and a
    quiet quarter looked identical."""
    reading = Reading(kind="news", query={}, status="failed", detail="no key")
    text = ev._status_line(reading)
    assert "UNAVAILABLE" in text and "Absence here means nothing" in text


def test_an_empty_section_is_present_and_explicit():
    reading = Reading(kind="news", query={}, status="empty", data=[])
    assert ev._status_line(reading) == "None found in the requested window."


def test_sections_are_ordered_for_reading_not_alphabetically():
    """Hard numbers before commentary: a view should form from the filing and the
    price before it meets a headline about them."""
    assert ev.ordered(("news", "prices", "news_sentiment", "fundamentals")) == (
        "prices",
        "fundamentals",
        "news",
        "news_sentiment",
    )
    # An unrecognised kind still appears, after the known ones.
    assert ev.ordered(("custom_kind", "prices")) == ("prices", "custom_kind")


def test_the_pack_covers_every_declared_kind(tmp_path):
    env = an_environment(tmp_path, kinds=("prices", "fundamentals", "ratios", "web_search"))
    text = env.evidence()
    assert "# Evidence — AAPL as of 2024-06-03" in text
    assert "strictly before" in text
    for title in ("## Prices", "## Fundamentals", "## Valuation ratios", "## Web search"):
        assert title in text
    # web_search can't be pre-rendered without a query, and says so.
    assert "needs a query" in text


def test_the_pack_and_the_tools_read_the_same_world(tmp_path):
    env = an_environment(tmp_path, kinds=("prices",))
    env.evidence()
    from_tool = env.tools.call("get_prices", {"symbol": "AAPL"})
    assert from_tool["status"] == "ok"
    # Both paths are recorded through the one access log.
    assert env.log.counts()["ok"] == 2


# ── assembly ─────────────────────────────────────────────────────────────────


def an_environment(tmp_path, *, kinds=("prices",), peers=False, cell=None):
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding

    fixtures.register_all()
    catalogued = {
        "prices": "flat_prices",
        "fundamentals": "annual_fundamentals",
        "ratios": "valuation_ratios",
        "web_search": "web_search",
        "news": "massive_news",
        "news_sentiment": "news_sentiment",
    }
    bindings = [DataBinding(kind=k, source=catalogued[k]) for k in kinds]
    sources = build_data_sources(
        bindings, config=MarketConfig(cache_root=tmp_path / "cache", offline=True)
    )
    return build_environment(
        cell=cell or a_cell(),
        sources=sources,
        universe=UNIVERSE,
        kinds=kinds,
        peers=peers,
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
    )


def test_building_an_environment_creates_its_session_and_trace(tmp_path):
    env = an_environment(tmp_path)
    assert env.session is not None
    assert env.session.path.is_dir()
    env.access.read("prices", symbol="AAPL")
    assert env.session.trace.is_file()


def test_the_environment_grants_only_what_is_bound_and_declared(tmp_path):
    env = an_environment(tmp_path, kinds=("prices", "fundamentals"))
    assert env.kinds == ("fundamentals", "prices")
    assert env.access.read("news", symbol="AAPL").status == "denied"


def test_two_cells_on_one_date_do_not_share_a_session(tmp_path):
    a = an_environment(tmp_path, cell=a_cell("AAPL"))
    b = an_environment(tmp_path, cell=a_cell("MSFT"))
    assert a.session.path != b.session.path
    a.access.read("prices", symbol="AAPL")
    b.access.read("prices", symbol="MSFT")
    assert "AAPL" in read_cell(a.session.path)["symbols"]
    assert "MSFT" in read_cell(b.session.path)["symbols"]
    assert a.log.counts() == {"ok": 1}


def test_closing_reports_what_the_cell_actually_saw(tmp_path):
    env = an_environment(tmp_path, kinds=("prices",))
    env.access.read("prices", symbol="AAPL")
    env.access.read("prices", symbol="TSLA")
    summary = env.close()
    assert summary["cell"] == "AAPL"
    assert summary["n_reads"] == 2
    assert summary["degraded"] is True
    assert summary["kinds_used"] == ["prices"]


def test_a_run_without_a_session_root_keeps_the_trace_in_memory(tmp_path):
    from fintel.market.data.synthetic import SyntheticPrices

    env = build_environment(
        cell=a_cell(),
        sources={"prices": SyntheticPrices()},
        universe=UNIVERSE,
        runtime=RuntimeConfig(),
    )
    assert env.session is None
    assert env.access.read("prices", symbol="AAPL").status == "ok"
    assert env.log.counts() == {"ok": 1}
