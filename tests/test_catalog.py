"""The catalog is the library a strategy picks from, and the thing it's checked against."""

from __future__ import annotations

from datetime import date

import pytest

from fintel.market import catalog
from fintel.market.data.base import DataError
from fintel.market.data.filings import MassiveFilingText, normalise_filing_text
from fintel.market.data.sentiment import NewsSentiment
from fintel.market.data.store import RecordCache
from fintel.market.data.web import WebSearch, query_hash
from fintel.market.factory import build_data_sources
from fintel.market.settings import MarketConfig
from fintel.models.market import DataBinding
from fintel.pit import Cutoff
from tests import fixtures

CUT = Cutoff(date(2024, 6, 3))

# Every kind the platform can serve today. A new kind must land here, so the
# catalog and the docs can't quietly diverge.
EXPECTED_KINDS = {
    "prices",
    "fundamentals",
    "news",
    "filing_text",
    "ratios",
    "news_sentiment",
    "web_search",
}


def test_the_library_covers_every_kind():
    catalog.register_builtins()
    assert EXPECTED_KINDS <= set(catalog.kinds())


def test_every_kind_has_at_least_one_source_with_fields():
    catalog.register_builtins()
    for kind in EXPECTED_KINDS:
        found = catalog.sources(kind=kind)
        assert found, f"no source serves {kind!r}"
        for info in found:
            assert info.fields, f"{info.name} declares no fields"


def test_computed_kinds_declare_their_upstreams():
    catalog.register_builtins()
    assert catalog.source("valuation_ratios").derives_from == ("prices", "fundamentals")
    assert catalog.source("news_sentiment").derives_from == ("news",)
    assert catalog.source("valuation_ratios").is_computed
    assert not catalog.source("massive_prices").is_computed


def test_ratio_field_roster_is_published_in_full():
    from fintel.market.data.ratios import RATIO_FIELDS

    catalog.register_builtins()
    names = catalog.source("valuation_ratios").field_names
    assert names[: len(RATIO_FIELDS)] == RATIO_FIELDS
    assert "date" in names and "entries" in names
    described = [f for f in catalog.fields_for("valuation_ratios") if f.description]
    assert len(described) >= len(RATIO_FIELDS)


def test_credential_requirements_are_declared():
    catalog.register_builtins()
    assert catalog.source("massive_news").requires_env == ("MASSIVE_API_KEY",)
    assert catalog.source("web_search").requires_env == ("BRAVE_API_KEY",)
    # A computed kind needs no key of its own; its upstreams do.
    assert catalog.source("valuation_ratios").requires_env == ()


# ── strategy list must match the catalog ─────────────────────────────────────


def test_a_valid_binding_list_has_no_findings():
    catalog.register_builtins()
    assert (
        catalog.check_bindings(
            [
                DataBinding(kind="prices", source="massive_prices", lookback_days=365),
                DataBinding(kind="fundamentals", source="massive_fundamentals"),
                DataBinding(kind="ratios", source="valuation_ratios"),
            ]
        )
        == []
    )


def test_unknown_source_is_reported_with_the_alternatives():
    catalog.register_builtins()
    problems = catalog.check_bindings([DataBinding(kind="prices", source="massiv_prices")])
    assert len(problems) == 1
    assert "unknown source 'massiv_prices'" in problems[0]
    assert "massive_prices" in problems[0]


def test_wrong_kind_for_a_known_source_is_reported():
    catalog.register_builtins()
    problems = catalog.check_bindings([DataBinding(kind="news", source="massive_prices")])
    assert any("serves kind 'prices'" in p for p in problems)


def test_unknown_param_is_reported_rather_than_ignored():
    catalog.register_builtins()
    problems = catalog.check_bindings(
        [DataBinding(kind="prices", source="massive_prices", lookback_dayz=30)]
    )
    assert any("does not accept param 'lookback_dayz'" in p for p in problems)


def test_missing_upstream_for_a_computed_kind_is_reported():
    catalog.register_builtins()
    problems = catalog.check_bindings(
        [
            DataBinding(kind="prices", source="massive_prices"),
            DataBinding(kind="ratios", source="valuation_ratios"),
        ]
    )
    assert any("computed from 'fundamentals'" in p for p in problems)


def test_a_kind_bound_twice_is_reported():
    catalog.register_builtins()
    problems = catalog.check_bindings(
        [
            DataBinding(kind="prices", source="massive_prices"),
            DataBinding(kind="prices", source="synthetic_prices"),
        ]
    )
    assert any("bound twice" in p for p in problems)


def test_all_findings_are_returned_not_just_the_first():
    catalog.register_builtins()
    problems = catalog.check_bindings(
        [
            DataBinding(kind="prices", source="nope"),
            DataBinding(kind="news", source="massive_news", bad_param=1),
            DataBinding(kind="ratios", source="valuation_ratios"),
        ]
    )
    assert len(problems) >= 3


def test_a_package_supplied_source_is_allowed():
    """An import path is how a package ships its own source; not a typo."""
    assert catalog.check_bindings([DataBinding(kind="prices", source="mypkg.data:Custom")]) == []


def test_required_env_is_aggregated_for_preflight():
    catalog.register_builtins()
    got = catalog.required_env(
        [
            DataBinding(kind="prices", source="massive_prices"),
            DataBinding(kind="web_search", source="web_search"),
            DataBinding(kind="prices", source="synthetic_prices"),
        ]
    )
    assert got == ["BRAVE_API_KEY", "MASSIVE_API_KEY"]


# ── filing text ──────────────────────────────────────────────────────────────


def test_filing_text_normalisation_and_id_fallback():
    got = normalise_filing_text(
        {"filing_date": "2024-05-01", "items_text": " body "},
        symbol="AAPL", form="8-k", text_key="items_text", default_section="8-K",
    )
    assert got["form_type"] == "8-K"
    assert got["text"] == "body"
    assert got["id"] == "8-K:2024-05-01:8-K"
    # An accession number wins when present.
    with_accession = normalise_filing_text(
        {"filing_date": "2024-05-01", "accession_number": "0000320193-24", "text": "x"},
        symbol="AAPL", form="10-K", text_key="text", default_section="business",
    )
    assert with_accession["id"] == "0000320193-24"


def test_filing_text_without_a_filing_date_is_dropped():
    assert (
        normalise_filing_text(
            {"text": "x"}, symbol="AAPL", form="8-K", text_key="text", default_section="8-K"
        )
        is None
    )


def test_filing_text_is_clamped_and_filterable(tmp_path):
    cache = RecordCache(root=tmp_path, kind="filing_text")
    cache.write(
        "AAPL",
        [(date(2023, 1, 1), date(2024, 12, 31))],
        [
            {"id": "a", "form_type": "8-K", "filing_date": "2024-05-01", "text": "alpha"},
            {"id": "b", "form_type": "10-K", "filing_date": "2024-05-02", "text": "bravo"},
            {"id": "c", "form_type": "8-K", "filing_date": "2024-07-01", "text": "future"},
        ],
    )
    src = MassiveFilingText(cache=cache, client=None)
    assert [r["id"] for r in src.fetch({"symbol": "AAPL"}, CUT)] == ["a", "b"]
    only_8k = src.fetch({"symbol": "AAPL", "forms": ["8-K"]}, CUT)
    assert [r["id"] for r in only_8k] == ["a"]
    truncated = src.fetch({"symbol": "AAPL", "max_chars": 2}, CUT)
    assert truncated[0]["text"] == "al"


def test_filing_text_offline_with_nothing_cached_raises(tmp_path):
    src = MassiveFilingText(cache=RecordCache(root=tmp_path, kind="filing_text"), client=None)
    with pytest.raises(DataError, match="nothing cached"):
        src.fetch({"symbol": "AAPL"}, CUT)


# ── news sentiment ───────────────────────────────────────────────────────────


class _News:
    name = "news"
    kinds = ("news",)

    def __init__(self, articles):
        self.articles = articles

    def fetch(self, query, cutoff):
        return cutoff.clamp_records(self.articles, "published_at")


def _insight(ticker: str, mood: str) -> dict:
    return {"ticker": ticker, "sentiment": mood}


def test_sentiment_scores_daily_net_in_minus_one_to_one():
    articles = [
        {"published_at": "2024-05-01", "insights": [_insight("AAPL", "positive")]},
        {"published_at": "2024-05-01", "insights": [_insight("AAPL", "negative")]},
        {"published_at": "2024-05-02", "insights": [_insight("AAPL", "positive")]},
    ]
    src = NewsSentiment(upstream={"news": _News(articles)})
    out = src.fetch({"symbol": "AAPL"}, CUT)
    assert out["series"] == [
        {"date": "2024-05-01", "score": 0.0, "n": 2},
        {"date": "2024-05-02", "score": 1.0, "n": 1},
    ]
    assert out["n_scored"] == 3


def test_sentiment_omits_silent_days_rather_than_scoring_them_zero():
    """No coverage and balanced coverage are different facts."""
    articles = [{"published_at": "2024-05-01", "insights": []}]
    out = NewsSentiment(upstream={"news": _News(articles)}).fetch({"symbol": "AAPL"}, CUT)
    assert out["series"] == []
    assert out["n_articles"] == 1
    assert out["mean_score"] is None


def test_sentiment_ignores_insights_about_other_tickers():
    articles = [{"published_at": "2024-05-01", "insights": [_insight("MSFT", "positive")]}]
    out = NewsSentiment(upstream={"news": _News(articles)}).fetch({"symbol": "AAPL"}, CUT)
    assert out["series"] == []


def test_sentiment_inherits_the_cutoff_from_its_upstream():
    articles = [
        {"published_at": "2024-05-01", "insights": [_insight("AAPL", "positive")]},
        {"published_at": "2024-07-01", "insights": [_insight("AAPL", "negative")]},
    ]
    out = NewsSentiment(upstream={"news": _News(articles)}).fetch({"symbol": "AAPL"}, CUT)
    assert [e["date"] for e in out["series"]] == ["2024-05-01"]


# ── web search ───────────────────────────────────────────────────────────────


def test_web_search_window_ends_the_day_before_the_decision(tmp_path):
    src = WebSearch(cache_root=tmp_path)
    since, through = src.window(CUT, 30)
    assert through == date(2024, 6, 2)
    assert since == date(2024, 5, 3)


def test_web_search_cache_key_is_exact(tmp_path):
    src = WebSearch(cache_root=tmp_path)
    a = src.path("apple earnings", date(2024, 5, 3), date(2024, 6, 2))
    b = src.path("Apple Earnings ", date(2024, 5, 3), date(2024, 6, 2))
    c = src.path("apple earnings", date(2024, 5, 4), date(2024, 6, 2))
    assert a == b  # normalised
    assert a != c  # a different window is a different question
    assert a.name.startswith("2024-06-02_2024-05-03_")


def test_web_search_reads_the_cache_without_a_key(tmp_path):
    import json

    src = WebSearch(cache_root=tmp_path)
    path = src.path("apple earnings", date(2024, 5, 3), date(2024, 6, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"query": "apple earnings", "sources": ["s"]}))
    assert src.fetch({"query": "apple earnings"}, CUT)["sources"] == ["s"]


def test_web_search_without_a_key_or_cache_raises(tmp_path):
    src = WebSearch(cache_root=tmp_path)
    with pytest.raises(DataError, match="no cached result"):
        src.fetch({"query": "anything"}, CUT)


def test_query_hash_is_stable_and_short():
    assert query_hash("apple") == query_hash(" APPLE ")
    assert len(query_hash("apple")) == 12


def test_parse_brave_age_prefers_ymd_never_relative():
    from fintel.market.data.web import parse_brave_age

    assert parse_brave_age(
        ["Thursday, July 30, 2026", "2026-07-30", "4 days ago", "2026-07-30T12:00:00Z"]
    ) == date(2026, 7, 30)
    assert parse_brave_age(["", "", "4 days ago", "2026-07-30T00:00:00Z"]) == date(2026, 7, 30)
    assert parse_brave_age([]) is None
    assert parse_brave_age(None) is None


def test_web_search_clamps_by_brave_age(tmp_path):
    """Soft freshness leaks are dropped when sources[url].age is outside the window."""
    import json

    from fintel.market.data.web import clamp_brave_by_age

    since, through = date(2026, 4, 16), date(2026, 4, 23)
    in_url = "https://example.com/apr"
    out_url = "https://example.com/jul"
    undated_url = "https://example.com/undated"
    payload = {
        "grounding": {
            "generic": [
                {"url": in_url, "title": "in", "snippets": ["a"]},
                {"url": out_url, "title": "leak", "snippets": ["b"]},
                {"url": undated_url, "title": "?", "snippets": ["c"]},
            ]
        },
        "sources": {
            in_url: {"age": ["Wed", "2026-04-20", "x", "2026-04-20T00:00:00Z"]},
            out_url: {"age": ["Thu", "2026-07-30", "x", "2026-07-30T00:00:00Z"]},
            undated_url: {"age": []},
        },
    }
    clamped = clamp_brave_by_age(payload, since, through)
    urls = [g["url"] for g in clamped["grounding"]["generic"]]
    assert urls == [in_url, undated_url]
    assert set(clamped["sources"]) == {in_url, undated_url}

    src = WebSearch(cache_root=tmp_path, clamp_by_age=True)
    path = src.path("apple earnings", since, through)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query": "apple earnings",
                "search_window": {"from": since.isoformat(), "to": through.isoformat()},
                "sources": payload,
            }
        )
    )
    # decision_date = through + 1 day to recreate the same window with lookback=7
    cut = Cutoff(decision_date=date(2026, 4, 24))
    out = src.fetch({"query": "apple earnings", "lookback_days": 7}, cut)
    assert [g["url"] for g in out["sources"]["grounding"]["generic"]] == [
        in_url,
        undated_url,
    ]

    # Off: full cached payload reaches the caller unchanged.
    raw = WebSearch(cache_root=tmp_path, clamp_by_age=False).fetch(
        {"query": "apple earnings", "lookback_days": 7}, cut
    )
    assert len(raw["sources"]["grounding"]["generic"]) == 3


def test_web_search_catalog_exposes_clamp_by_age():
    info = catalog.source("web_search")
    names = {p.name: p for p in info.params}
    assert names["clamp_by_age"].dtype == "bool"
    assert names["clamp_by_age"].default is True
    assert names["clamp_by_age"].per_call is False


# ── the whole library builds together ────────────────────────────────────────


def test_a_strategy_can_activate_every_kind_at_once(tmp_path):
    """The end state the strategy layer needs: pick a subset, get it wired."""
    fixtures.register_all()
    bindings = [
        DataBinding(kind="prices", source="flat_prices", price=30.0),
        DataBinding(kind="fundamentals", source="annual_fundamentals"),
        DataBinding(kind="news", source="massive_news"),
        DataBinding(kind="ratios", source="valuation_ratios"),
        DataBinding(kind="news_sentiment", source="news_sentiment"),
        DataBinding(kind="web_search", source="web_search"),
        DataBinding(kind="filing_text", source="massive_filing_text"),
    ]
    assert catalog.check_bindings(bindings) == []
    built = build_data_sources(bindings, config=MarketConfig(cache_root=tmp_path, offline=True))
    assert set(built) == {
        "prices",
        "fundamentals",
        "news",
        "ratios",
        "news_sentiment",
        "web_search",
        "filing_text",
    }
    for kind, source in built.items():
        assert kind in source.kinds
