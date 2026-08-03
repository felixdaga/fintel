"""Run-level reachability probe — classifies ok / empty / failed per kind.

The probe calls `source.fetch` directly (the market primitive), so these
tests use tiny fake sources that return data, return nothing, or raise —
covering the three reachability outcomes without touching the network.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Any

from fintel.market.data.base import DataError
from fintel.market.probe import probe
from fintel.pit import Cutoff


class _FakeSource:
    """A DataSource stub with a fixed behaviour."""

    def __init__(self, name: str, kind: str, behaviour: str, payload: Any = None):
        self.name = name
        self.kinds = (kind,)
        self._behaviour = behaviour
        self._payload = payload

    def fetch(self, query: dict, cutoff: Cutoff) -> Any:
        if self._behaviour == "ok":
            return self._payload if self._payload is not None else [{"x": 1}]
        if self._behaviour == "empty":
            return []
        if self._behaviour == "empty_df":
            import pandas as pd

            return pd.DataFrame()
        if self._behaviour == "data_error":
            raise DataError("401 unauthorized")
        if self._behaviour == "boom":
            raise RuntimeError("adapter blew up")
        raise AssertionError(f"unknown behaviour {self._behaviour!r}")


def test_probe_ok_when_all_kinds_return_data():
    sources = {
        "prices": _FakeSource("stub_prices", "prices", "ok", [{"close": 1.0}]),
        "news": _FakeSource("stub_news", "news", "ok", [{"title": "t"}]),
    }
    result = probe(sources=sources, symbol="AAPL", cutoff=Date(2025, 1, 2))
    assert result.ok
    assert [k.kind for k in result.kinds] == ["news", "prices"]  # sorted
    assert all(k.status == "ok" for k in result.kinds)
    assert all(k.reachable for k in result.kinds)
    assert result.failed_kinds == []


def test_probe_empty_counts_as_reachable():
    # A source that answers with nothing is still reachable — it just has no
    # data for that symbol. That is NOT a gate failure; a real cell would
    # record an `empty` read, not a `failed` one.
    sources = {
        "news": _FakeSource("stub_news", "news", "empty"),
        "prices": _FakeSource("stub_prices", "prices", "empty_df"),
    }
    result = probe(sources=sources)
    assert result.ok, "empty reads are reachable, not failures"
    assert all(k.status == "empty" for k in result.kinds)


def test_probe_failed_is_not_reachable_and_does_not_raise():
    sources = {
        "prices": _FakeSource("stub_prices", "prices", "ok"),
        "news": _FakeSource("stub_news", "news", "data_error"),
    }
    result = probe(sources=sources)
    assert not result.ok
    failed = result.failed_kinds
    assert [k.kind for k in failed] == ["news"]
    assert failed[0].status == "failed"
    assert "401 unauthorized" in failed[0].detail


def test_probe_adapter_bug_is_failed_not_crash():
    # An unrecognised exception is a probe failure, never raised — the caller
    # decides whether to gate the run.
    sources = {"prices": _FakeSource("stub_prices", "prices", "boom")}
    result = probe(sources=sources)
    assert not result.ok
    assert result.kinds[0].status == "failed"
    assert "RuntimeError" in result.kinds[0].detail


def test_probe_unbound_kind_is_failed():
    sources = {"prices": _FakeSource("stub_prices", "prices", "ok")}
    # Ask for a kind that has no source bound.
    result = probe(sources=sources, kinds=["prices", "news"])
    news = next(k for k in result.kinds if k.kind == "news")
    assert news.status == "failed"
    assert "not bound" in news.detail


def test_probe_to_dict_round_trips():
    sources = {"prices": _FakeSource("stub_prices", "prices", "ok", [{"close": 1.0}])}
    d = probe(sources=sources).to_dict()
    assert d["ok"] is True
    assert d["kinds"][0]["kind"] == "prices"
    assert d["kinds"][0]["status"] == "ok"
    assert d["symbol"] == "AAPL"  # fallback symbol when none passed


def test_probe_uses_kind_declared_lookback_not_the_flat_default():
    """A quarterly kind (fundamentals) declares a 730d lookback on its spec.
    The probe must use that, not the 30d default — otherwise a 30d window reads
    "empty" for fundamentals even when the source is reachable."""
    seen: dict[str, int] = {}

    class _SpecSource:
        name = "stub_fundamentals"

        class spec:
            lookback_days = 730

        def fetch(self, query: dict, cutoff: Cutoff) -> Any:
            seen["lookback_days"] = int(query["lookback_days"])
            return [{"form": "10-K"}]

    sources = {"fundamentals": _SpecSource()}
    probe(sources=sources, symbol="NVDA", cutoff=Date(2026, 4, 1))
    assert seen["lookback_days"] == 730

