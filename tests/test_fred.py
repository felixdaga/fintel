"""FRED macro: cache-first fills, offline serve, prefetch warm path."""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pytest

from fintel.market.data.base import DataError
from fintel.market.data.fred import DEFAULT_BUNDLE, FredMacro, _resolve_series_id
from fintel.market.data.store import RecordCache
from fintel.market.prefetch import prefetch
from fintel.pit import Cutoff


class _FakeResp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Minimal httpx stand-in: records calls and returns scripted payloads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> _FakeResp:
        path = url.rsplit("/", 1)[-1]
        # series vs series/observations — url ends with the path segment.
        if url.endswith("/series"):
            kind = "series"
        elif url.endswith("/observations"):
            kind = "observations"
        else:
            kind = path
        self.calls.append((kind, dict(params or {})))
        series_id = (params or {}).get("series_id", "X")
        if kind == "series":
            return _FakeResp(
                200,
                {
                    "seriess": [
                        {
                            "id": series_id,
                            "title": f"Title {series_id}",
                            "units_short": "%",
                            "frequency": "Daily",
                            "seasonal_adjustment_short": "NSA",
                        }
                    ]
                },
            )
        # observations
        start = Date.fromisoformat(params["observation_start"])
        end = Date.fromisoformat(params["observation_end"])
        obs = []
        d = start
        while d <= end:
            obs.append({"date": d.isoformat(), "value": str(100 + (d - start).days)})
            d += timedelta(days=1)
        return _FakeResp(200, {"observations": obs})

    def close(self) -> None:
        return None


def _source(tmp_path: Path, *, online: bool = True) -> FredMacro:
    src = FredMacro(
        cache=RecordCache(root=tmp_path, kind="macro"),
        api_key="test-key" if online else None,
        lookback_days=10,
        indicators=("vix",),
    )
    if online:
        src._client = _FakeClient()  # noqa: SLF001
    return src


def test_resolve_alias_and_raw_id():
    assert _resolve_series_id("vix") == "VIXCLS"
    assert _resolve_series_id("FEDFUNDS") == "FEDFUNDS"
    with pytest.raises(ValueError):
        _resolve_series_id("not a valid series phrase")


def test_fetch_writes_cache_and_second_call_is_hit(tmp_path: Path):
    src = _source(tmp_path, online=True)
    cutoff = Cutoff(decision_date=Date(2026, 4, 24))
    out1 = src.fetch({}, cutoff)
    assert len(out1["series"]) == 1
    assert out1["series"][0]["series_id"] == "VIXCLS"
    assert out1["series"][0]["observations"]
    assert (tmp_path / "macro" / "VIXCLS.json").exists()
    assert (tmp_path / "macro" / "VIXCLS.meta.json").exists()

    n_calls = len(src._client.calls)  # noqa: SLF001
    out2 = src.fetch({}, cutoff)
    assert out2["series"][0]["latest_value"] == out1["series"][0]["latest_value"]
    assert len(out2["series"][0]["observations"]) == len(out1["series"][0]["observations"])
    assert len(src._client.calls) == n_calls  # noqa: SLF001 — no new network


def test_offline_serves_cache(tmp_path: Path):
    online = _source(tmp_path, online=True)
    cutoff = Cutoff(decision_date=Date(2026, 4, 24))
    warm = online.fetch({}, cutoff)
    offline = _source(tmp_path, online=False)
    out = offline.fetch({}, cutoff)
    assert out["series"][0]["series_id"] == warm["series"][0]["series_id"]
    assert out["series"][0]["observations"] == warm["series"][0]["observations"]


def test_offline_miss_raises(tmp_path: Path):
    src = _source(tmp_path, online=False)
    with pytest.raises(DataError, match="nothing cached"):
        src.fetch({}, Cutoff(decision_date=Date(2026, 4, 24)))


def test_prefetch_warms_macro_once(tmp_path: Path):
    src = _source(tmp_path, online=True)
    # Widen indicators so warm covers the configured set.
    src.indicators = ("vix", "fed_funds_rate")
    result = prefetch(
        symbols=["AAPL", "MSFT"],  # universe — macro should still warm once
        sources={"macro": src},
        from_date=Date(2026, 4, 1),
        through_date=Date(2026, 4, 23),
        workers=2,
    )
    assert "*:macro" in result.warmed
    assert result.failed == {}
    assert (tmp_path / "macro" / "VIXCLS.json").exists()
    assert (tmp_path / "macro" / "FEDFUNDS.json").exists()
    # Warm should not multiply by universe size.
    assert result.warmed.count("*:macro") == 1


def test_default_bundle_non_empty():
    assert len(DEFAULT_BUNDLE) >= 5
    for alias in DEFAULT_BUNDLE:
        assert _resolve_series_id(alias)
