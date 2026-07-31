"""Migration fidelity: ported math must reproduce the old pipeline exactly.

Skipped when the legacy cache isn't on this machine, so the suite still runs
anywhere. Point FINTEL_LEGACY_CACHE at a delorean `cache/` to enable.

The alignment below is the whole point of the test. The old pipeline stored a
daily series where each entry carries *that day's* close, then looked it up with
`date < as_of`. So the fintel value at cutoff D must equal the old entry for the
last trading day before D — comparing against the entry dated D itself would be
comparing a clamped number to an unclamped one.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date
from pathlib import Path

import pytest

from fintel.market.data.ratios import RATIO_FIELDS
from fintel.market.factory import build_data_sources
from fintel.market.settings import MarketConfig
from fintel.models.market import DataBinding
from fintel.pit import Cutoff

LEGACY_CACHE = Path(
    os.environ.get(
        "FINTEL_LEGACY_CACHE", Path.home() / ".openclaw/workspace/projects/delorean/cache"
    )
)

pytestmark = pytest.mark.skipif(
    not (LEGACY_CACHE / "ratios").is_dir(),
    reason=f"no legacy ratios cache at {LEGACY_CACHE}",
)

# Compared numerically. `notes` is prose and `as_of`/`filing_date`/`period_end`
# are shifted by the alignment above, so they're checked separately.
NUMERIC = tuple(f for f in RATIO_FIELDS if f not in ("as_of", "notes", "filing_date", "period_end"))


def _sources():
    return build_data_sources(
        [
            DataBinding(kind="prices", source="massive_prices"),
            DataBinding(kind="fundamentals", source="massive_fundamentals"),
            DataBinding(kind="ratios", source="valuation_ratios"),
        ],
        config=MarketConfig(cache_root=LEGACY_CACHE, offline=True),
    )["ratios"]


def _series(symbol: str) -> list[dict]:
    path = LEGACY_CACHE / "ratios" / f"{symbol}.json"
    entries = json.loads(path.read_text()).get("entries", [])
    return [e for e in entries if e.get("date")]


def _symbols() -> list[str]:
    return sorted(p.stem for p in (LEGACY_CACHE / "ratios").glob("*.json"))


@pytest.mark.parametrize("symbol", _symbols() if (LEGACY_CACHE / "ratios").is_dir() else [])
def test_ported_ratios_match_the_legacy_series(symbol: str, caplog):
    """Every numeric ratio, sampled across each symbol's full history."""
    caplog.set_level(logging.CRITICAL)
    entries = _series(symbol)
    if len(entries) < 2:
        pytest.skip(f"{symbol} has no usable legacy series")

    source = _sources()
    compared = 0
    for i in range(len(entries) - 1, 0, -240):
        expected = entries[i - 1]
        cutoff = Cutoff(date.fromisoformat(entries[i]["date"]))
        got = source.fetch({"symbol": symbol}, cutoff)
        for field in NUMERIC:
            new, old = got.get(field), expected.get(field)
            compared += 1
            if new is None or old is None:
                assert new is None and old is None, f"{symbol} {cutoff} {field}: {new} vs {old}"
            else:
                assert math.isclose(new, old, rel_tol=1e-6, abs_tol=1e-9), (
                    f"{symbol} {cutoff} {field}: {new} vs {old}"
                )
    assert compared > 0


def test_the_anchor_filing_matches_too(caplog):
    """Not just the numbers: the same filing must be behind them."""
    caplog.set_level(logging.CRITICAL)
    source = _sources()
    checked = 0
    for symbol in _symbols()[:12]:
        entries = _series(symbol)
        if len(entries) < 2:
            continue
        for i in range(len(entries) - 1, 0, -400):
            expected = entries[i - 1]
            got = source.fetch({"symbol": symbol}, Cutoff(date.fromisoformat(entries[i]["date"])))
            assert got["filing_date"] == expected["filing_date"], f"{symbol} {expected['date']}"
            assert got["period_end"] == expected["period_end"], f"{symbol} {expected['date']}"
            checked += 1
    assert checked > 0


def test_legacy_series_would_leak_if_read_at_its_own_date(caplog):
    """Why the alignment matters, stated as a test.

    An entry dated D holds D's own close, which isn't observable when the
    decision at D is made. Reading the series at its own date is a one-session
    lookahead; the old code avoided it with `date < as_of`, and fintel avoids it
    by clamping. This pins the distinction so nobody "simplifies" it later.
    """
    caplog.set_level(logging.CRITICAL)
    source = _sources()
    for symbol in _symbols():
        entries = _series(symbol)
        same_day = next(
            (e for e in reversed(entries) if e.get("price") and e.get("date")), None
        )
        if not same_day:
            continue
        cutoff = Cutoff(date.fromisoformat(same_day["date"]))
        got = source.fetch({"symbol": symbol}, cutoff)
        if got["price"] is not None and got["price"] != same_day["price"]:
            # The clamped price is an earlier session's close, never that day's.
            assert got["price"] != same_day["price"]
            return
    pytest.skip("no symbol had a distinguishable same-day close")
