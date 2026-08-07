"""Factor-neutralized residual IC, NW t, and naive residual NAV.

Ported from Delorean's attribution.py + portfolio.py for the_challenge.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "runs" / "cache"
FACTOR_RUNS = Path(__file__).resolve().parents[1] / "runs" / "challenge_factors"

FACTORS = [
    ("value.earnings_yield", "factor_value_earnings_yield"),
    ("momentum.12_1", "factor_momentum_12_1"),
    ("quality.operating_profitability", "factor_quality_operating_profitability"),
    ("growth.asset_growth", "factor_growth_asset_growth"),
    ("size.market_cap_inverse", "factor_size_market_cap_inverse"),
    ("volatility.low_60", "factor_volatility_low_60"),
    ("reversal.1m", "factor_reversal_1m"),
]
HORIZONS = [1, 2, 3, 4]
ACTIVE_BUDGET = 0.5
IC_METHOD = "pearson"

GICS_SECTOR_DJIA30 = {
    "AAPL": "IT",
    "AMGN": "HC",
    "AMZN": "CD",
    "AXP": "FIN",
    "BA": "IND",
    "CAT": "IND",
    "CRM": "IT",
    "CSCO": "IT",
    "CVX": "EN",
    "DIS": "CS",
    "DOW": "MAT",
    "GOOGL": "CS",
    "GS": "FIN",
    "HD": "CD",
    "HON": "IND",
    "IBM": "IT",
    "INTC": "IT",
    "JNJ": "HC",
    "JPM": "FIN",
    "KO": "CS",
    "MCD": "CD",
    "MMM": "IND",
    "MRK": "HC",
    "MSFT": "IT",
    "NKE": "CD",
    "NVDA": "IT",
    "PG": "CS",
    "SHW": "MAT",
    "TRV": "FIN",
    "UNH": "HC",
    "V": "FIN",
    "VZ": "CS",
    "WMT": "CS",
}

DJIA_RECONSTITUTIONS = [
    # Feb 2024: WBA → AMZN
    (
        date(2024, 2, 28),
        frozenset(
            {
                "AAPL",
                "AMGN",
                "AMZN",
                "AXP",
                "BA",
                "CAT",
                "CRM",
                "CSCO",
                "CVX",
                "DIS",
                "DOW",
                "GS",
                "HD",
                "HON",
                "IBM",
                "INTC",
                "JNJ",
                "JPM",
                "KO",
                "MCD",
                "MMM",
                "MRK",
                "MSFT",
                "NKE",
                "PG",
                "TRV",
                "UNH",
                "V",
                "VZ",
                "WMT",
            }
        ),
    ),
    # Nov 2024: INTC → NVDA, DOW → SHW
    (
        date(2024, 11, 8),
        frozenset(
            {
                "AAPL",
                "AMGN",
                "AMZN",
                "AXP",
                "BA",
                "CAT",
                "CRM",
                "CSCO",
                "CVX",
                "DIS",
                "GS",
                "HD",
                "HON",
                "IBM",
                "JNJ",
                "JPM",
                "KO",
                "MCD",
                "MMM",
                "MRK",
                "MSFT",
                "NKE",
                "NVDA",
                "PG",
                "SHW",
                "TRV",
                "UNH",
                "V",
                "VZ",
                "WMT",
            }
        ),
    ),
]

# Pre-Feb 2024 baseline (WBA instead of AMZN)
_DJIA_PRE_FEB_2024 = DJIA_RECONSTITUTIONS[0][1] - {"AMZN"} | {"WBA"}


def djia_members(as_of: date) -> list[str]:
    for cutoff, m in DJIA_RECONSTITUTIONS:
        if as_of >= cutoff:
            return sorted(m)
    return sorted(_DJIA_PRE_FEB_2024)


def _pd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def price_lookup():
    from fintel.market.data.store import PriceStore
    from fintel.market.realized import PriceLookup

    return PriceLookup(PriceStore(CACHE))


def _zscore_transform(scores: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score (mean 0, std 1) — Delorean's default transform.

    Uses population variance (÷n) to match Delorean's transform.zscore.
    """
    vals = list(scores.values())
    if len(vals) < 2:
        return dict(scores)
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))  # population std, matching Delorean
    if sd == 0:
        return {s: 0.0 for s in scores}
    return {s: (float(v) - mu) / sd for s, v in scores.items()}


def load_run_scores(run_dir: Path) -> dict[date, dict[str, float]]:
    """Load {date: {symbol: score}} from a run dir.

    Supports both fintel layout (trials/<date>/decision.json) and
    Delorean layout (decisions/<date>.json with agent_response.views).
    """
    # Delorean: decisions/<date>.json
    decs = run_dir / "decisions"
    if decs.is_dir():
        out: dict[date, dict[str, float]] = {}
        for f in sorted(decs.iterdir()):
            if not f.name.endswith(".json"):
                continue
            d = _pd(f.stem)
            raw = json.loads(f.read_text())
            views = raw.get("agent_response", {}).get("views", {})
            scores = {
                s: float(v["score"])
                for s, v in views.items()
                if isinstance(v, dict) and "score" in v
            }
            if scores:
                out[d] = scores
        if out:
            return out

    # Fintel: trials/<date>/decision.json
    trials = run_dir / "trials"
    if not trials.is_dir():
        raise FileNotFoundError(f"no trials/ or decisions/ in {run_dir}")
    out = {}
    for td in sorted(trials.iterdir()):
        if not td.is_dir():
            continue
        dec = td / "decision.json"
        if not dec.is_file():
            continue
        d = _pd(td.name)
        raw = json.loads(dec.read_text())
        scores = {
            s: float(v["score"]) for s, v in raw.items() if isinstance(v, dict) and "score" in v
        }
        if scores:
            out[d] = scores
    return out


def ensemble(runs: list[dict[date, dict[str, float]]]) -> dict[date, dict[str, float]]:
    dates = sorted({d for r in runs for d in r})
    out: dict[date, dict[str, float]] = {}
    for d in dates:
        per: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            for s, v in r.get(d, {}).items():
                per[s].append(v)
        out[d] = {s: float(np.mean(v)) for s, v in per.items() if v}
    return out


def load_factor_scores(factor_dir: Path) -> dict[date, dict[str, float]]:
    decs = factor_dir / "decisions"
    out: dict[date, dict[str, float]] = {}
    for f in sorted(decs.iterdir()):
        if not f.name.endswith(".json"):
            continue
        d = _pd(f.stem)
        raw = json.loads(f.read_text())
        views = raw.get("agent_response", {}).get("views", {})
        scores = {
            s: float(v["score"]) for s, v in views.items() if isinstance(v, dict) and "score" in v
        }
        if scores:
            out[d] = scores
    return out
