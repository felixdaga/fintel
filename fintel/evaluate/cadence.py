"""Decision-grid cadence → periods per year for annualization.

Used by `fintel report` (KPI ICIR and holdings Sharpe/IR/vol/turnover).
Strategy name wins (`biweekly` before `weekly` — substring trap); otherwise
median gap between consecutive decision dates.
"""

from __future__ import annotations

from datetime import date as Date


def detect_cadence(*, name: str = "", dates: list[Date] | None = None) -> dict:
    """Return `{cadence, ppy, median_gap_days, n_dates}`."""
    name_l = (name or "").lower()
    ds = sorted(dates or [])
    med: float | None = None
    if len(ds) >= 2:
        gaps = [(ds[i] - ds[i - 1]).days for i in range(1, len(ds))]
        med = float(sorted(gaps)[len(gaps) // 2])

    if "biweekly" in name_l:
        cadence, ppy = "biweekly", 26.0
    elif "weekly" in name_l:
        cadence, ppy = "weekly", 52.0
    elif "monthly" in name_l:
        cadence, ppy = "monthly", 12.0
    elif "quarter" in name_l:
        cadence, ppy = "quarterly", 4.0
    elif med is not None and 11 <= med <= 18:
        cadence, ppy = "biweekly", 26.0
    elif med is not None and 4 <= med <= 10:
        cadence, ppy = "weekly", 52.0
    elif med is not None and 20 <= med <= 40:
        cadence, ppy = "monthly", 12.0
    elif med is not None and 70 <= med <= 110:
        cadence, ppy = "quarterly", 4.0
    elif med:
        cadence, ppy = f"custom_{med:.0f}d", 365.25 / med
    else:
        cadence, ppy = "unknown", 12.0

    return {
        "cadence": cadence,
        "ppy": ppy,
        "median_gap_days": med,
        "n_dates": len(ds),
    }
