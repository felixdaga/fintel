"""Name → object. The only place market components get constructed.

A strategy points at a universe / schedule / data source by name; anything not
builtin resolves as `module:Callable`. A custom callable receives the declared
params, plus `config=` if it asks for one — so a trivial universe stays trivial
while a cache-backed one still gets the platform's cache root.
"""

from __future__ import annotations

import inspect
from datetime import date as Date
from datetime import datetime
from typing import Any

from fintel.market.calendar import TradingCalendar
from fintel.market.constituents import INDEX_PRESETS, historical_universe
from fintel.market.schedule import CustomDates, Quarterly, Schedule, SinglePoint
from fintel.market.settings import MarketConfig
from fintel.market.universe import STATIC_PRESETS, Universe, static_preset, symbol_universe
from fintel.models.market import ScheduleRef, UniverseRef
from fintel.utils.import_path import resolve

SCHEDULES: dict[str, str] = {
    "single_point": "fintel.market.schedule:SinglePoint",
    "custom_dates": "fintel.market.schedule:CustomDates",
    "quarterly": "fintel.market.schedule:Quarterly",
}


def as_date(value: Any) -> Date:
    """TOML gives native dates for bare literals and strings for quoted ones."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    if isinstance(value, str):
        return Date.fromisoformat(value[:10])
    raise TypeError(f"cannot read a date from {value!r}")


def _call(target: str, params: dict, config: MarketConfig | None) -> Any:
    fn = resolve(target)
    if config is not None:
        try:
            accepts = "config" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            return fn(config=config, **params)
    return fn(**params)


def build_universe(ref: UniverseRef, *, config: MarketConfig) -> Universe:
    if ref.symbols:
        return symbol_universe(list(ref.symbols))
    if ref.preset:
        if ref.preset in INDEX_PRESETS:
            return historical_universe(ref.preset, config=config)
        if ref.preset in STATIC_PRESETS:
            return static_preset(ref.preset)
        raise ValueError(
            f"unknown universe preset {ref.preset!r}; index presets "
            f"{sorted(INDEX_PRESETS)}, static presets {sorted(STATIC_PRESETS)}"
        )
    if ref.source:
        return _call(ref.source, ref.params, config)
    raise ValueError("universe needs one of: preset, symbols, source")


def build_schedule(ref: ScheduleRef, *, calendar: TradingCalendar | None = None) -> Schedule:
    cal = calendar or TradingCalendar()
    params = ref.params
    start = as_date(params["start"]) if params.get("start") else None
    end = as_date(params["end"]) if params.get("end") else None

    if ref.kind == "single_point":
        if "on" not in params:
            raise ValueError("single_point schedule needs `on`, the decision date")
        return SinglePoint(on=as_date(params["on"]))
    if ref.kind == "custom_dates":
        raw = params.get("dates") or []
        if not raw:
            raise ValueError("custom_dates schedule needs a non-empty `dates` list")
        return CustomDates(dates_=tuple(as_date(d) for d in raw), start=start, end=end)
    if ref.kind == "quarterly":
        return Quarterly(start=start, end=end, calendar=cal)
    if ":" in ref.kind:
        return resolve(ref.kind)(**params)
    raise ValueError(f"unknown schedule kind {ref.kind!r}; available: {sorted(SCHEDULES)}")
