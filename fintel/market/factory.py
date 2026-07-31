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

from fintel.market import catalog
from fintel.market.calendar import TradingCalendar
from fintel.market.constituents import INDEX_PRESETS, historical_universe
from fintel.market.data.base import DataSource
from fintel.market.data.http import MassiveClient
from fintel.market.data.massive import FUNDAMENTALS, NEWS, MassivePrices, MassiveRecords
from fintel.market.data.store import PriceStore, RecordCache
from fintel.market.data.synthetic import SyntheticPrices
from fintel.market.schedule import CustomDates, Quarterly, Schedule, SinglePoint
from fintel.market.settings import MarketConfig
from fintel.market.universe import STATIC_PRESETS, Universe, static_preset, symbol_universe
from fintel.models.market import DataBinding, ScheduleRef, UniverseRef
from fintel.utils.import_path import resolve

SCHEDULES: dict[str, str] = {
    "single_point": "fintel.market.schedule:SinglePoint",
    "custom_dates": "fintel.market.schedule:CustomDates",
    "quarterly": "fintel.market.schedule:Quarterly",
}

# Data sources are not listed here — `catalog` owns that registry so the library
# a user browses and the objects the platform builds cannot drift apart.
catalog.register_builtins()


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


# ── data sources ─────────────────────────────────────────────────────────────
# One client per job, so retries and request counts are shared rather than
# re-established per source. None means "cache only".


def _client(config: MarketConfig) -> MassiveClient | None:
    if config.offline:
        return None
    key = config.require_key("MASSIVE_API_KEY", config.massive_api_key)
    return MassiveClient(key)


def massive_prices(*, config: MarketConfig, **params: Any) -> MassivePrices:
    return MassivePrices(
        store=PriceStore(root=config.cache_root),
        client=_client(config),
        **{k: v for k, v in params.items() if k in {"adjusted", "history_start"}},
    )


def massive_fundamentals(*, config: MarketConfig, **_: Any) -> MassiveRecords:
    return MassiveRecords(
        spec=FUNDAMENTALS,
        cache=RecordCache(root=config.cache_root, kind="fundamentals"),
        client=_client(config),
        name="massive_fundamentals",
    )


def massive_news(*, config: MarketConfig, **_: Any) -> MassiveRecords:
    return MassiveRecords(
        spec=NEWS,
        cache=RecordCache(root=config.cache_root, kind="news"),
        client=_client(config),
        name="massive_news",
    )


def synthetic_prices(**params: Any) -> SyntheticPrices:
    return SyntheticPrices(**{k: v for k, v in params.items() if k in {"base_price", "daily_vol"}})


def build_data_source(
    binding: DataBinding,
    *,
    config: MarketConfig,
    upstream: dict[str, DataSource] | None = None,
) -> DataSource:
    """One source. A catalog name, or `module:Callable` for anything unregistered.

    The result must serve `binding.kind`, so a manifest that points a kind at a
    source which doesn't produce it fails here rather than when an agent asks a
    question and silently gets nothing back.
    """
    if catalog.has_source(binding.source):
        info = catalog.source(binding.source)
        target = info.target
        if info.kind != binding.kind:
            raise ValueError(
                f"source {binding.source!r} serves kind {info.kind!r}, "
                f"but the manifest binds it to {binding.kind!r}"
            )
        unknown = set(binding.params) - {p.name for p in info.params}
        if unknown:
            raise ValueError(
                f"source {binding.source!r} does not accept {sorted(unknown)}; "
                f"accepted params: {sorted(p.name for p in info.params)}"
            )
    else:
        target = binding.source
        if ":" not in target:
            raise ValueError(
                f"unknown data source {binding.source!r}; registered: "
                f"{[s.name for s in catalog.sources()]}, or give an import path"
            )

    params = dict(binding.params)
    if upstream:
        params["upstream"] = upstream
    source = _call(target, params, config)

    served = tuple(getattr(source, "kinds", ()))
    if binding.kind not in served:
        raise ValueError(
            f"data source {binding.source!r} serves {list(served)}, "
            f"but the manifest binds it to kind {binding.kind!r}"
        )
    return source


def build_data_sources(
    bindings: list[DataBinding], *, config: MarketConfig
) -> dict[str, DataSource]:
    """Every declared kind, computed ones last with their upstreams injected.

    A computed source's upstream kinds must be bound explicitly by the same
    manifest. Silently defaulting them would mean a package's ratios quietly
    changed provider because the platform picked a different price source.
    """
    plain = [b for b in bindings if not _computed_upstreams(b)]
    computed = [b for b in bindings if _computed_upstreams(b)]

    built: dict[str, DataSource] = {}
    for binding in plain:
        built[binding.kind] = build_data_source(binding, config=config)

    for binding in computed:
        needs = _computed_upstreams(binding)
        absent = [k for k in needs if k not in built]
        if absent:
            raise ValueError(
                f"source {binding.source!r} for kind {binding.kind!r} is computed from "
                f"{list(needs)}; add a [[data]] block for {absent} to this manifest"
            )
        built[binding.kind] = build_data_source(
            binding, config=config, upstream={k: built[k] for k in needs}
        )
    return built


def _computed_upstreams(binding: DataBinding) -> tuple[str, ...]:
    if not catalog.has_source(binding.source):
        return ()
    return catalog.source(binding.source).derives_from
