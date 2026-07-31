"""Universes, decision grids, and data — the market side of the platform.

The user configures this once (`MarketConfig`: cache root, credentials). A
strategy names what it wants to activate, picked from `catalog`.
"""

from fintel.market import catalog
from fintel.market.calendar import TradingCalendar
from fintel.market.catalog import Field, Param, SourceInfo, UniverseInfo
from fintel.market.constituents import HistoricalUniverse, historical_universe
from fintel.market.factory import (
    build_data_source,
    build_data_sources,
    build_schedule,
    build_universe,
)
from fintel.market.realized import PriceLookup
from fintel.market.schedule import CustomDates, Quarterly, Schedule, SinglePoint
from fintel.market.settings import MarketConfig
from fintel.market.universe import StaticUniverse, Universe

__all__ = [
    "CustomDates",
    "Field",
    "HistoricalUniverse",
    "MarketConfig",
    "Param",
    "PriceLookup",
    "Quarterly",
    "Schedule",
    "SinglePoint",
    "SourceInfo",
    "StaticUniverse",
    "TradingCalendar",
    "Universe",
    "UniverseInfo",
    "build_data_source",
    "build_data_sources",
    "build_schedule",
    "build_universe",
    "catalog",
    "historical_universe",
]
