"""Universes, decision grids, and data — the market side of the platform.

The user configures this once (`MarketConfig`: cache root, credentials); a
strategy only names what it wants to activate.
"""

from fintel.market.calendar import TradingCalendar
from fintel.market.constituents import HistoricalUniverse, historical_universe
from fintel.market.factory import build_schedule, build_universe
from fintel.market.schedule import CustomDates, Quarterly, Schedule, SinglePoint
from fintel.market.settings import MarketConfig
from fintel.market.universe import StaticUniverse, Universe

__all__ = [
    "CustomDates",
    "HistoricalUniverse",
    "MarketConfig",
    "Quarterly",
    "Schedule",
    "SinglePoint",
    "StaticUniverse",
    "TradingCalendar",
    "Universe",
    "build_schedule",
    "build_universe",
    "historical_universe",
]
