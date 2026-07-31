"""Which symbols are in play at a date.

Two shapes, and the difference is survivorship bias:

  · `StaticUniverse`     — one frozen list, same at every date.
  · `HistoricalUniverse` — real index membership resolved per date
                           (see constituents.py). Prefer this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Protocol, runtime_checkable

from fintel.models.common import Symbol


@runtime_checkable
class Universe(Protocol):
    name: str

    def active_at(self, as_of: Date) -> list[Symbol]: ...


@dataclass(frozen=True)
class StaticUniverse:
    """A frozen list. `active_at` ignores the date by design.

    Carries survivorship and pre-inclusion bias for any date before the
    snapshot: the names weren't all members then, and the ones that later
    dropped out are missing. Fine for a curated thesis list, wrong for
    measuring an index.
    """

    symbols: tuple[Symbol, ...]
    name: str = "static"
    snapshot_date: Date | None = None

    def active_at(self, as_of: Date) -> list[Symbol]:  # noqa: ARG002 - by design
        return list(self.symbols)


# Frozen snapshots. Named for what they are and when they were true, so a stale
# one is obvious at the call site.
DJIA_2024_11_08: tuple[Symbol, ...] = (
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
)  # fmt: skip

# GOOGL replaced VZ on 2026-06-29.
DJIA_2026_06_29: tuple[Symbol, ...] = tuple(sorted(set(DJIA_2024_11_08) - {"VZ"} | {"GOOGL"}))

# 35 names, GICS-sector-balanced, including Nuclear/Power and Renewables sleeves
# the DJIA omits.
SECTOR35: tuple[Symbol, ...] = (
    "AAPL", "AMT", "AMZN", "AVGO", "BA", "BAC", "CAT", "CCJ", "CVX", "ENPH",
    "FSLR", "GE", "GOOGL", "GS", "HD", "JNJ", "JPM", "LIN", "LLY", "MCD",
    "META", "MRK", "MSFT", "NEE", "NFLX", "NVDA", "ORCL", "PG", "TSLA", "UBER",
    "UNH", "V", "VST", "WMT", "XOM",
)  # fmt: skip

STATIC_PRESETS: dict[str, tuple[tuple[Symbol, ...], Date | None]] = {
    "djia_2024_11_08": (DJIA_2024_11_08, Date(2024, 11, 8)),
    "djia_2026_06_29": (DJIA_2026_06_29, Date(2026, 6, 29)),
    "sector35": (SECTOR35, None),
}


def static_preset(name: str) -> StaticUniverse:
    if name not in STATIC_PRESETS:
        raise ValueError(
            f"unknown static universe preset {name!r}; available: {sorted(STATIC_PRESETS)}"
        )
    symbols, snapshot = STATIC_PRESETS[name]
    return StaticUniverse(symbols=symbols, name=name, snapshot_date=snapshot)


def symbol_universe(symbols: list[Symbol], name: str = "custom") -> StaticUniverse:
    if not symbols:
        raise ValueError("universe symbol list is empty")
    return StaticUniverse(symbols=tuple(dict.fromkeys(symbols)), name=name)
