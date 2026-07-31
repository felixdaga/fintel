"""Stand-ins for third-party extensions, so the `module:Callable` seam is
exercised by something outside the platform rather than assumed to work."""

from __future__ import annotations

from fintel.market.settings import MarketConfig
from fintel.market.universe import StaticUniverse


def tiny_universe(**_: object) -> StaticUniverse:
    """A custom universe needing no platform services stays trivial."""
    return StaticUniverse(symbols=("XYZ",), name="tiny")


def cache_aware_universe(*, config: MarketConfig, tag: str = "") -> StaticUniverse:
    """One that needs the cache root declares `config` and is handed it."""
    return StaticUniverse(symbols=(config.cache_root.name, tag), name="cache_aware")
