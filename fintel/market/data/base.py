"""The data contract.

`fetch` takes a `Cutoff`, never a bare date. A source cannot be called without
the point-in-time boundary, and the boundary cannot be mistaken for an ordinary
parameter. The unclamped price path for scoring lives in `market/realized.py`
under a different name for the same reason — see docs/architecture.md.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fintel.pit import Cutoff


class DataError(RuntimeError):
    """A fetch could not be answered. Distinct from 'answered, and empty'.

    The old client returned `{}` for auth and entitlement failures, so a bad key
    or an out-of-plan date range looked exactly like a symbol with no news.
    """


class EntitlementError(DataError):
    """The provider refused the requested window (plan doesn't cover it)."""


@runtime_checkable
class DataSource(Protocol):
    name: str
    kinds: tuple[str, ...]

    def fetch(self, query: dict, cutoff: Cutoff) -> Any: ...


def require(query: dict, key: str, source: str) -> Any:
    if key not in query:
        raise DataError(f"{source}: query is missing required key {key!r}; got {sorted(query)}")
    return query[key]
