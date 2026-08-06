"""Run-level reachability probe: can each declared source actually return data?

Preflight checks that the declared world is *resolvable* (bindings exist, env
vars set, files present) without fetching — that's its point: cheap, no network.
But "resolvable" is not "reachable": a source can be declared and keyed yet
401 on every call, or the API can be down. A run that discovers this on the
third cell has already paid for two cells of LLM time.

The probe runs once at the run level, before any cell, and answers one question
per kind: "does this source return *anything* for one symbol?" It calls the
same market primitive a cell's read path ultimately calls — `source.fetch(query,
cutoff)` — so it exercises the real fetch, the real cache, the real error
classification. There is no second fetch implementation to drift; the only
thing skipped is the environment's policy/recording wrapper, which a
connectivity check has no use for.

A probe read that comes back with data (`ok`) or no data (`empty`) proves
reachability (the source answered; it just may have nothing for that symbol).
A `failed` read means the source is unreachable — the run is stopped at the
gate, not three cells in.

Layer note: this module lives in `market` (L3) and depends only on `pit`
(L2, for `Cutoff`) and `market.data.base` (L3) — it does not reach into
`environment`. The environment's `DataAccess.read` wraps `source.fetch` with
policy + recording for a real cell; the probe needs neither, so it calls the
primitive directly. Both paths hit the same `source.fetch`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

from fintel.market.data.base import DataError, DataSource
from fintel.market.prefetch import _lookback_days
from fintel.models.common import Symbol
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

# A connectivity token for the rare case the universe can't be resolved
# (offline run against a cache that has nothing yet). The probe still
# distinguishes "reachable" from "unreachable" against any symbol — an
# unreachable source 401s the same way for AAPL as for a real member.
_FALLBACK_SYMBOL: Symbol = "AAPL"


@dataclass
class KindProbe:
    """One kind's probe result."""

    kind: str
    source: str
    status: str  # ok | empty | failed
    latency_ms: float = 0.0
    n: int | None = None
    detail: str = ""

    @property
    def reachable(self) -> bool:
        return self.status in ("ok", "empty")


@dataclass
class ProbeResult:
    """All kinds' probe results, plus whether the run may proceed."""

    symbol: Symbol
    kinds: list[KindProbe] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return all(k.reachable for k in self.kinds)

    @property
    def failed_kinds(self) -> list[KindProbe]:
        return [k for k in self.kinds if not k.reachable]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "kinds": [
                {
                    "kind": k.kind,
                    "source": k.source,
                    "status": k.status,
                    "latency_ms": round(k.latency_ms, 2),
                    "n": k.n,
                    "detail": k.detail,
                }
                for k in self.kinds
            ],
        }


def probe(
    *,
    sources: dict[str, DataSource],
    kinds: list[str] | None = None,
    symbol: Symbol | None = None,
    cutoff: Date | None = None,
    lookback_days: int = 30,
) -> ProbeResult:
    """Probe every bound kind once, via the real fetch primitive.

    Args:
        sources:    the run's data sources (kind -> DataSource).
        kinds:       kinds to probe; default = all bound kinds, sorted.
        symbol:      probe symbol; default = `_FALLBACK_SYMBOL`. Any symbol works
                     for reachability — a `failed` read is unreachable regardless.
        cutoff:      PIT cutoff; default = today. The probe is run-level, so today
                     is fine — it only tests reachability, not a real cell's window.
        lookback_days: fallback window (days) for kinds that don't declare their
                     own lookback. Kinds that declare one (e.g. fundamentals=730d)
                     use that, so a quarterly kind has a filing in-window instead
                     of reading "empty" under a too-narrow 30d window.

    Returns a `ProbeResult`; never raises — a `failed` kind is a result, not an
    exception. The caller decides whether `result.ok` gates the run.
    """
    probe_symbol = symbol or _FALLBACK_SYMBOL
    probe_cutoff = Cutoff(cutoff or Date.today())
    kinds = list(kinds) if kinds is not None else sorted(sources)
    started = time.perf_counter()

    out: list[KindProbe] = []
    for kind in kinds:
        source = sources.get(kind)
        if source is None:
            out.append(
                KindProbe(
                    kind=kind,
                    source="",
                    status="failed",
                    detail=f"kind {kind!r} not bound to a source",
                )
            )
            continue
        # Use the kind's own declared lookback (e.g. fundamentals=730d) so a
        # quarterly kind actually has a filing in-window — a flat 30d window
        # would read "empty" for fundamentals even when the source is perfectly
        # reachable. Falls back to the caller's lookback_days for kinds that
        # don't declare one (e.g. daily prices).
        lb = _lookback_days(kind, source, None) or lookback_days
        # Subject follows the catalog: most kinds key on symbol; web_search
        # keys on a free-text query. Passing the wrong one is a probe false
        # failure, not an unreachable source.
        query = _probe_query(kind, probe_symbol, lb)
        t0 = time.perf_counter()
        try:
            data = source.fetch(query, probe_cutoff)
            status = "empty" if _is_empty(data) else "ok"
            detail = ""
            n = _count(data)
        except DataError as exc:
            status, detail, n = "failed", str(exc), None
            logger.warning("probe: %s (%s) unreachable: %s", kind, source.name, exc)
        except Exception as exc:  # noqa: BLE001 - an adapter bug is a probe failure
            status, detail, n = "failed", f"{type(exc).__name__}: {exc}", None
            logger.exception("probe: %s (%s) raised", kind, source.name)
        out.append(
            KindProbe(
                kind=kind,
                source=getattr(source, "name", ""),
                status=status,
                latency_ms=(time.perf_counter() - t0) * 1000,
                n=n,
                detail=detail,
            )
        )

    return ProbeResult(
        symbol=probe_symbol,
        kinds=out,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _probe_query(kind: str, symbol: Symbol, lookback_days: int) -> dict:
    """Build the minimal fetch query the catalog's subject requires."""
    from fintel.market import catalog

    subject = "symbol"
    for info in catalog.sources(kind=kind):
        subject = info.subject
        break
    query: dict[str, Any] = {"lookback_days": lookback_days}
    if subject == "query":
        # A connectivity token — any short string exercises the provider + cache.
        query["query"] = f"{symbol} stock"
    elif subject == "symbol":
        query["symbol"] = symbol
    # subject == "none": no identity key
    return query


def _is_empty(data: Any) -> bool:
    """Mirror of `environment.access._is_empty`, kept local so the probe stays
    in the market layer (no environment import). A dict with any populated
    value is not empty; a DataFrame with `.empty` is empty; None is empty."""
    if data is None:
        return True
    if isinstance(data, dict):
        substantive = {k: v for k, v in data.items() if k not in ("as_of", "notes")}
        return not any(v not in (None, [], {}, "") for v in substantive.values())
    if hasattr(data, "empty"):  # DataFrame
        return bool(data.empty)
    if isinstance(data, (list, tuple, set, str)):
        return len(data) == 0
    return False


def _count(data: Any) -> int | None:
    if data is None:
        return 0
    if isinstance(data, (list, tuple, set)):
        return len(data)
    if hasattr(data, "__len__") and not isinstance(data, (str, dict)):
        try:
            return len(data)
        except TypeError:
            return None
    return None
