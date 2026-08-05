"""The one agent-facing data path.

Every read an agent performs goes through `DataAccess.read`. That matters for two
reasons the old design learned the hard way.

**PIT is enforced once.** The old code implemented the clamp in the data source,
again in the bundle builder, again per-tool in the MCP server, again in the
dossier renderers, and again in the memory store — five implementations that
disagreed. `get_fundamentals` used `filing_date <= decision_date` where the source
used `<`; `get_filings` had no upper bound at all and trusted its input; the
bundled price path served pre-frozen bars without re-checking; a `web_search`
cache hit skipped the check its live path performed. Here the cutoff comes from
the cell, is injected by this module, and no caller can pass one — so there is no
second implementation to drift.

**Absence and failure are different answers.** The old tools returned `[]` for a
missing API key, a source that raised, an unconfigured provider, and a company
with genuinely no news. An agent cannot tell those apart, and neither can a
reviewer reading the trace afterwards. Every read here returns a `Reading` whose
status says which happened.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from fintel.environment.cell import Cell
from fintel.environment.policy import AccessDenied, AccessPolicy
from fintel.market.data.base import DataError, DataSource
from fintel.market.render import estimate_tokens, fetched_chars, predict_capped_chars
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

ReadingStatus = Literal["ok", "empty", "failed", "denied"]

# What a source may be asked that isn't a declared param. `symbol` and `query`
# identify the subject of a fetch; the rest is the source's own vocabulary.
SUBJECT_KEYS = ("symbol", "query")


def _cache_key(kind: str, query: dict) -> str:
    return kind + "|" + json.dumps(query, sort_keys=True, default=str)


class DedupSource:
    """Single-flight proxy over a shared ``DataSource``.

    The wrapped source is shared across every cell in a run (built once in
    ``run_run`` and threaded into each ``DataAccess``). When N concurrent cells
    issue the *same* ``(query, cutoff)`` fetch — a thundering herd, classically
    the symbol-independent macro bundle fetched once per cell — only the first
    call hits the network; the rest block on the same in-flight result and share
    it. Symbol-specific kinds (prices, fundamentals, …) have distinct queries
    per cell, so they are unaffected — each cell still fetches its own.

    Only *in-flight* calls are deduped. The key is dropped the instant the fetch
    finishes (success or failure), so a later, non-concurrent call fetches
    fresh. A transient failure is therefore never made sticky — this is herd
    protection, not a result cache. Per-cell memoization (``DataAccess._cache``)
    still handles repeated reads *within* one cell; this handles the cross-cell
    burst the per-cell cache structurally cannot see.
    """

    def __init__(self, inner: DataSource, lock: threading.Lock, inflight: dict) -> None:
        # Bypass __getattr__ during init so the private attrs land in __dict__.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_inflight", inflight)

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", "")

    @property
    def kinds(self) -> tuple[str, ...]:
        return getattr(self._inner, "kinds", ())

    def __getattr__(self, attr: str) -> Any:
        # Forward any other attribute (lookback_days, spec, …) to the inner
        # source. __getattr__ runs only when normal lookup fails, so the
        # properties above and _inner/_lock/_inflight never recurse here.
        return getattr(self._inner, attr)

    def fetch(self, query: dict, cutoff: Cutoff) -> Any:
        key = json.dumps(query, sort_keys=True, default=str) + "|" + cutoff.decision_date.isoformat()
        with self._lock:
            fut = self._inflight.get(key)
            lead = fut is None
            if lead:
                fut = Future()
                self._inflight[key] = fut
        if lead:
            try:
                fut.set_result(self._inner.fetch(query, cutoff))
            except BaseException as exc:  # noqa: BLE001 - waiters must see it
                fut.set_exception(exc)
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
        return fut.result()


def dedup_sources(sources: dict[str, DataSource]) -> dict[str, DedupSource]:
    """Wrap every source behind one shared single-flight layer.

    One lock and one in-flight map for the whole run, so every cell dedups
    against the same table. Applied at the run level (``run_run``) so all cell
    reads go through it regardless of which kind or source.
    """
    lock = threading.Lock()
    inflight: dict[str, Future] = {}
    return {k: DedupSource(s, lock, inflight) for k, s in sources.items()}


def _is_empty(data: Any) -> bool:
    """Whether a source answered with nothing.

    Deliberately conservative: a dict that carries any populated value is not
    empty, because computed kinds return a fixed key set where most values are
    legitimately None (a ratio with no filing behind it still reports `notes`).
    """
    if data is None:
        return True
    if isinstance(data, dict):
        # History wrappers (ratios.entries, news_sentiment.series) count even when
        # top-level scalar fields are mostly None.
        if data.get("entries") or data.get("series"):
            return False
        substantive = {k: v for k, v in data.items() if k not in ("as_of", "notes", "date")}
        return not any(v not in (None, [], {}, "") for v in substantive.values())
    if hasattr(data, "empty"):  # DataFrame
        return bool(data.empty)
    if isinstance(data, (list, tuple, set, str)):
        return len(data) == 0
    return False


@dataclass(frozen=True)
class Reading:
    """One answered question, and how well it was answered."""

    kind: str
    query: dict
    status: ReadingStatus
    data: Any = None
    detail: str = ""
    source: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    # Char accounting (see fintel.market.render). raw_chars is exact for the
    # fetched payload; capped_chars is the predicted rendered size after the
    # per-kind render caps. Both are 0 for failures/empties/denials.
    raw_chars: int = 0
    capped_chars: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def record(self) -> dict:
        """The trace shape. Data is excluded — the trace records what was asked
        and how it went, not a second copy of the cache."""
        out = {
            "kind": self.kind,
            "query": self.query,
            "status": self.status,
            "source": self.source,
            "latency_ms": round(self.latency_ms, 2),
            "raw_chars": self.raw_chars,
            "capped_chars": self.capped_chars,
            "raw_tokens": estimate_tokens(self.raw_chars),
            "capped_tokens": estimate_tokens(self.capped_chars),
        }
        if self.cached:
            out["cached"] = True
        if self.detail:
            out["detail"] = self.detail
        if isinstance(self.data, (list, tuple)):
            out["n"] = len(self.data)
        elif hasattr(self.data, "__len__") and not isinstance(self.data, (str, dict)):
            out["n"] = len(self.data)
        return out

    def payload(self) -> dict:
        """What an agent receives. Failure and denial are stated, not implied."""
        if self.status == "ok":
            return {"status": "ok", "data": self.data}
        if self.status == "empty":
            detail = self.detail or (
                f"no {self.kind} available for this query before the cutoff"
            )
            return {"status": "empty", "data": self.data, "detail": detail}
        return {"status": self.status, "data": None, "error": self.detail}


@dataclass
class DataAccess:
    """Kind-keyed, PIT-clamped, policy-checked, recorded."""

    cell: Cell
    sources: dict[str, DataSource]
    policy: AccessPolicy
    on_read: Any = None  # Callable[[Reading], None]
    memoize: bool = True
    _readings: list[Reading] = field(default_factory=list, init=False)
    _cache: dict[str, Reading] = field(default_factory=dict, init=False)

    @property
    def kinds(self) -> tuple[str, ...]:
        """Kinds this cell can actually read: declared by the strategy *and* bound."""
        return tuple(sorted(set(self.policy.kinds) & set(self.sources)))

    @property
    def readings(self) -> tuple[Reading, ...]:
        return tuple(self._readings)

    def read(self, kind: str, **query: Any) -> Reading:
        started = time.perf_counter()
        key = _cache_key(kind, query)
        if self.memoize and key in self._cache:
            # A repeated question inside one cell is the same question: an agent
            # re-reading prices mid-reasoning must not see a different answer.
            # Recorded again, marked, so read counts stay honest.
            cached = self._cache[key]
            return self._finish(replace(cached, latency_ms=0.0, cached=True), started, store=False)
        try:
            self.policy.check_kind(kind)
            source = self.sources.get(kind)
            if source is None:
                raise AccessDenied(
                    f"kind {kind!r} is declared but not bound to a source; "
                    f"available: {list(self.kinds)}"
                )
            for key in SUBJECT_KEYS:
                if key == "symbol" and key in query:
                    self.policy.check_symbol(query[key])
            clamped = self.policy.clamp_query(kind, query)
        except AccessDenied as exc:
            return self._finish(
                Reading(kind=kind, query=dict(query), status="denied", detail=str(exc)),
                started,
            )

        # The cutoff is injected here and nowhere else. An agent has no way to
        # supply, widen, or bypass it.
        try:
            data = source.fetch(clamped, self.cell.cutoff)
        except DataError as exc:
            logger.warning("%s read failed for %s: %s", kind, self.cell.describe(), exc)
            return self._finish(
                Reading(
                    kind=kind, query=clamped, status="failed",
                    detail=str(exc), source=getattr(source, "name", ""),
                ),
                started,
            )
        except Exception as exc:  # noqa: BLE001 - an adapter bug must not read as "no data"
            logger.exception("%s source raised for %s", kind, self.cell.describe())
            return self._finish(
                Reading(
                    kind=kind, query=clamped, status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    source=getattr(source, "name", ""),
                ),
                started,
            )

        return self._finish(
            Reading(
                kind=kind,
                query=clamped,
                status="empty" if _is_empty(data) else "ok",
                data=data,
                source=getattr(source, "name", ""),
            ),
            started,
        )

    def _finish(self, reading: Reading, started: float, *, store: bool = True) -> Reading:
        if not reading.cached:
            reading = replace(reading, latency_ms=(time.perf_counter() - started) * 1000)
        # Char accounting: exact for the fetched payload, predicted after the
        # per-kind render caps. Failures/empties/denials carry 0 — there is no
        # payload to render. See fintel.market.render.
        if reading.status == "ok" and reading.data is not None:
            rc = self.policy.render_cap_map.get(reading.kind, {})
            reading = replace(
                reading,
                raw_chars=fetched_chars(reading.data),
                capped_chars=predict_capped_chars(reading.kind, reading.data, rc),
            )
        # Never memoize a failure: it may be transient, and a sticky one would
        # turn a blip into an outage for the rest of the cell.
        if store and self.memoize and reading.status in ("ok", "empty"):
            self._cache[_cache_key(reading.kind, reading.query)] = reading
        self._readings.append(reading)
        if self.on_read is not None:
            self.on_read(reading)
        return reading

    def deny(self, kind: str, query: dict, detail: str) -> Reading:
        """Record a refusal that never reached a source (bad tool args, unknown tool).

        Schema-level denials used to return a payload without touching the log —
        which is how a cell could look clean while every MCP call failed with
        ``requires ['symbol']; got ['kwargs']``.
        """
        return self._finish(
            Reading(kind=kind, query=dict(query), status="denied", detail=detail),
            time.perf_counter(),
        )

    def summary(self) -> dict[str, int]:
        """Counts by status, for the cell record. A run whose reads mostly failed
        should not be reported as a run whose data was mostly absent."""
        out: dict[str, int] = {}
        for reading in self._readings:
            out[reading.status] = out.get(reading.status, 0) + 1
        return out
