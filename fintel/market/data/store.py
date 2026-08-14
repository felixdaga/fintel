"""On-disk caches. Reads and writes only — no PIT opinion, no network.

Two shapes: dated records as JSON with a coverage sidecar, and price bars as
parquet. Both are keyed per symbol so a package's `cache/` is a portable,
diffable unit.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime as DateTime
from pathlib import Path

import pandas as pd

from fintel.market.calendar import TradingCalendar
from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError
from fintel.market.data.coverage import Span

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume")
LOCK_TIMEOUT_S = 60.0
_CAL = TradingCalendar()


def _bar_dates(df: pd.DataFrame) -> list[Date]:
    if df is None or df.empty or "date" not in df.columns:
        return []
    out: list[Date] = []
    for raw in df["date"]:
        if isinstance(raw, DateTime):
            out.append(raw.date())
        elif isinstance(raw, Date):
            out.append(raw)
    return out


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)


@contextmanager
def locked(target: Path, *, timeout: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """Serialise read-modify-write on one cache file.

    Atomic writes alone don't make a merge safe. Two cells fetching different
    spans of the same symbol both read the old coverage, both append their own,
    and whichever writes last erases the other's records — a lost update that
    looks like a cache that simply doesn't have the data. Callers must re-read
    inside the lock, which is why merging lives on the store rather than in each
    source.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    if fcntl is None:
        yield
        return
    with lock_path.open("w") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise DataError(
                        f"timed out after {timeout:.0f}s waiting to write {target.name}; "
                        f"another process may be stuck holding {lock_path.name}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass(frozen=True)
class RecordCache:
    """`{"_coverage": [[from, through], ...], "records": [...]}` per symbol."""

    root: Path
    kind: str

    def path(self, symbol: str) -> Path:
        return self.root / self.kind / f"{symbol}.json"

    def read(self, symbol: str) -> tuple[list[Span], list[dict]]:
        path = self.path(symbol)
        if not path.exists():
            return [], []
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("corrupt cache %s: %s — treating as empty", path, exc)
            return [], []
        return cov.from_json(blob.get("_coverage")), list(blob.get("records") or [])

    def write(self, symbol: str, coverage: list[Span], records: list[dict]) -> None:
        atomic_write(
            self.path(symbol),
            json.dumps({"_coverage": cov.to_json(coverage), "records": records}, default=str),
        )

    def merge(
        self,
        symbol: str,
        records: list[dict],
        spans: list[Span],
        *,
        key: Callable[[dict], str],
        sort: Callable[[dict], str],
    ) -> list[dict]:
        """Fold fresh records into whatever is on disk now, under a lock.

        Re-reads inside the lock so a concurrent writer's records survive.
        """
        with locked(self.path(symbol)):
            coverage, existing = self.read(symbol)
            merged = {key(record): record for record in existing}
            merged.update({key(record): record for record in records})
            out = sorted(merged.values(), key=sort)
            self.write(symbol, cov.coalesce([*coverage, *spans]), out)
            return out


@dataclass(frozen=True)
class PriceStore:
    """Daily bars per symbol as parquet, with an optional coverage sidecar.

    Coverage falls back to the frame's own min/max date when no sidecar exists,
    which keeps caches written by the previous implementation readable.
    """

    root: Path

    def path(self, symbol: str) -> Path:
        return self.root / "prices" / f"{symbol}.parquet"

    def _sidecar(self, symbol: str) -> Path:
        return self.path(symbol).with_suffix(".coverage.json")

    def read(self, symbol: str) -> pd.DataFrame | None:
        path = self.path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning("unreadable parquet %s: %s — treating as empty", path, exc)
            return None
        if df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    def coverage(self, symbol: str) -> list[Span]:
        side = self._sidecar(symbol)
        if side.exists():
            try:
                return cov.from_json(json.loads(side.read_text()).get("_coverage"))
            except (json.JSONDecodeError, OSError):
                pass
        df = self.read(symbol)
        if df is None or df.empty:
            return []
        return [(df["date"].iloc[0], df["date"].iloc[-1])]

    def write(self, symbol: str, df: pd.DataFrame, coverage: list[Span]) -> None:
        """Replace the cached frame. Atomic, so a concurrent reader — which takes
        no lock — can never observe a half-written parquet. Writing in place made
        readers see truncated files and treat a populated cache as a miss."""
        path = self.path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            out.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        atomic_write(self._sidecar(symbol), json.dumps({"_coverage": cov.to_json(coverage)}))

    def merge(self, symbol: str, fresh: pd.DataFrame, span: Span) -> pd.DataFrame:
        """Add a fetched span to the cache. Re-reads under the lock, so a
        concurrent writer's bars are not lost.

        Coverage records the fetch window so weekends and holidays inside it
        are not retried. NYSE sessions the vendor skipped between the first
        and last returned bar are punched out — a truncated response must
        not look fully fetched.
        """
        with locked(self.path(symbol)):
            existing = self.read(symbol)
            combined = (
                fresh if existing is None else pd.concat([existing, fresh], ignore_index=True)
            )
            holes = _CAL.interior_missing_sessions(_bar_dates(fresh), span[0], span[1])
            coverage = cov.without_days(cov.coalesce([*self.coverage(symbol), span]), holes)
            self.write(symbol, combined, coverage)
            out = self.read(symbol)
        return out if out is not None else combined

    def record_empty_span(self, symbol: str, span: Span) -> None:
        """Remember that a span was fetched and held nothing, so it isn't re-fetched.
        Without this, 'never asked' and 'asked, nothing there' are the same state."""
        with locked(self.path(symbol)):
            existing = self.read(symbol)
            frame = existing if existing is not None else pd.DataFrame(columns=PRICE_COLUMNS)
            self.write(symbol, frame, [*self.coverage(symbol), span])

    def symbols(self) -> list[str]:
        d = self.root / "prices"
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.parquet"))

    def bars_on_or_before(self, symbol: str, day: Date) -> pd.DataFrame | None:
        """Every bar up to and including `day`. No PIT opinion — the caller's job."""
        df = self.read(symbol)
        if df is None:
            return None
        out = df[df["date"] <= day]
        return out if not out.empty else None
