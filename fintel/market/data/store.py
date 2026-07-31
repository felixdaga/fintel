"""On-disk caches. Reads and writes only — no PIT opinion, no network.

Two shapes: dated records as JSON with a coverage sidecar, and price bars as
parquet. Both are keyed per symbol so a package's `cache/` is a portable,
diffable unit.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd

from fintel.market.data import coverage as cov
from fintel.market.data.coverage import Span

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)


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
        path = self.path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        out.to_parquet(path, index=False)
        atomic_write(self._sidecar(symbol), json.dumps({"_coverage": cov.to_json(coverage)}))

    def merge(self, symbol: str, fresh: pd.DataFrame, span: Span) -> pd.DataFrame:
        existing = self.read(symbol)
        combined = fresh if existing is None else pd.concat([existing, fresh], ignore_index=True)
        self.write(symbol, combined, [*self.coverage(symbol), span])
        return self.read(symbol) or combined

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
