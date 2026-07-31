"""No datum available on or after the decision date may reach an agent.

Visibility is keyed on when a datum became *available*, not the period it
describes. A Q1 statement filed 2022-05-15 is invisible to a 2022-04-01
decision, so a source with a publication lag must clamp on its filing column.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


def to_timestamp(value: Any) -> pd.Timestamp | None:
    """Coerce to a tz-naive Timestamp. `None` means 'no usable date'."""
    if value is None or value is pd.NaT:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


@dataclass(frozen=True)
class Cutoff:
    """The visibility boundary for one decision date.

    Strictly-before midnight of `decision_date`: a datum stamped with the
    decision date itself is out, since same-day availability is not knowable
    from a date alone.
    """

    decision_date: date

    def __post_init__(self) -> None:
        if isinstance(self.decision_date, datetime):
            object.__setattr__(self, "decision_date", self.decision_date.date())
        if not isinstance(self.decision_date, date):
            raise TypeError(f"decision_date must be a date, got {type(self.decision_date)}")

    @property
    def boundary(self) -> pd.Timestamp:
        return pd.Timestamp(self.decision_date)

    def allows(self, value: Any) -> bool:
        """Undated data is never allowed — it cannot be proven historical."""
        ts = to_timestamp(value)
        return ts is not None and ts < self.boundary

    def clamp_frame(self, df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        """Drop rows not visible yet. `column=None` clamps on the index."""
        if df.empty:
            return df
        if column is None:
            stamps = pd.to_datetime(pd.Series(df.index, index=df.index), errors="coerce", utc=False)
        else:
            if column not in df.columns:
                raise KeyError(f"clamp column {column!r} not in frame: {list(df.columns)}")
            stamps = pd.to_datetime(df[column], errors="coerce")
        if isinstance(stamps.dtype, pd.DatetimeTZDtype):
            stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
        return df[stamps.notna() & (stamps < self.boundary)]

    def clamp_records(self, records: Iterable[dict], key: str, *, sort: bool = True) -> list[dict]:
        kept = [r for r in records if self.allows(r.get(key))]
        if sort:
            kept.sort(key=lambda r: to_timestamp(r.get(key)) or pd.Timestamp.min)
        return kept

    def violations(self, records: Sequence[dict], key: str) -> list[dict]:
        """The audit direction: what a source should not have returned."""
        return [r for r in records if not self.allows(r.get(key))]
