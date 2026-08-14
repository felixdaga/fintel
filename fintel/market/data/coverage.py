"""Which date spans a cache has actually fetched.

Tracked as a list of inclusive [from, through] intervals rather than a single
`fetched_through` bound. A lone upper bound is wrong under backward and
interleaved date jumps: a cache warmed through 2026 claims to "cover" a 2024
request while holding zero 2024 records, and the run reports an empty result
instead of fetching.

For daily prices the span is the *fetch window*, not a promise that every
session has a bar. ``PriceStore.merge`` punches NYSE sessions the vendor
skipped, so a truncated response cannot hide interior holes behind one
sidecar interval. ``record_empty_span`` still covers the whole window —
that is "we asked, nothing is there" (delisted, unentitled), which must
not be retried.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as Date
from datetime import timedelta
from typing import Any

Span = tuple[Date, Date]

DAY = timedelta(days=1)


def coalesce(spans: list[Span]) -> list[Span]:
    """Sort and merge overlapping or adjacent inclusive intervals."""
    out: list[Span] = []
    for a, b in sorted(spans):
        if a > b:
            continue
        if out and a <= out[-1][1] + DAY:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def missing(coverage: list[Span], need_from: Date, need_through: Date) -> list[Span]:
    """Sub-intervals of [need_from, need_through] not already covered."""
    if need_from > need_through:
        return []
    gaps: list[Span] = []
    cursor = need_from
    for a, b in coalesce(coverage):
        if b < cursor:
            continue
        if a > need_through:
            break
        if a > cursor:
            gaps.append((cursor, min(a - DAY, need_through)))
        cursor = max(cursor, b + DAY)
        if cursor > need_through:
            break
    if cursor <= need_through:
        gaps.append((cursor, need_through))
    return gaps


def covers(coverage: list[Span], need_from: Date, need_through: Date) -> bool:
    return not missing(coverage, need_from, need_through)


def without_days(coverage: list[Span], days: Iterable[Date]) -> list[Span]:
    """Punch individual days out of coverage spans.

    Used when a fetch window came back sparse: weekends and holidays in the
    window stay covered (they were never going to have bars), but a missing
    NYSE session must not look fetched.
    """
    drop = {d for d in days if d is not None}
    if not drop:
        return coalesce(coverage)
    out: list[Span] = []
    for a, b in coalesce(coverage):
        cur = a
        for d in sorted(x for x in drop if a <= x <= b):
            if cur < d:
                out.append((cur, d - DAY))
            cur = d + DAY
        if cur <= b:
            out.append((cur, b))
    return coalesce(out)


def to_json(spans: list[Span]) -> list[list[str]]:
    return [[a.isoformat(), b.isoformat()] for a, b in coalesce(spans)]


def from_json(raw: Any) -> list[Span]:
    out: list[Span] = []
    for pair in raw or []:
        try:
            out.append(
                (Date.fromisoformat(str(pair[0])[:10]), Date.fromisoformat(str(pair[1])[:10]))
            )
        except (TypeError, ValueError, IndexError):
            continue
    return coalesce(out)
