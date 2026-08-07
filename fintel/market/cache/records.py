"""Gap-aware cache-first fills for dated-record feeds (fundamentals, news, …)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError
from fintel.market.data.store import RecordCache

logger = logging.getLogger(__name__)

Identity = Callable[[dict], Any]
SortKey = Callable[[dict], Any]
FetchSpan = Callable[[Date, Date], list[dict]]


def ensure_records(
    cache: RecordCache,
    key: str,
    since: Date,
    through: Date,
    *,
    fetch_span: FetchSpan,
    identity: Identity,
    sort: SortKey,
    online: bool,
    source_name: str,
    kind_label: str,
) -> list[dict]:
    """Return records covering ``[since, through]``, filling gaps when online.

    * Full coverage → return cache.
    * Gaps + online → fetch each gap, merge under the store lock.
    * Gaps + offline + empty cache → ``DataError``.
    * Gaps + offline + partial cache → warn and serve what is cached.
    """
    if since > through:
        return []
    coverage, records = cache.read(key)
    gaps = cov.missing(coverage, since, through)
    if not gaps:
        return records
    if not online:
        if not coverage:
            raise DataError(
                f"{source_name}: nothing cached for {key} {kind_label} covering "
                f"[{since}, {through}] and no network access configured"
            )
        logger.warning(
            "%s: %s %s cache is short of [%s, %s]; serving what is cached",
            source_name,
            key,
            kind_label,
            since,
            through,
        )
        return records

    fresh: dict[Any, dict] = {}
    for lo, hi in gaps:
        for item in fetch_span(lo, hi):
            fresh[identity(item)] = item
    return cache.merge(
        key,
        list(fresh.values()),
        gaps,
        key=identity,
        sort=sort,
    )


@dataclass
class CachedRecordsFeed:
    """Reusable cache-first feeder around a vendor ``fetch_span(key, lo, hi)``."""

    cache: RecordCache
    fetch_span: Callable[[str, Date, Date], list[dict]]
    identity: Identity
    sort: SortKey
    online: bool
    source_name: str
    kind_label: str

    def ensure(self, key: str, since: Date, through: Date) -> list[dict]:
        return ensure_records(
            self.cache,
            key,
            since,
            through,
            fetch_span=lambda lo, hi: self.fetch_span(key, lo, hi),
            identity=self.identity,
            sort=self.sort,
            online=self.online,
            source_name=self.source_name,
            kind_label=self.kind_label,
        )
