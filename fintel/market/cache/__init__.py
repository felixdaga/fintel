"""Feed-level cache policy.

Data sources own *vendor* I/O (fetch a span / query from the network). This
package owns *cache-first* behaviour: gap detection, offline miss errors,
short-cache warnings, and merge. Enabling cache for a new records source is
wiring ``ensure_records`` (or ``CachedRecordsFeed``) around a ``fetch_span`` —
not reimplementing the loop inside the vendor file.

On-disk shapes stay in :mod:`fintel.market.data.store` (``RecordCache``,
``PriceStore``). This package never writes its own layout.
"""

from fintel.market.cache.prices import CachedPricesFeed, ensure_prices, interior_session_holes
from fintel.market.cache.query import ensure_query_blob
from fintel.market.cache.records import CachedRecordsFeed, ensure_records

__all__ = [
    "CachedPricesFeed",
    "CachedRecordsFeed",
    "ensure_prices",
    "ensure_query_blob",
    "ensure_records",
    "interior_session_holes",
]
