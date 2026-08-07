"""Exact-key blob cache (web_search): hit → return; miss → fetch + atomic write."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from fintel.market.data.base import DataError
from fintel.market.data.store import atomic_write

logger = logging.getLogger(__name__)


def ensure_query_blob(
    path: Path,
    *,
    online: bool,
    fetch: Callable[[], dict],
    source_name: str,
    miss_detail: str,
) -> dict:
    """Return a JSON blob at ``path``, fetching once on miss when online."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("corrupt %s cache %s: %s", source_name, path, exc)

    if not online:
        raise DataError(f"{source_name}: {miss_detail}")

    blob = fetch()
    atomic_write(path, json.dumps(blob))
    return blob
