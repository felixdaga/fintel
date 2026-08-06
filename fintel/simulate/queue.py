"""Bounded parallel execution. One primitive, used at every fan-out.

`map_parallel(fn, items, bound)` returns results in *input order*, not completion
order, so a fan-out's reduction is deterministic regardless of which cell
finished first. `bound=1` runs sequentially — the default, and the shape every
non-thread-safe agent must use.

Failures are returned, not raised: a backtest that aborts on one bad cell loses
every other cell's work for that date, which is the defect `agents.invoke`
already fixes at the cell level. Here we carry that policy up: a failed item is
a `None` in its slot, and the reducer decides what that means.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(fn: Callable[[T], R], items: list[T], *, bound: int = 1) -> list[R | None]:
    if bound <= 1 or len(items) <= 1:
        out: list[R | None] = []
        for item in items:
            try:
                out.append(fn(item))
            except Exception:
                out.append(None)
        return out

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=bound) as pool:
        future_to_index = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = None
    return results
