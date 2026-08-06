"""Two cells running at once must not overwrite each other.

The old repo hit all of these for real: concurrent cells sharing one session
directory, every symbol on a date writing into one `decisions/<date>.json`, and
cache merges losing an update. These tests exist so a fix can't silently regress.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import pytest

from fintel.environment import Cell, RuntimeConfig, build_environment
from fintel.market.data import coverage as cov
from fintel.market.data.store import PriceStore, RecordCache, locked
from fintel.models import ids
from fintel.models.paths import JobPaths

DAY = date(2024, 6, 3)
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN"]


def bars(start: str, n: int, price: float) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "date": days.date,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000.0,
        }
    )


# ── artifacts are named after their writer ───────────────────────────────────


def test_each_cell_owns_its_own_output_file(tmp_path):
    """The old layout put every symbol on a date into one decisions/<date>.json,
    so concurrent cells overwrote each other and views vanished without error."""
    trial = JobPaths.under(tmp_path, "job-1").run(1).trial(DAY)
    paths = {sym: trial.cell(sym) for sym in UNIVERSE}
    assert len(set(paths.values())) == len(UNIVERSE)

    def write(symbol: str) -> None:
        trial.cells_dir.mkdir(parents=True, exist_ok=True)
        trial.cell(symbol).write_text(json.dumps({"symbol": symbol}))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, UNIVERSE))

    assert len(trial.cell_files()) == len(UNIVERSE)
    recovered = {json.loads(p.read_text())["symbol"] for p in trial.cell_files()}
    assert recovered == set(UNIVERSE)


def test_the_reduced_decision_is_a_separate_single_writer_file(tmp_path):
    trial = JobPaths.under(tmp_path, "job-1").run(1).trial(DAY)
    assert trial.decision.name == "decision.json"
    assert trial.decision.parent == trial.root
    assert trial.cell("AAPL").parent == trial.cells_dir
    # A cell can never land on the reduction by accident.
    assert trial.decision not in {trial.cell(s) for s in UNIVERSE}


def test_traces_are_per_cell(tmp_path):
    trial = JobPaths.under(tmp_path, "job-1").run(1).trial(DAY)
    assert trial.trace("AAPL") != trial.trace("MSFT")


def test_portfolio_cells_get_the_sentinel_name(tmp_path):
    trial = JobPaths.under(tmp_path, "job-1").run(1).trial(DAY)
    assert trial.cell(ids.cell_id(None)).name == "__portfolio__.json"


# ── session directories ──────────────────────────────────────────────────────


def test_concurrent_cells_never_share_a_session_dir(tmp_path):
    """No pid walking, no shared fallback: the path is a function of the cell."""
    from fintel.environment.session import read_cell
    from fintel.market.data.synthetic import SyntheticPrices

    def run(symbol: str) -> tuple[str, str]:
        env = build_environment(
            cell=Cell(run_id="r1", decision_date=DAY, symbols=(symbol,)),
            sources={"prices": SyntheticPrices()},
            universe=UNIVERSE,
            runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
        )
        env.access.read("prices", symbol=symbol)
        return str(env.session.path), read_cell(env.session.path)["symbols"][0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, UNIVERSE))

    assert len({path for path, _ in results}) == len(UNIVERSE)
    # Every session still describes the cell that created it.
    for path, symbol in results:
        assert symbol in path


def test_a_second_cell_cannot_quietly_take_over_a_session(tmp_path):
    from fintel.environment.session import SessionDir, SessionError

    cell = Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",))
    SessionDir(root=tmp_path, cell=cell).create()
    with pytest.raises(SessionError, match="already has contents"):
        SessionDir(root=tmp_path, cell=cell).create()


# ── cache merges under concurrency ───────────────────────────────────────────


def test_concurrent_price_merges_do_not_lose_an_update(tmp_path):
    """Atomic writes alone aren't enough: read-modify-write needs the lock, or
    the last writer erases the other's bars and it looks like a cache miss."""
    store = PriceStore(root=tmp_path)
    spans = [
        ("2024-01-01", 20, 100.0),
        ("2024-03-01", 20, 200.0),
        ("2024-05-01", 20, 300.0),
        ("2024-07-01", 20, 400.0),
    ]

    def merge(args) -> None:
        start, n, price = args
        frame = bars(start, n, price)
        store.merge("AAPL", frame, (frame["date"].iloc[0], frame["date"].iloc[-1]))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(merge, spans))

    got = store.read("AAPL")
    assert len(got) == 80, "a merge was lost"
    assert set(got["close"]) == {100.0, 200.0, 300.0, 400.0}
    # Coverage must mention all four spans.
    assert len(store.coverage("AAPL")) == 4


def test_concurrent_record_merges_do_not_lose_an_update(tmp_path):
    cache = RecordCache(root=tmp_path, kind="news")

    def merge(i: int) -> None:
        day = date(2024, 1, 1 + i)
        cache.merge(
            "AAPL",
            [{"id": f"n{i}", "published_at": day.isoformat()}],
            [(day, day)],
            key=lambda r: r["id"],
            sort=lambda r: r["published_at"],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(merge, range(12)))

    coverage, records = cache.read("AAPL")
    assert len(records) == 12, "a merge was lost"
    assert {r["id"] for r in records} == {f"n{i}" for i in range(12)}
    assert cov.covers(coverage, date(2024, 1, 1), date(2024, 1, 12))


def test_a_reader_never_sees_a_half_written_frame(tmp_path, caplog):
    """Readers take no lock, so the write itself must be atomic. Writing parquet
    in place let a reader observe a truncated file, log it as unreadable, and
    treat a populated cache as a miss."""
    import logging

    store = PriceStore(root=tmp_path)
    store.merge("AAPL", bars("2024-01-01", 40, 100.0), (date(2024, 1, 1), date(2024, 2, 23)))

    seen: list[int] = []

    def reader() -> None:
        for _ in range(60):
            got = store.read("AAPL")
            seen.append(0 if got is None else len(got))

    def writer() -> None:
        for i in range(20):
            store.merge(
                "AAPL", bars("2024-03-01", 20, 200.0 + i), (date(2024, 3, 1), date(2024, 3, 28))
            )

    with caplog.at_level(logging.WARNING, logger="fintel.market.data.store"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda f: f(), [reader, writer, reader, writer]))

    assert "unreadable parquet" not in caplog.text
    # Every read saw a complete frame — never zero, never a partial row count.
    assert seen and all(n in (40, 60) for n in seen), sorted(set(seen))


def test_an_empty_span_is_remembered_so_it_is_not_refetched(tmp_path):
    store = PriceStore(root=tmp_path)
    store.record_empty_span("XYZ", (date(2024, 1, 1), date(2024, 1, 31)))
    assert cov.covers(store.coverage("XYZ"), date(2024, 1, 1), date(2024, 1, 31))
    assert store.read("XYZ") is None


def test_the_lock_serialises_and_releases(tmp_path):
    target = tmp_path / "x.json"
    order: list[str] = []

    def critical(tag: str) -> None:
        with locked(target):
            order.append(f"{tag}-in")
            order.append(f"{tag}-out")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(critical, "abcd"))

    # No interleaving: every in is immediately followed by its own out.
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0]


# ── repeated reads inside one cell ───────────────────────────────────────────


def test_a_repeated_read_is_stable_and_marked(tmp_path):
    from fintel.market.data.synthetic import SyntheticPrices

    env = build_environment(
        cell=Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",)),
        sources={"prices": SyntheticPrices()},
        universe=UNIVERSE,
        runtime=RuntimeConfig(),
    )
    first = env.access.read("prices", symbol="AAPL")
    second = env.access.read("prices", symbol="AAPL")
    assert second.cached is True
    assert first.cached is False
    assert len(first.data) == len(second.data)
    # Both appear in the record, so read counts stay honest.
    reads = env.log.reads
    assert len(reads) == 2
    assert reads[1]["cached"] is True


def test_a_different_query_is_not_served_from_the_repeat_cache(tmp_path):
    from fintel.market.data.synthetic import SyntheticPrices

    env = build_environment(
        cell=Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",)),
        sources={"prices": SyntheticPrices()},
        universe=UNIVERSE,
        runtime=RuntimeConfig(),
    )
    a = env.access.read("prices", symbol="AAPL", lookback_days=10)
    b = env.access.read("prices", symbol="AAPL", lookback_days=60)
    assert b.cached is False
    assert len(b.data) > len(a.data)


def test_a_failure_is_never_memoized(tmp_path):
    """A transient blip must not become an outage for the rest of the cell."""
    from fintel.environment.access import DataAccess
    from fintel.environment.policy import AccessPolicy
    from fintel.market.data.base import DataError

    class Flaky:
        name = "flaky"
        kinds = ("prices",)

        def __init__(self):
            self.n = 0

        def fetch(self, query, cutoff):
            self.n += 1
            if self.n == 1:
                raise DataError("transient")
            return [{"date": "2024-05-01"}]

    source = Flaky()
    access = DataAccess(
        cell=Cell(run_id="r1", decision_date=DAY, symbols=("AAPL",)),
        sources={"prices": source},
        policy=AccessPolicy(kinds=frozenset({"prices"}), decidable=frozenset({"AAPL"})),
    )
    assert access.read("prices", symbol="AAPL").status == "failed"
    assert access.read("prices", symbol="AAPL").status == "ok"
    assert source.n == 2


# ── per-role tool subsets ────────────────────────────────────────────────────


def test_a_role_can_get_a_narrower_tool_set(tmp_path):
    """A multi-role desk splits tools by role, but both roles read through the
    one clamped, recorded path — a role boundary is not a second data path."""
    from tests.test_environment import an_environment

    env = an_environment(tmp_path, kinds=("prices", "fundamentals", "news"))
    quant = env.tools.subset(("prices", "fundamentals"))
    narrative = env.tools.subset(("news",))

    assert set(quant.names) == {"get_prices", "get_fundamentals"}
    assert set(narrative.names) == {"get_news"}
    assert quant.access is env.access and narrative.access is env.access

    assert narrative.call("get_prices", {"symbol": "AAPL"})["status"] == "denied"
    assert quant.call("get_prices", {"symbol": "AAPL"})["status"] == "ok"
    # Both roles' reads land in the same record.
    assert env.log.counts()["ok"] >= 1


def test_a_subset_cannot_invent_a_kind(tmp_path):
    from tests.test_environment import an_environment

    env = an_environment(tmp_path, kinds=("prices",))
    with pytest.raises(ValueError, match="not available to this cell"):
        env.tools.subset(("prices", "news"))
