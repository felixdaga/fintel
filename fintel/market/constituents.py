"""Point-in-time index membership — the survivorship-clean universe.

Backed by the open `index-constitution` dataset, which records each
constituent's opt-in / opt-out date. Reconstructing membership per decision
date means names that later dropped out are still scored while they were
members, and names added later are not scored before they joined. A frozen
snapshot cannot do this.

Edge convention, matching the dataset: a name is a member on its opt-in date
and is not on its opt-out date. A blank opt-out means still a member.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import pandas as pd

from fintel.market.settings import MarketConfig
from fintel.models.common import Symbol

logger = logging.getLogger(__name__)

GITHUB_BASE = "https://raw.githubusercontent.com/unliftedq/index-constitution/main"
FOREVER = "9999-12-31"

# preset name → (dataset key, human label)
INDEX_PRESETS: dict[str, tuple[str, str]] = {
    "dow30": ("dow30", "Dow Jones Industrial Average"),
    "sp500": ("sp500", "S&P 500"),
    "nasdaq100": ("nasdaq100", "NASDAQ-100"),
    "csi300": ("csi300", "CSI 300"),
    "csi500": ("csi500", "CSI 500"),
}

REQUIRED_COLUMNS = ("symbol", "opt-in", "opt-out")


@dataclass(frozen=True)
class UniverseReport:
    """What `verify` found. `issues` are advisory; fatal problems raise."""

    preset: str
    index_key: str
    n_rows: int
    n_symbols: int
    per_date: dict[str, int]
    earliest_opt_in: str | None
    issues: list[str] = field(default_factory=list)
    # The dataset carries membership only. Index weights (price-weighted for the
    # DJIA, float-cap for SPX) are a separate computation this source cannot give.
    weights_available: bool = False

    @property
    def min_members(self) -> int:
        return min(self.per_date.values(), default=0)

    @property
    def max_members(self) -> int:
        return max(self.per_date.values(), default=0)


@dataclass(frozen=True)
class HistoricalUniverse:
    """Index constituents resolved per date. Load once, query many dates."""

    name: str
    index_key: str
    history: pd.DataFrame  # normalised: symbol, opt_in, opt_out (ISO strings)

    def active_at(self, as_of: Date) -> list[Symbol]:
        asof = as_of.isoformat()
        df = self.history
        members = df.loc[(df["opt_in"] <= asof) & (df["opt_out"] > asof), "symbol"]
        return sorted(set(members.tolist()))

    @property
    def current_members(self) -> list[Symbol]:
        return self.active_at(Date.today())

    def union_over(self, dates: list[Date]) -> list[Symbol]:
        """Every symbol needed across the grid — what a prefetch must warm."""
        out: set[Symbol] = set()
        for d in dates:
            out.update(self.active_at(d))
        return sorted(out)

    def verify(self, decision_dates: list[Date]) -> UniverseReport:
        """Fail before any model spend if the grid can't be resolved.

        Fatal: a date resolving to zero members. That produced silent empty runs
        before — a stale cache or a date predating the index looks identical to a
        working run until the report comes out blank.
        """
        df = self.history
        if df.empty:
            raise ValueError(f"constituents history for {self.name!r} is empty")

        earliest = str(df["opt_in"].min()) if len(df) else None
        per_date: dict[str, int] = {}
        zeros: list[str] = []
        for d in decision_dates:
            n = len(self.active_at(d))
            per_date[d.isoformat()] = n
            if n == 0:
                zeros.append(d.isoformat())
        if zeros:
            # A date before the index existed lands here too, so the earliest
            # opt-in is the diagnostic that actually explains it.
            raise ValueError(
                f"universe {self.name!r} resolves to ZERO members on {len(zeros)} "
                f"decision date(s): {zeros}. This would be a silent empty run. "
                f"Earliest opt-in in the table is {earliest}."
            )

        issues: list[str] = []
        counts = list(per_date.values())
        if counts:
            modal = max(set(counts), key=counts.count)
            for ds, n in per_date.items():
                if n != modal:
                    issues.append(
                        f"{ds} has {n} members (modal {modal}) — rebalance boundary in window"
                    )

        return UniverseReport(
            preset=self.name,
            index_key=self.index_key,
            n_rows=len(df),
            n_symbols=int(df["symbol"].nunique()),
            per_date=per_date,
            earliest_opt_in=earliest,
            issues=issues,
        )


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"constituents table missing columns {missing}; got {list(df.columns)}")
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str).str.strip(),
            "opt_in": df["opt-in"].astype(str).str.slice(0, 10),
            "opt_out": df["opt-out"]
            .where(df["opt-out"].notna(), FOREVER)
            .astype(str)
            .str.slice(0, 10),
        }
    )
    bad_in = out.loc[pd.to_datetime(out["opt_in"], errors="coerce").isna(), "symbol"]
    if len(bad_in):
        raise ValueError(f"unparseable opt-in dates for {sorted(set(bad_in))[:5]}")
    inverted = out.loc[out["opt_in"] > out["opt_out"], "symbol"]
    if len(inverted):
        raise ValueError(f"opt-in after opt-out for {sorted(set(inverted))[:5]}")
    return out


def load_history(index_key: str, config: MarketConfig, *, refresh: bool = False) -> pd.DataFrame:
    """Read the cached table, fetching once if absent.

    Cached under the strategy's own cache root so a shipped package replays with
    the exact membership it was built against.
    """
    path: Path = config.dir("constituents") / f"{index_key}.csv"
    if refresh or not path.exists():
        if config.offline:
            raise FileNotFoundError(
                f"constituents cache missing for {index_key!r} at {path} and offline mode "
                f"is on; populate the cache or allow network access"
            )
        import httpx

        url = f"{GITHUB_BASE}/history/{index_key}.csv"
        logger.info("constituents: fetching %s", url)
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        path.write_text(resp.text)
    # utf-8-sig: the dataset ships a BOM.
    return normalise(pd.read_csv(path, encoding="utf-8-sig"))


def historical_universe(
    preset: str, *, config: MarketConfig, refresh: bool = False
) -> HistoricalUniverse:
    if preset not in INDEX_PRESETS:
        raise ValueError(
            f"unknown index universe preset {preset!r}; available: {sorted(INDEX_PRESETS)}"
        )
    index_key, _label = INDEX_PRESETS[preset]
    return HistoricalUniverse(
        name=preset,
        index_key=index_key,
        history=load_history(index_key, config, refresh=refresh),
    )
