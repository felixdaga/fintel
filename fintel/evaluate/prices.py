"""Build a `PriceLookup` (the unclamped scoring price path) for a finished job.

`market/realized.py` is the one module allowed to read past the decision
date — that's the measurement. This helper wires it to a job's cache so the
KPI layer can compute forward returns without touching the simulation. The
cache root is resolved from the job config's `output_root` (`<output>/cache`),
overridable for a cache shared with another run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import date as Date
from pathlib import Path

from fintel.market.data.store import PriceStore
from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.evaluate import Signals

logger = logging.getLogger(__name__)


def price_lookup_for(job_dir: Path, *, cache_root: str | Path | None = None) -> PriceLookup:
    """Build a `PriceLookup` over the job's price cache."""
    if cache_root is None:
        # Default: <job output_root>/cache. The job config records output_root.
        cache_root = _default_cache_root(job_dir)
    store = PriceStore(root=Path(cache_root).expanduser())
    return PriceLookup(store=store)


def hold_mark_symbols(signals: Signals) -> list[Symbol]:
    """Symbols in the last decision book — the names that matter for a hold-to-mark."""
    dates = sorted(signals.ensemble)
    if not dates:
        return sorted(signals.universe)
    return sorted(signals.ensemble[dates[-1]])


def mark_as_of(prices: PriceLookup, signals: Signals, *, min_coverage: float = 1.0) -> Date | None:
    """Latest cache bar date for the held book (not the full historical universe)."""
    return prices.latest_bar_date(hold_mark_symbols(signals), min_coverage=min_coverage)


def ensure_job_prices(
    job_dir: Path,
    symbols: Iterable[Symbol],
    *,
    through: Date | None = None,
    cache_root: str | Path | None = None,
) -> Date | None:
    """Fill price-cache gaps through ``through`` (default: today) when online.

    Uses the job's frozen ``r1/config.json`` data bindings + env secrets.
    Offline / missing keys: no-op (returns whatever the cache already has).
    Returns the latest common bar date across ``symbols`` after the fill.
    """
    job_dir = Path(job_dir).expanduser().resolve()
    through = through or Date.today()
    syms = list(symbols)
    if not syms:
        return None

    prices = price_lookup_for(job_dir, cache_root=cache_root)
    src = _prices_source(job_dir, cache_root=cache_root)
    if src is not None and hasattr(src, "ensure"):
        # MassivePrices.ensure ignores ``since`` and fills from history_start.
        since = Date(through.year - 5, 1, 1)
        for sym in syms:
            try:
                src.ensure(sym, since, through)
            except Exception as exc:  # noqa: BLE001 — analytics warm path must not crash
                logger.warning("ensure_job_prices: %s failed for %s: %s", type(src).__name__, sym, exc)
        prices = price_lookup_for(job_dir, cache_root=cache_root)
    return prices.latest_bar_date(syms)


def _prices_source(job_dir: Path, *, cache_root: str | Path | None = None):
    """Build the job's prices DataSource, or None if config/secrets unavailable."""
    try:
        from fintel.market.factory import build_data_sources
        from fintel.market.settings import MarketConfig
        from fintel.models.run import RunConfig
        from fintel.utils.secrets import bootstrap_env
    except ImportError:
        return None

    run_cfg_path = job_dir / "r1" / "config.json"
    if not run_cfg_path.is_file():
        return None
    try:
        bootstrap_env()
        cfg = RunConfig.model_validate(json.loads(run_cfg_path.read_text()))
        root = Path(cache_root).expanduser() if cache_root is not None else _default_cache_root(job_dir)
        market = MarketConfig.from_env(cache_root=root)
        sources = build_data_sources(cfg.data, config=market)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_job_prices: could not build prices source: %s", exc)
        return None
    return sources.get("prices")


def _default_cache_root(job_dir: Path) -> Path:
    """Resolve ``<output_root>/cache`` for a finished job.

    Job configs often store a *relative* ``output_root`` (e.g. ``\"runs\"``).
    Resolving that against the process cwd breaks notebooks launched from
    ``fintel/evaluate/`` (empty cache → zero ICs / nonsense NAV). Prefer an
    absolute ``output_root`` when present; otherwise use ``job_dir.parent``
    (the job always lives at ``<output_root>/<job_id>``).
    """
    job_dir = Path(job_dir).expanduser().resolve()
    config_path = job_dir / "config.json"
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
            out = cfg.get("output_root")
            if out:
                out_path = Path(out).expanduser()
                if out_path.is_absolute():
                    return out_path / "cache"
        except json.JSONDecodeError:
            pass
    # Relative / missing output_root: job_dir is <output_root>/<job_id>.
    return job_dir.parent / "cache"
