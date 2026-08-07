"""Build a `PriceLookup` (the unclamped scoring price path) for a finished job.

`market/realized.py` is the one module allowed to read past the decision
date — that's the measurement. This helper wires it to a job's cache so the
KPI layer can compute forward returns without touching the simulation. The
cache root is resolved from the job config's `output_root` (`<output>/cache`),
overridable for a cache shared with another run.
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.market.data.store import PriceStore
from fintel.market.realized import PriceLookup


def price_lookup_for(job_dir: Path, *, cache_root: str | Path | None = None) -> PriceLookup:
    """Build a `PriceLookup` over the job's price cache."""
    if cache_root is None:
        # Default: <job output_root>/cache. The job config records output_root.
        cache_root = _default_cache_root(job_dir)
    store = PriceStore(root=Path(cache_root).expanduser())
    return PriceLookup(store=store)


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
