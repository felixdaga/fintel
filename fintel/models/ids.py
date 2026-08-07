"""The one naming scheme. Nothing else may invent an id format."""

from __future__ import annotations

import re
import secrets
from datetime import UTC, date, datetime

# Collapse anything that is not a safe id char. Prefer underscores so job ids
# read as ``strategy_agent_date_runinfo``.
_TOKEN = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 40) -> str:
    """Legacy hyphen slug (paths / short labels). Prefer :func:`token` for ids."""
    return _TOKEN.sub("-", text.strip().lower()).strip("-")[:limit].strip("-")


def token(text: str, limit: int = 48) -> str:
    """Lowercase underscore token for job-id segments."""
    return _TOKEN.sub("_", text.strip().lower()).strip("_")[:limit].strip("_")


def new_job_id(
    *,
    strategy: str,
    agent: str,
    now: datetime | None = None,
    model: str | None = None,
    k_repeats: int | None = None,
    runinfo: str | None = None,
) -> str:
    """``{strategy}_{agent}_{YYYYMMDD}_{runinfo}``.

    ``runinfo`` defaults to ``{model}_k{k}_{hex2}`` (model basename only) so
    same-day re-runs stay unique without a wall-clock timestamp in the stem.
    """
    day = (now or datetime.now(UTC)).strftime("%Y%m%d")
    strat = token(strategy, 64)
    ag = token(agent, 48)
    if not strat or not ag:
        raise ValueError("strategy and agent are required for job_id")
    if runinfo is None:
        bits: list[str] = []
        if model and str(model).strip():
            bits.append(token(str(model).rsplit("/", 1)[-1], 32))
        if k_repeats is not None:
            bits.append(f"k{int(k_repeats)}")
        bits.append(secrets.token_hex(2))
        runinfo = "_".join(bits)
    else:
        runinfo = token(runinfo, 64)
        if not runinfo:
            raise ValueError("runinfo must be non-empty when provided")
    return f"{strat}_{ag}_{day}_{runinfo}"


def run_id(job_id: str, k_index: int) -> str:
    if k_index < 1:
        raise ValueError(f"k_index is 1-based, got {k_index}")
    return f"{job_id}-r{k_index}"


def trial_id(decision_date: date) -> str:
    return decision_date.isoformat()


def cell_id(symbol: str | None) -> str:
    from fintel.models.common import PORTFOLIO_CELL

    return symbol or PORTFOLIO_CELL
