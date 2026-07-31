"""The one naming scheme. Nothing else may invent an id format."""

from __future__ import annotations

import re
import secrets
from datetime import UTC, date, datetime

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 40) -> str:
    return _SLUG.sub("-", text.strip().lower()).strip("-")[:limit].strip("-")


def new_job_id(*, strategy: str, agent: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{slug(strategy)}-{slug(agent, 16)}-{stamp}-{secrets.token_hex(2)}"


def run_id(job_id: str, k_index: int) -> str:
    if k_index < 1:
        raise ValueError(f"k_index is 1-based, got {k_index}")
    return f"{job_id}-r{k_index}"


def trial_id(decision_date: date) -> str:
    return decision_date.isoformat()


def cell_id(symbol: str | None) -> str:
    from fintel.models.common import PORTFOLIO_CELL

    return symbol or PORTFOLIO_CELL
