"""Filing text — 8-K item text and 10-K sections, clamped on filing_date.

Two upstream shapes behind one kind: 8-K text is sorted by filing date, while
the 10-K sections endpoint rejects that sort and needs period_end. The caller
asks for `filing_text` and doesn't care.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from fintel.market.data import coverage as cov
from fintel.market.data.base import DataError, require
from fintel.market.data.http import MassiveClient
from fintel.market.data.store import RecordCache
from fintel.models.common import Symbol
from fintel.pit import Cutoff

logger = logging.getLogger(__name__)

# The only 10-K sections the provider exposes.
TEN_K_SECTIONS: tuple[str, ...] = ("business", "risk_factors")
DEFAULT_FORMS: tuple[str, ...] = ("8-K", "10-K")


def normalise_filing_text(
    item: dict, *, symbol: Symbol, form: str, text_key: str, default_section: str
) -> dict | None:
    filing_date = str(item.get("filing_date") or "")[:10]
    if not filing_date:
        return None
    text = str(item.get(text_key) or item.get("text") or "").strip()
    section = item.get("section") or default_section
    accession = item.get("accession_number") or ""
    return {
        "id": accession or f"{form.upper()}:{filing_date}:{section}",
        "ticker": item.get("ticker", symbol),
        "form_type": form.upper(),
        "filing_date": filing_date,
        "period_end": str(item.get("period_end") or "")[:10],
        "section": section,
        "text": text,
    }


def _request(form: str, section: str | None) -> tuple[str, dict, str, str]:
    """(path, extra params, text key, default section) for one upstream call."""
    if form.upper() == "8-K":
        return "/stocks/filings/8-K/vX/text", {"sort": "filing_date.desc"}, "items_text", "8-K"
    params: dict = {"sort": "period_end.desc"}
    if section:
        params["section"] = section
    return "/stocks/filings/10-K/vX/sections", params, "text", section or "10-K"


@dataclass
class MassiveFilingText:
    cache: RecordCache
    client: MassiveClient | None = None
    forms: tuple[str, ...] = DEFAULT_FORMS
    sections: tuple[str, ...] = TEN_K_SECTIONS
    lookback_days: int = 730
    name: str = "massive_filing_text"
    kinds: tuple[str, ...] = ("filing_text",)

    def fetch(self, query: dict, cutoff: Cutoff) -> list[dict]:
        symbol = require(query, "symbol", self.name)
        lookback = int(query.get("lookback_days", self.lookback_days))
        since = cutoff.decision_date - timedelta(days=lookback)
        through = cutoff.decision_date - timedelta(days=1)

        records = self._ensure(symbol, since, through)
        out = cutoff.clamp_records(records, "filing_date")
        out = [r for r in out if r["filing_date"] >= since.isoformat()]
        if wanted := query.get("forms"):
            keep = {str(f).upper() for f in wanted}
            out = [r for r in out if r["form_type"] in keep]
        if query.get("max_chars"):
            limit = int(query["max_chars"])
            out = [{**r, "text": r["text"][:limit]} for r in out]
        return out

    def _ensure(self, symbol: Symbol, since: Date, through: Date) -> list[dict]:
        coverage, records = self.cache.read(symbol)
        gaps = cov.missing(coverage, since, through)
        if not gaps:
            return records
        if self.client is None:
            if not coverage:
                raise DataError(
                    f"{self.name}: nothing cached for {symbol} filing_text covering "
                    f"[{since}, {through}] and no network access configured"
                )
            logger.warning(
                "%s: %s cache is short of [%s, %s]; serving what is cached",
                self.name, symbol, since, through,
            )
            return records

        fresh: dict[str, dict] = {}
        for lo, hi in gaps:
            for form in self.forms:
                sections = self.sections if form.upper() == "10-K" else (None,)
                for section in sections:
                    for rec in self._fetch_one(symbol, form, section, lo, hi):
                        fresh[rec["id"]] = rec
        return self.cache.merge(
            symbol,
            list(fresh.values()),
            gaps,
            key=lambda r: r["id"],
            sort=lambda r: (r["filing_date"], r["id"]),
        )

    def _fetch_one(
        self, symbol: Symbol, form: str, section: str | None, lo: Date, hi: Date
    ) -> list[dict]:
        assert self.client is not None
        path, extra, text_key, default_section = _request(form, section)
        params = {
            "ticker": symbol,
            "filing_date.gte": lo.isoformat(),
            "filing_date.lt": (hi + timedelta(days=1)).isoformat(),
            "limit": 50,
            **extra,
        }
        out = []
        for item in self.client.paginate(path, params):
            rec = normalise_filing_text(
                item, symbol=symbol, form=form, text_key=text_key,
                default_section=default_section,
            )
            if rec is not None:
                out.append(rec)
        return out
