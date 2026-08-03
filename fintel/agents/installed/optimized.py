"""Fintel adapter for Delorean's OptimizedAgent.

Boundary (architecture §1 + §8):

* Strategy package binds kinds (prices, fundamentals, ratios, news,
  filing_text, web_search).
* This adapter partitions those kinds into quantitative vs qualitative packs
  and feeds specialists — the platform does not need to know about roles.
* All reads go through ``env.access`` (``pit_enforcement='access'``).
* Cell fan-out is the platform's: one cell → one symbol under single_name.

Requires Delorean on ``PYTHONPATH`` (LangGraph / langchain live there).

Run::

    fintel simulation packages/systematic_stockrate_djia \\
      --agent optimized \\
      --model xiaomi/mimo-v2.5-pro \\
      --agent-opt enable_verification=true \\
      --cache-root ../delorean/cache
"""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from dataclasses import dataclass, field, fields
from datetime import UTC, date as Date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from delorean.agents.optimized_agent.agent import OptimizedAgent
from delorean.agents.optimized_agent.stages import set_stage_listener
from delorean.agents.tradingagent.trace import RunRecorder
from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse, SourceRef, View
from fintel.models.trace import ReasoningTrace, TraceStep, Usage

logger = logging.getLogger(__name__)


# ── Evidence packs (quant/qual via env.access) ─────────────────────────────

# Role → kinds. Must be a subset of what the strategy package binds.
QUANT_KINDS: tuple[str, ...] = ("prices", "fundamentals", "ratios")
QUAL_KINDS: tuple[str, ...] = ("news", "filing_text", "web_search")

_INCOME_KEYS = [
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "operating_expense", "net_income", "eps_diluted", "ebitda",
]
_BALANCE_KEYS = [
    "total_assets", "total_liabilities", "total_equity", "total_debt",
    "cash", "current_assets", "current_liabilities", "inventory",
]
_CASHFLOW_KEYS = [
    "net_cash_flow_operating", "net_cash_flow_investing",
    "net_cash_flow_financing", "free_cash_flow", "capital_expenditure",
    "dividends_paid",
]

DJIA_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "AMGN": "Amgen",
    "AMZN": "Amazon",
    "AXP": "American Express",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "DIS": "Disney",
    "DOW": "Dow",
    "GS": "Goldman Sachs",
    "HD": "Home Depot",
    "HON": "Honeywell",
    "IBM": "IBM",
    "INTC": "Intel",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "MCD": "McDonalds",
    "MMM": "3M",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "NKE": "Nike",
    "PG": "Procter & Gamble",
    "TRV": "Travelers",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "VZ": "Verizon",
    "WBA": "Walgreens",
    "WMT": "Walmart",
}


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e9:
        return f"{f / 1e9:.2f}B"
    if abs(f) >= 1e6:
        return f"{f / 1e6:.2f}M"
    if abs(f) >= 1e3:
        return f"{f / 1e3:.2f}K"
    return f"{f:.3f}".rstrip("0").rstrip(".")


def _trajectory(hist: list[dict], keys: list[str], label: str) -> str:
    if not hist:
        return f"{label}: n/a"
    lines: list[str] = []
    for r in hist:
        pe = r.get("period_end") or r.get("filing_date") or "?"
        kv = " ".join(f"{k}={_fmt(r.get(k))}" for k in keys if r.get(k) is not None)
        lines.append(f"  {pe}: {kv}" if kv else f"  {pe}: (no data)")
    return f"{label} (newest first):\n" + "\n".join(lines)


@dataclass
class EvidenceConfig:
    """Lookbacks / caps aligned to ``configs/optimized_agent_full.yaml``."""

    price_lookback_days: int = 365
    fundamentals_lookback_days: int = 720
    valuation_lookback_days: int = 365
    news_lookback_days: int = 50
    filing_narrative_lookback_days: int = 800
    fundamentals_max_quarters: int = 20
    valuation_history_points: int = 12
    filing_narrative_max_chars: int = 29999
    news_max_articles: int = 12
    news_include_summaries: bool = True
    news_summary_max_chars: int = 400
    web_snippets_per_query: int = 5
    web_snippet_max_chars: int = 220
    web_search_mode: str = "dual_opp_risk"
    web_search_opp_query: str = "{entity} growth catalysts outlook"
    web_search_risk_query: str = "{entity} risks lawsuit regulation"
    web_search_single_query: str = "{ticker} stock news"
    max_web_searches: int = 2
    company_names: dict[str, str] = field(default_factory=lambda: dict(DJIA_COMPANY_NAMES))


@dataclass
class FintelEvidence:
    """Prefetch + render packs for one cell (one symbol under single_name)."""

    access: Any  # fintel.environment.access.DataAccess
    symbol: str
    decision_date: Date
    config: EvidenceConfig = field(default_factory=EvidenceConfig)

    def quantitative_block(self) -> str:
        return (
            self._fundamentals_section()
            + "\n\n"
            + self._valuation_section()
            + "\n\n"
            + self._price_section()
        )

    def qualitative_block(self) -> str:
        return "\n\n".join(
            [
                self._company_news_section(),
                self._web_context_section(),
                self._filing_narrative(),
            ]
        )

    # ── reads ───────────────────────────────────────────────────────────────

    # ── quantitative ────────────────────────────────────────────────────────

    def _fundamentals_section(self) -> str:
        as_of = self.decision_date.isoformat()
        lookback = self.config.fundamentals_lookback_days
        reading = self.access.read(
            "fundamentals",
            symbol=self.symbol,
            lookback_days=lookback,
        )
        if reading.status != "ok" or not reading.data:
            detail = reading.detail or reading.status
            return (
                f"No fundamentals available for {self.symbol} "
                f"(PIT < {as_of}, lookback {lookback}d): {detail}."
            )
        raw = list(reversed(list(reading.data)))  # newest first
        panel = self._quarterly_panel(raw)
        hist = panel if panel else raw[: self.config.fundamentals_max_quarters]
        latest = hist[0]
        keys = [
            "revenue", "net_income", "eps_diluted", "gross_profit",
            "operating_income", "ebitda", "total_assets", "total_equity",
            "total_debt", "cash", "free_cash_flow", "operating_margin",
            "net_margin", "roe",
        ]
        kv = " ".join(
            f"{k}={_fmt(latest.get(k))}" for k in keys if latest.get(k) is not None
        )
        funds = (
            f"Latest fundamentals (filing={latest.get('filing_date')} "
            f"period_end={latest.get('period_end')} {latest.get('timeframe')} "
            f"{latest.get('fiscal_period')}; PIT < {as_of}; "
            f"lookback {lookback}d):\n{kv}"
        )
        traj = []
        for r in hist:
            rev, eps = _fmt(r.get("revenue")), _fmt(r.get("eps_diluted"))
            if rev != "n/a" or eps != "n/a":
                traj.append(
                    f"  {r.get('period_end')} {r.get('fiscal_period')} "
                    f"rev={rev} eps={eps}"
                )
        if traj:
            funds += (
                f"\nquarterly trajectory (newest first, "
                f"n={len(hist)} capped at {self.config.fundamentals_max_quarters}):\n"
                + "\n".join(traj)
            )
        funds += (
            "\n\n"
            + self._growth_block(panel)
            + "\n\n"
            + _trajectory(hist, _INCOME_KEYS, "Income statement")
            + "\n\n"
            + _trajectory(hist, _BALANCE_KEYS, "Balance sheet")
            + "\n\n"
            + _trajectory(hist, _CASHFLOW_KEYS, "Cash flow")
        )
        return funds

    def _quarterly_panel(self, records: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for r in records:
            if str(r.get("timeframe") or "").lower() != "quarterly":
                continue
            pe = (r.get("period_end") or "")[:10]
            if not pe or pe in seen:
                continue
            seen.add(pe)
            out.append(r)
            if len(out) >= self.config.fundamentals_max_quarters:
                break
        return out

    def _growth_block(self, panel: list[dict]) -> str:
        if not panel:
            return "Growth (QoQ / YoY / TTM): n/a (no quarterly panel)."
        latest = panel[0]
        keys = ("revenue", "eps_diluted", "net_income", "gross_profit", "free_cash_flow")

        def _f(r: dict, k: str) -> float | None:
            v = r.get(k)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _pct(a: float | None, b: float | None) -> str:
            if a is None or b is None or b == 0:
                return "n/a"
            return f"{(a / b - 1) * 100:+.1f}%"

        qoq_bits = []
        if len(panel) >= 2:
            prior = panel[1]
            for k in keys:
                qoq_bits.append(f"{k}={_pct(_f(latest, k), _f(prior, k))}")
        yoy_bits = []
        fp = latest.get("fiscal_period")
        yoy_match = next(
            (r for r in panel[1:] if r.get("fiscal_period") == fp and r is not latest),
            None,
        )
        if yoy_match is not None:
            for k in keys:
                yoy_bits.append(f"{k}={_pct(_f(latest, k), _f(yoy_match, k))}")

        ttm_keys = ("revenue", "net_income", "gross_profit", "free_cash_flow", "operating_income")
        ttm_bits = []
        if len(panel) >= 4:
            last4 = panel[:4]
            for k in ttm_keys:
                vals = [_f(r, k) for r in last4]
                if any(v is None for v in vals):
                    ttm_bits.append(f"{k}=n/a")
                else:
                    ttm_bits.append(f"{k}={_fmt(sum(vals))}")  # type: ignore[arg-type]

        return "\n".join(
            [
                f"Growth panel (latest period_end={latest.get('period_end')} "
                f"{latest.get('fiscal_period')}, n_quarters={len(panel)}, "
                f"lookback={self.config.fundamentals_lookback_days}d):",
                "  QoQ: " + (" ".join(qoq_bits) if qoq_bits else "n/a"),
                "  YoY: "
                + (" ".join(yoy_bits) if yoy_bits else "n/a (need prior-year same fiscal_period)"),
                "  TTM (sum last 4Q): "
                + (" ".join(ttm_bits) if ttm_bits else "n/a (need ≥4 quarters)"),
            ]
        )

    def _valuation_section(self) -> str:
        """Daily ratio history via fintel's computed ``ratios`` kind (Delorean shape)."""
        as_of = self.decision_date.isoformat()
        lookback = self.config.valuation_lookback_days
        reading = self.access.read(
            "ratios", symbol=self.symbol, lookback_days=lookback
        )
        if reading.status != "ok" or not isinstance(reading.data, dict):
            detail = reading.detail or reading.status
            return (
                f"No valuation ratios available for {self.symbol} "
                f"(PIT < {as_of}, lookback {lookback}d): {detail}."
            )
        data = reading.data
        entries = [
            e
            for e in (data.get("entries") or [])
            if isinstance(e, dict) and (e.get("date") or "") < as_of
        ]
        if not entries:
            # Fall back to flattened latest if entries somehow empty.
            if data.get("pe_diluted") is None and data.get("price") is None:
                return (
                    f"No valuation ratios available for {self.symbol} "
                    f"(PIT < {as_of}, lookback {lookback}d)."
                )
            entries = [{**data, "date": data.get("date") or data.get("as_of")}]

        latest = entries[-1]
        keys = [
            "pe_diluted", "ev_to_ebit", "fcf_yield", "p_b", "p_s",
            "earnings_yield", "net_margin", "roe", "debt_to_equity",
            "gross_margin", "operating_margin",
        ]
        kv = " ".join(f"{k}={_fmt(latest.get(k))}" for k in keys if latest.get(k) is not None)

        def _rng(field: str) -> str:
            xs = [float(e[field]) for e in entries if e.get(field) is not None]
            return f"{min(xs):.1f}..{max(xs):.1f}" if xs else "n/a"

        n_pts = min(len(entries), self.config.valuation_history_points)
        if n_pts <= 1:
            sampled = [latest]
        else:
            idxs = [round(i * (len(entries) - 1) / (n_pts - 1)) for i in range(n_pts)]
            seen: set[int] = set()
            sampled = []
            for i in idxs:
                if i in seen:
                    continue
                seen.add(i)
                sampled.append(entries[i])
        hist_keys = ["pe_diluted", "ev_to_ebit", "fcf_yield", "p_b", "net_margin", "roe"]
        hist_lines = []
        for e in sampled:
            parts = " ".join(
                f"{k}={_fmt(e.get(k))}" for k in hist_keys if e.get(k) is not None
            )
            hist_lines.append(
                f"  {e.get('date')}: {parts}" if parts else f"  {e.get('date')}: (n/a)"
            )

        return (
            f"Valuation ratios for {self.symbol} (latest {latest.get('date')}, "
            f"PIT < {as_of}, lookback {lookback}d, "
            f"window {entries[0].get('date')}..{latest.get('date')}):\n"
            f"{kv}\n"
            f"range over window: pe_diluted {_rng('pe_diluted')} | "
            f"ev_to_ebit {_rng('ev_to_ebit')} | fcf_yield {_rng('fcf_yield')} | "
            f"p_b {_rng('p_b')} | net_margin {_rng('net_margin')} | roe {_rng('roe')}\n"
            f"sparse history ({len(hist_lines)} pts, oldest→newest):\n"
            + "\n".join(hist_lines)
        )

    def _price_section(self) -> str:
        lookback = self.config.price_lookback_days
        as_of = self.decision_date.isoformat()
        reading = self.access.read(
            "prices", symbol=self.symbol, lookback_days=lookback
        )
        df = reading.data
        if reading.status != "ok" or df is None or not len(df):
            detail = reading.detail or reading.status
            return (
                f"Prices ({self.symbol}, PIT < {as_of}, "
                f"lookback {lookback}d): none ({detail})."
            )
        if "date" in df.columns:
            dates = df["date"].astype(str).str[:10]
            df = df.loc[dates < as_of].copy()
        if not len(df):
            return (
                f"Prices ({self.symbol}, PIT < {as_of}, "
                f"lookback {lookback}d): none after PIT filter."
            )
        last = df.iloc[-1]
        close = float(last["close"])
        last_date = str(last["date"])[:10]
        try:
            import pandas as pd

            vol = last["volume"]
            vol_s = "n/a" if pd.isna(vol) else str(int(vol))
        except Exception:  # noqa: BLE001
            vol_s = "n/a"

        def _ret(bars: int) -> tuple[str, str]:
            if len(df) < bars + 1:
                return "n/a", "n/a"
            prior = df.iloc[-(bars + 1)]
            prior_close = float(prior["close"])
            if prior_close == 0:
                return "n/a", str(prior["date"])[:10]
            pct = (close / prior_close - 1.0) * 100.0
            return f"{pct:+.1f}%", str(prior["date"])[:10]

        q_bars = 63 if lookback >= 90 else max(21, lookback // 4)
        y_bars = 252 if lookback >= 300 else max(63, int(lookback * 252 / 365))
        ret_q, from_q = _ret(q_bars)
        ret_y, from_y = _ret(y_bars)

        path_lines: list[str] = []
        if len(df) >= q_bars:
            n_pts = min(5, 1 + (len(df) - 1) // q_bars)
            for i in range(n_pts):
                idx = len(df) - 1 - i * q_bars
                if idx < 0:
                    break
                row = df.iloc[idx]
                c = float(row["close"])
                d = str(row["date"])[:10]
                if i == 0:
                    path_lines.append(f"  {d}: close={c:.2f} (latest)")
                else:
                    newer = float(df.iloc[idx + q_bars]["close"])
                    dlt = (newer / c - 1.0) * 100.0 if c else float("nan")
                    path_lines.append(
                        f"  {d}: close={c:.2f} → next≈{q_bars}d delta={dlt:+.1f}%"
                    )

        hi_lo_bars = min(len(df), y_bars if y_bars <= len(df) else len(df))
        window = df.tail(hi_lo_bars)
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        lines = [
            f"Prices ({self.symbol}, PIT < {as_of}, lookback {lookback}d):",
            f"  latest {last_date}: close={close:.2f} "
            f"open={float(last['open']):.2f} high={float(last['high']):.2f} "
            f"low={float(last['low']):.2f} "
            f"volume={vol_s}",
            f"  range over ~{hi_lo_bars} bars: high={hi:.2f} low={lo:.2f}",
            f"  quarterly delta (~{q_bars} trading days, from {from_q}): {ret_q}",
            f"  annual delta (~{y_bars} trading days, from {from_y}): {ret_y}",
        ]
        if path_lines:
            lines.append(
                f"  quarterly path (newest first, step≈{q_bars} trading days):"
            )
            lines.extend(path_lines)
        return "\n".join(lines)

    # ── qualitative ─────────────────────────────────────────────────────────

    def _company_name(self) -> str:
        return (self.config.company_names or {}).get(self.symbol) or self.symbol

    def _format_web_query(self, template: str) -> str:
        entity = self._company_name()
        return (
            template.replace("{entity}", entity)
            .replace("{ticker}", self.symbol)
            .replace("{company}", entity)
            .strip()
        )

    def _company_news_section(self) -> str:
        ceil = self.decision_date
        ceil_iso = ceil.isoformat()
        since = self.config.news_lookback_days
        window_start = (ceil - timedelta(days=since)).isoformat()
        reading = self.access.read(
            "news", symbol=self.symbol, lookback_days=since
        )
        arts = list(reading.data or []) if reading.status == "ok" else []
        pit: list[dict] = []
        for a in arts:
            pub = (a.get("published_at") or "")[:10]
            if not pub or pub >= ceil_iso or pub < window_start:
                continue
            pit.append(a)
        pit = sorted(pit, key=lambda a: a.get("published_at", ""), reverse=True)
        pit = pit[: self.config.news_max_articles]
        if not pit:
            detail = "" if reading.status == "ok" else f" ({reading.detail or reading.status})"
            return (
                f"Company news ({self.symbol}, last {since}d, "
                f"PIT {window_start} ≤ published_at < {ceil_iso}): none in the feed."
                f"{detail}"
            )
        lines: list[str] = []
        for a in pit:
            pub = (a.get("published_at") or "")[:10]
            title = (a.get("title") or "").strip()
            line = f"  {pub} — {title}"
            if self.config.news_include_summaries:
                summary = (a.get("summary") or "").strip()
                if summary:
                    if len(summary) > self.config.news_summary_max_chars:
                        summary = (
                            summary[: self.config.news_summary_max_chars].rstrip() + "…"
                        )
                    line += f"\n    summary: {summary}"
            lines.append(line)
        return (
            f"Company news ({self.symbol}, last {since}d, "
            f"PIT {window_start} ≤ published_at < {ceil_iso}, "
            f"newest first, n={len(pit)} cap={self.config.news_max_articles}"
            f"{', with summaries' if self.config.news_include_summaries else ''}):\n"
            + "\n".join(lines)
        )

    def _web_context_section(self) -> str:
        n = max(0, int(self.config.max_web_searches))
        if n <= 0 or "web_search" not in self.access.kinds:
            return ""
        since = self.config.news_lookback_days
        ceil_iso = self.decision_date.isoformat()
        window_start = (self.decision_date - timedelta(days=since)).isoformat()
        mode = self.config.web_search_mode.lower().strip()
        if mode in {"dual_opp_risk", "dual", "opp_risk"} and n >= 2:
            queries = [
                ("opportunity", self._format_web_query(self.config.web_search_opp_query)),
                ("risk", self._format_web_query(self.config.web_search_risk_query)),
            ][:2]
        else:
            queries = [
                ("general", self._format_web_query(self.config.web_search_single_query)),
            ][:n]

        blocks: list[str] = []
        seen_urls: set[str] = set()
        for label, query in queries:
            reading = self.access.read(
                "web_search",
                query=query,
                lookback_days=since,
                max_results=self.config.web_snippets_per_query,
            )
            sources = []
            if reading.status == "ok" and isinstance(reading.data, dict):
                sources = reading.data.get("sources") or []
            kept: list[dict] = []
            for s in sources:
                if not isinstance(s, dict):
                    continue
                url = (s.get("url") or "").strip()
                if url and url in seen_urls:
                    continue
                pub = (s.get("published_date") or "")[:10]
                if pub and (pub >= ceil_iso or pub < window_start):
                    continue
                if url:
                    seen_urls.add(url)
                kept.append(s)
                if len(kept) >= self.config.web_snippets_per_query:
                    break
            if not kept:
                detail = "" if reading.status == "ok" else f" [{reading.status}]"
                blocks.append(
                    f"Web context [{label}] (search \"{query}\", last {since}d, "
                    f"PIT < {ceil_iso}): none.{detail}"
                )
                continue
            slines = []
            for s in kept:
                pub = s.get("published_date") or "?"
                snip = (s.get("snippet") or s.get("description") or "")[
                    : self.config.web_snippet_max_chars
                ]
                slines.append(
                    f"  {str(pub)[:10]} — {s.get('title', '')} "
                    f"({s.get('hostname', s.get('meta_url', {}).get('hostname', ''))}): "
                    f"{snip}"
                )
            blocks.append(
                f"Web context [{label}] (search \"{query}\", last {since}d, "
                f"PIT {window_start} ≤ published < {ceil_iso}, "
                f"n={len(kept)} cap={self.config.web_snippets_per_query}):\n"
                + "\n".join(slines)
            )
        return "\n\n".join(blocks)

    def _filing_narrative(self) -> str:
        label = "Latest 10-K business section"
        lookback = self.config.filing_narrative_lookback_days
        max_chars = self.config.filing_narrative_max_chars
        reading = self.access.read(
            "filing_text",
            symbol=self.symbol,
            lookback_days=lookback,
            forms=["10-K"],
            max_chars=max_chars,
        )
        if reading.status != "ok" or not reading.data:
            detail = reading.detail or reading.status
            return f"{label}: none ({detail})."
        rows = [
            r
            for r in reading.data
            if str(r.get("form_type") or "").upper() == "10-K"
            and str(r.get("section") or "").lower() in {"business", "10-k", ""}
        ]
        if not rows:
            rows = list(reading.data)[:1]
        else:
            rows = rows[:1]
        blocks: list[str] = []
        for r in rows:
            fd = r.get("filing_date") or "?"
            sec = r.get("section") or "business"
            text = (r.get("text") or "")[:max_chars]
            age_note = ""
            try:
                age_days = (self.decision_date - Date.fromisoformat(str(fd)[:10])).days
                if age_days > 400:
                    age_note = (
                        f"\n  note: filing is {age_days}d before decision date; "
                        "pack may be missing a newer 10-K."
                    )
            except ValueError:
                pass
            blocks.append(
                f"  filing_date={fd} section={sec}{age_note}\n{text}"
            )
        return f"{label}:\n" + "\n\n".join(blocks)





# ── Adapter ──────────────────────────────────────────────────────────────────

_BOOL = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
_INT_FIELDS = {
    "price_lookback_days",
    "fundamentals_lookback_days",
    "valuation_lookback_days",
    "news_lookback_days",
    "filing_narrative_lookback_days",
    "fundamentals_max_quarters",
    "filing_narrative_max_chars",
    "news_max_articles",
    "news_summary_max_chars",
    "web_snippets_per_query",
    "web_snippet_max_chars",
    "max_web_searches",
    "specialist_max_tokens",
    "synthesis_max_tokens",
    "recursion_limit",
    "dossier_evidence_chars",
    "dossier_report_chars",
}
_BOOL_FIELDS = {
    "enable_verification",
    "news_include_summaries",
}
_FLOAT_FIELDS = {"temperature"}


def _coerce(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if key in _BOOL_FIELDS:
        return _BOOL.get(raw.lower(), raw.lower() in {"true", "1", "yes"})
    if key in _INT_FIELDS:
        return int(raw)
    if key in _FLOAT_FIELDS:
        return float(raw)
    if key == "company_names":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise TypeError("company_names must be a dict")
        return {str(k): str(v) for k, v in parsed.items()}
    return value


def _to_fintel_view(view: Any) -> View:
    raw = view.model_dump() if hasattr(view, "model_dump") else dict(view)
    sources = []
    for src in raw.get("sources_cited") or []:
        if isinstance(src, dict):
            try:
                sources.append(
                    SourceRef(
                        source_type=str(src.get("source_type") or "other"),
                        source_id=str(src.get("source_id") or ""),
                        relevance=float(src.get("relevance") or 1.0),
                        excerpt=src.get("excerpt"),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(src, str):
            sources.append(SourceRef(source_type=src, source_id=src))
    return View(
        symbol=str(raw["symbol"]),
        score=float(raw["score"]),
        conviction=float(raw.get("conviction", 0.5)),
        time_horizon=str(raw.get("time_horizon") or "quarter"),
        rationale=str(raw.get("rationale") or ""),
        key_factors=[str(x) for x in (raw.get("key_factors") or [])],
        sources_cited=sources,
    )


@dataclass
class OptimizedFintelAgent:
    """In-process OptimizedAgent host for fintel cells."""

    model: str = "xiaomi/mimo-v2.5-pro"
    profile: str = "delorean"
    temperature: float = 0.0
    reasoning_effort_quick: str | None = "high"
    reasoning_effort_deep: str | None = "high"
    specialist_max_tokens: int | None = 6000
    synthesis_max_tokens: int | None = 6000
    pm_emit_mode: str = "tool_force_strict"
    enable_verification: bool = True

    price_lookback_days: int = 365
    fundamentals_lookback_days: int = 720
    valuation_lookback_days: int = 365
    news_lookback_days: int = 50
    filing_narrative_lookback_days: int = 800
    fundamentals_max_quarters: int = 20
    filing_narrative_max_chars: int = 29999
    news_max_articles: int = 12
    news_include_summaries: bool = True
    news_summary_max_chars: int = 400
    web_snippets_per_query: int = 5
    web_snippet_max_chars: int = 220
    web_search_mode: str = "dual_opp_risk"
    web_search_opp_query: str = "{entity} growth catalysts outlook"
    web_search_risk_query: str = "{entity} risks lawsuit regulation"
    web_search_single_query: str = "{ticker} stock news"
    max_web_searches: int = 2
    company_names: dict[str, str] = field(
        default_factory=lambda: dict(DJIA_COMPANY_NAMES)
    )
    recursion_limit: int = 30
    dossier_evidence_chars: int = 16000
    dossier_report_chars: int = 8000

    # Strategy pack context (wired by simulate.cell.build_agent; optional).
    mission_text: str = ""
    output_schema_text: str = ""

    name: str = "optimized"
    version: str = "0.3.5"
    pit_enforcement: ClassVar[PitEnforcement] = "access"

    _inner: OptimizedAgent | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, str) and f.name in (
                _INT_FIELDS | _BOOL_FIELDS | _FLOAT_FIELDS | {"company_names"}
            ):
                setattr(self, f.name, _coerce(f.name, val))

    @staticmethod
    def preflight_checks(**params: Any) -> list[str]:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return [
                "OPENROUTER_API_KEY is not set; the optimized agent cannot call OpenRouter"
            ]
        return []

    def _agent(self) -> OptimizedAgent:
        if self._inner is None:
            self._inner = OptimizedAgent(
                model=self.model,
                profile=self.profile,
                temperature=self.temperature,
                reasoning_effort_quick=self.reasoning_effort_quick,
                reasoning_effort_deep=self.reasoning_effort_deep,
                specialist_max_tokens=self.specialist_max_tokens,
                synthesis_max_tokens=self.synthesis_max_tokens,
                pm_emit_mode=self.pm_emit_mode,
                enable_verification=self.enable_verification,
                price_lookback_days=self.price_lookback_days,
                fundamentals_lookback_days=self.fundamentals_lookback_days,
                valuation_lookback_days=self.valuation_lookback_days,
                news_lookback_days=self.news_lookback_days,
                filing_narrative_lookback_days=self.filing_narrative_lookback_days,
                fundamentals_max_quarters=self.fundamentals_max_quarters,
                filing_narrative_max_chars=self.filing_narrative_max_chars,
                news_max_articles=self.news_max_articles,
                news_include_summaries=self.news_include_summaries,
                news_summary_max_chars=self.news_summary_max_chars,
                web_snippets_per_query=self.web_snippets_per_query,
                web_snippet_max_chars=self.web_snippet_max_chars,
                web_search_mode=self.web_search_mode,
                web_search_opp_query=self.web_search_opp_query,
                web_search_risk_query=self.web_search_risk_query,
                web_search_single_query=self.web_search_single_query,
                max_web_searches=self.max_web_searches,
                company_names=dict(self.company_names or {}),
                max_concurrent=1,  # platform fans out cells
                recursion_limit=self.recursion_limit,
                dossier_evidence_chars=self.dossier_evidence_chars,
                dossier_report_chars=self.dossier_report_chars,
            )
        return self._inner

    def _evidence_config(self) -> EvidenceConfig:
        return EvidenceConfig(
            price_lookback_days=self.price_lookback_days,
            fundamentals_lookback_days=self.fundamentals_lookback_days,
            valuation_lookback_days=self.valuation_lookback_days,
            news_lookback_days=self.news_lookback_days,
            filing_narrative_lookback_days=self.filing_narrative_lookback_days,
            fundamentals_max_quarters=self.fundamentals_max_quarters,
            valuation_history_points=12,
            filing_narrative_max_chars=self.filing_narrative_max_chars,
            news_max_articles=self.news_max_articles,
            news_include_summaries=self.news_include_summaries,
            news_summary_max_chars=self.news_summary_max_chars,
            web_snippets_per_query=self.web_snippets_per_query,
            web_snippet_max_chars=self.web_snippet_max_chars,
            web_search_mode=self.web_search_mode,
            web_search_opp_query=self.web_search_opp_query,
            web_search_risk_query=self.web_search_risk_query,
            web_search_single_query=self.web_search_single_query,
            max_web_searches=self.max_web_searches,
            company_names=dict(self.company_names or {}),
        )

    def decide(self, env: Environment) -> AgentResponse:
        symbols = tuple(sorted(env.policy.decidable))
        if not symbols:
            return AgentResponse(views={}, outcome="empty", detail="no decidable symbols")

        # single_name cells are one symbol; portfolio would pass all — still
        # run sequentially so each pack stays role-partitioned.
        views: dict[str, View] = {}
        steps: list[TraceStep] = []
        dossiers: dict[str, dict] = {}
        errors: list[str] = []
        usage = Usage()

        nerve = env.nerve
        cell_name = env.cell.name
        decision_date = env.cell.decision_date.isoformat()
        for sym in symbols:
            self._emit(
                nerve,
                "agent_stage",
                cell=cell_name,
                decision_date=decision_date,
                stage="evidence",
                symbol=sym,
            )

            builder = FintelEvidence(
                access=env.access,
                symbol=sym,
                decision_date=env.cell.decision_date,
                config=self._evidence_config(),
            )
            # Specialist split: only the kinds each role needs.
            # Record evidence stages on the Delorean recorder so pipeline_health
            # does not flag them missing (nerve stages alone are not enough).
            recorder = RunRecorder()
            t0 = time.monotonic()
            with _stage(nerve, "evidence_quantitative", cell_name, decision_date, sym):
                quant = builder.quantitative_block()
            recorder.add_stage(
                "evidence_quantitative", (time.monotonic() - t0) * 1000.0
            )
            t0 = time.monotonic()
            with _stage(nerve, "evidence_qualitative", cell_name, decision_date, sym):
                qual = builder.qualitative_block()
            recorder.add_stage(
                "evidence_qualitative", (time.monotonic() - t0) * 1000.0
            )

            def _on_graph_stage(stage_name: str, _sym: str = sym) -> None:
                self._emit(
                    nerve,
                    "agent_stage",
                    cell=cell_name,
                    decision_date=decision_date,
                    stage=stage_name,
                    symbol=_sym,
                )

            set_stage_listener(_on_graph_stage)
            try:
                with _stage(nerve, "pipeline", cell_name, decision_date, sym):
                    view, dossier, recorder, decision_md, error = self._agent().decide_one(
                        symbol=sym,
                        decision_date=env.cell.decision_date,
                        quantitative_evidence=quant,
                        qualitative_evidence=qual,
                        recorder=recorder,
                    )
            finally:
                set_stage_listener(None)

            self._persist_dossier(env, sym, dossier)
            dossiers[sym] = dossier
            steps.extend(self._trace_steps(sym, recorder, decision_md, error, dossier))
            usage = usage.merge(self._usage_from_recorder(recorder))
            if error:
                errors.append(f"{sym}: {error}")
                logger.warning("optimized fintel: %s failed — %s", sym, error)
            elif view is not None:
                views[sym] = _to_fintel_view(view)

        outcome: Outcome
        detail = ""
        if views:
            outcome = "ok"
        elif errors:
            outcome = "crashed"
            detail = "; ".join(errors)[:2000]
        else:
            outcome = "abstained"
            detail = "pipeline produced no views (gates or empty PM emit)"

        return AgentResponse(
            views=views,
            outcome=outcome,
            detail=detail,
            usage=usage,
            trace=ReasoningTrace(
                final_explanation=(
                    f"OptimizedAgent: {len(views)}/{len(symbols)} views on "
                    f"{env.cell.decision_date} via {self.model}; "
                    + (
                        "quant/qual → structured-verify → final-PM."
                        if self.enable_verification
                        else "quant/qual → final-PM (verification off)."
                    )
                ),
                steps=steps,
                metadata={
                    "agent": self.name,
                    "version": self.version,
                    "model": self.model,
                    "enable_verification": self.enable_verification,
                    "quant_kinds": ["prices", "fundamentals", "ratios"],
                    "qual_kinds": ["news", "filing_text", "web_search"],
                    "errors": errors,
                    "dossiers": dossiers,
                    "mission_digest_present": bool(self.mission_text.strip()),
                },
            ),
        )

    @staticmethod
    def _usage_from_recorder(recorder: Any) -> Usage:
        n = 0
        tin = tout = 0
        cost = 0.0
        saw_cost = False
        for call in getattr(recorder, "llm_calls", []) or []:
            n += 1
            tin += int(getattr(call, "tokens_in", 0) or 0)
            tout += int(getattr(call, "tokens_out", 0) or 0)
            c = getattr(call, "cost_usd", None)
            if c is not None:
                cost += float(c)
                saw_cost = True
        if n == 0:
            return Usage()
        return Usage(
            n_llm_calls=n,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost if saw_cost else None,
            basis="reported" if saw_cost else "unknown",
        )

    def _trace_steps(
        self,
        sym: str,
        recorder: Any,
        decision_md: str,
        error: str | None,
        dossier: dict,
    ) -> list[TraceStep]:
        out: list[TraceStep] = []
        for index, call in enumerate(getattr(recorder, "llm_calls", []) or []):
            started = datetime.now(UTC)
            if getattr(call, "started_at", None):
                try:
                    started = datetime.fromisoformat(call.started_at)
                except ValueError:
                    pass
            out.append(
                TraceStep(
                    step_id=f"optimized-llm-{sym}-{index}",
                    kind="llm_call",
                    started_at=started,
                    duration_ms=int(getattr(call, "duration_ms", 0) or 0),
                    model=getattr(call, "model", None) or self.model,
                    tokens_in=getattr(call, "tokens_in", None),
                    tokens_out=getattr(call, "tokens_out", None),
                    cost_usd=getattr(call, "cost_usd", None),
                    payload={
                        "ticker": sym,
                        "stage": getattr(call, "stage", "") or "",
                        "cost_basis": "reported"
                        if getattr(call, "cost_usd", None) is not None
                        else "unknown",
                    },
                )
            )
        summary: dict[str, Any] = {
            "ticker": sym,
            "pipeline": (
                "quant‖qual → verify → final_pm"
                if self.enable_verification
                else "quant‖qual → final_pm"
            ),
            "gate_summary": dossier.get("gate_summary"),
            "pm_emit_path": dossier.get("pm_emit_path"),
            "health": dossier.get("health"),
        }
        if error:
            summary["error"] = error
        if decision_md:
            summary["final_decision"] = decision_md[:5000]
        out.append(
            TraceStep(
                step_id=f"optimized-summary-{sym}",
                kind="other",
                started_at=datetime.now(UTC),
                model=self.model,
                payload=summary,
            )
        )
        return out

    def _persist_dossier(self, env: Environment, symbol: str, dossier: dict) -> None:
        """Write specialist/PM dossier into the cell session (adapter-owned)."""
        session = getattr(env, "session", None)
        if session is None or not getattr(session, "path", None):
            return
        try:
            path = Path(session.path)
            path.mkdir(parents=True, exist_ok=True)
            payload = {
                "symbol": symbol,
                "decision_date": env.cell.decision_date.isoformat(),
                "cell": env.cell.name,
                "agent": self.name,
                "version": self.version,
                "model": self.model,
                **dossier,
            }
            (path / "dossier.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.debug("failed to persist dossier for %s", symbol, exc_info=True)

    def _emit(self, nerve: Any, kind: str, **fields: Any) -> None:
        if nerve is None or not hasattr(nerve, "emit"):
            return
        try:
            nerve.emit(kind, **fields)
        except Exception:  # noqa: BLE001
            pass


class _stage:
    def __init__(
        self,
        nerve: Any,
        name: str,
        cell: str,
        decision_date: str,
        symbol: str,
    ) -> None:
        self.nerve = nerve
        self.name = name
        self.cell = cell
        self.decision_date = decision_date
        self.symbol = symbol

    def __enter__(self) -> None:
        self._emit(
            "agent_stage",
            cell=self.cell,
            decision_date=self.decision_date,
            stage=self.name,
            symbol=self.symbol,
        )

    def __exit__(self, *exc: Any) -> None:
        return None

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.nerve is None or not hasattr(self.nerve, "emit"):
            return
        try:
            self.nerve.emit(kind, **fields)
        except Exception:  # noqa: BLE001
            pass


__all__ = ["OptimizedFintelAgent"]
