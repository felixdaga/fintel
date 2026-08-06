"""Evidence packs for the optimized agent (delorean-free).

The pure-fintel rendering layer used by the native optimized agent
(``fintel/agents/adapters/optimized.py``). It builds a ``FintelEvidence``
from ``env.access`` and renders quantitative / qualitative text blocks; neither
this module nor anything it imports depends on Delorean, LangGraph, or
langchain.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


QUANT_KINDS: tuple[str, ...] = ("prices", "fundamentals", "ratios", "macro", "news_sentiment")
QUAL_KINDS: tuple[str, ...] = ("news", "filing_text", "web_search")

_INCOME_KEYS = [
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "operating_expense", "net_income", "eps_diluted", "ebitda",
]
_BALANCE_KEYS = [
    "total_assets", "total_liabilities", "total_equity", "total_debt",
    "cash", "current_assets", "current_liabilities", "inventory",
]
# Must match catalog / massive.normalise_financial field names — not the raw
# Polygon statement labels (those are mapped at normalise time).
_CASHFLOW_KEYS = [
    "operating_cash_flow", "capex", "free_cash_flow",
]

# Company-name fallback moved to the strategy package (company_names.json).
# A different universe ships its own; the PIT constituents resolver is the
# primary source and this dict is only the static fallback when it lacks a name.


# ── Point-in-time company naming ───────────────────────────────────────────
# The constituents dataset records each membership period with the company's
# name *as it was known then* (e.g. WMT: "Wal-Mart Stores, Inc." until
# 2018-06-26, then "Walmart Inc."). Resolving the name per (symbol, decision
# date) — rather than from the static dict above — means web-search queries use
# the contemporary entity name, which returns markedly better results than the
# ticker, and never leak a future rename back into a past decision (PIT).
_PIT_NAMES_CACHE: dict[str, list[tuple[str, str, str, str]]] = {}
# key: constituents dir path → rows of (symbol, name, opt_in_iso, opt_out_iso)


def _load_pit_names(constituents_dir: Path) -> list[tuple[str, str, str, str]]:
    """Read every cached index CSV once per run; keep the `name` column.

    The on-disk CSV is the raw dataset (with `name`); only the in-memory
    `normalise()` for membership drops it, so we read the file directly.
    """
    key = str(constituents_dir)
    cached = _PIT_NAMES_CACHE.get(key)
    if cached is not None:
        return cached
    rows: list[tuple[str, str, str, str]] = []
    if not constituents_dir.exists():
        _PIT_NAMES_CACHE[key] = rows
        return rows

    for csv_path in sorted(constituents_dir.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    sym = (r.get("symbol") or "").strip()
                    name = (r.get("name") or "").strip()
                    opt_in = (r.get("opt-in") or "")[:10]
                    opt_out_raw = r.get("opt-out")
                    opt_out = "9999-12-31" if not opt_out_raw else opt_out_raw[:10]
                    if sym and name and opt_in:
                        rows.append((sym, name, opt_in, opt_out))
        except Exception as exc:  # corrupt/missing file: skip, fall back per row
            logger.debug("pit names: skipped %s: %s", csv_path, exc)
    _PIT_NAMES_CACHE[key] = rows
    return rows


def _pit_company_name(market_config: Any, symbol: str, decision_date: Date) -> str | None:
    """Resolve the company name as known at `decision_date`, PIT.

    Picks the membership period with the latest opt-in at or before the
    decision date — the period that contains the date in the normal
    contiguous case, and the most-recent prior name if there is a dataset
    gap. Never returns a name from a period that starts after the decision
    date, so a future rename cannot leak backwards.
    """
    if market_config is None:
        return None
    try:
        constituents_dir = market_config.dir("constituents")
    except Exception:
        return None
    rows = _load_pit_names(Path(constituents_dir))
    if not rows:
        return None
    d = decision_date.isoformat()
    best_in: str | None = None
    best_name: str | None = None
    for sym, name, opt_in, _opt_out in rows:
        if sym != symbol or opt_in > d:
            continue
        if best_in is None or opt_in > best_in:
            best_in = opt_in
            best_name = name
    return best_name


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


# Two-tier web search plan for a fundamental/trajectory rater.
#
# Structural tier — slow-moving dimensions that need a 30d window to capture
# (business model, competitive position, risks, capital-allocation strategy).
# These map onto what a 10-K business section used to give, but fresher and
# richer (structured guidance tables, segment breakdowns).
WEB_STRUCTURAL_QUERIES: tuple[tuple[str, str], ...] = (
    ("business", "{entity} business model segments revenue"),
    ("competitive", "{entity} competitive position market share"),
    ("risk", "{entity} risks headwinds regulation"),
    ("strategy", "{entity} management strategy capital allocation"),
)
# Updates tier — last week only. Overlaps with the `news` kind (14d); that's
# intentional: this catches recent earnings/development catalysts the
# fundamental view should be aware of, without spending a 30d window on them.
WEB_UPDATE_QUERIES: tuple[tuple[str, str], ...] = (
    ("updates", "{entity} latest news developments earnings"),
)


def _trajectory(hist: list[dict], keys: list[str], label: str) -> str:
    if not hist:
        return f"{label}: n/a"
    lines: list[str] = []
    for r in hist:
        pe = r.get("period_end") or r.get("filing_date") or "?"
        kv = " ".join(f"{k}={_fmt(r.get(k))}" for k in keys if r.get(k) is not None)
        lines.append(f"  {pe}: {kv}" if kv else f"  {pe}: (no data)")
    return f"{label} (newest first):\n" + "\n".join(lines)


# Fiscal-period predecessor (newest → next-older) for a standard 4-quarter year.
_PREV_FISCAL_PERIOD = {"Q1": "Q4", "Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}


def quarters_contiguous(panel_newest_first: list[dict]) -> tuple[bool, str]:
    """Whether successive quarterly rows form one contiguous fiscal chain.

    Used before labeling a sum as TTM. A gap (e.g. missing Q4 between Q1 and Q3)
    must not be called trailing twelve months — that was the AAPL smoke failure
    mode. Returns ``(ok, detail)``; ``detail`` explains a gap or confirms the
    period chain.
    """
    if len(panel_newest_first) < 2:
        return True, "n/a (fewer than 2 quarters)"
    fps = [
        str(r.get("fiscal_period") or "").strip().upper()
        for r in panel_newest_first
    ]
    if all(fp in _PREV_FISCAL_PERIOD for fp in fps):
        for i in range(len(fps) - 1):
            if fps[i + 1] != _PREV_FISCAL_PERIOD[fps[i]]:
                return (
                    False,
                    f"fiscal gap: {'→'.join(fps)} "
                    f"(expected {_PREV_FISCAL_PERIOD[fps[i]]} after {fps[i]})",
                )
        return True, "→".join(fps)

    # Fall back to period_end spacing when fiscal_period is missing/nonstandard.
    ends: list[Date] = []
    for r in panel_newest_first:
        pe = (r.get("period_end") or "")[:10]
        try:
            ends.append(Date.fromisoformat(pe))
        except ValueError:
            return False, f"unparseable period_end={pe!r}"
    for i in range(len(ends) - 1):
        gap = (ends[i] - ends[i + 1]).days
        if gap < 45 or gap > 120:
            return (
                False,
                f"period_end gap {gap}d between {ends[i]} and {ends[i + 1]} "
                f"(need ~1 fiscal quarter)",
            )
    return True, "period_end spacing ok"


@dataclass
class EvidenceConfig:
    """Presentation caps for evidence packs.

    Lookbacks are not here: the strategy's ``[[data]].lookback_days`` is the
    single source of truth, read from ``env.policy.lookback_cap_map`` at pack
    build time. Keeping lookbacks out of this config is what stops the
    fundamentals-720-vs-730 drift from recurring.

    Rendering philosophy: the lookback is the *only* deliberate *record-level*
    filter — every record inside the window is rendered. The only per-item
    truncations are the render caps (``snippet_max_chars`` for web_search,
    ``summary_max_chars`` for news), whose defaults live in the markets
    catalog and which a strategy may override via ``[[data]].params``; they
    bound the *text* of each item, not whether it appears. The single
    ``evidence_budget_chars`` ceiling is the safety net: when the cumulative
    pack would exceed it, the lowest-priority sections are truncated
    (visibly) so the context window is never blown. With a generous budget
    it almost never fires; the lookback and render caps do the work.
    """

    # The one truncation knob: total chars across the whole quant (or qual) pack.
    evidence_budget_chars: int = 400_000
    valuation_history_points: int = 12
    # Two-tier web search: structural (slow-moving, 30d) + updates (last week).
    # The structural lookback honours the strategy binding; the update tier is
    # a fixed short window. Per-query results are shown in full (no per-snippet
    # char cap); the evidence budget governs total size.
    web_structural_lookback_days: int = 30
    web_update_lookback_days: int = 7
    web_snippets_per_query: int = 5
    company_names: dict[str, str] = field(default_factory=dict)


@dataclass
class FintelEvidence:
    """Prefetch + render packs for one cell (one symbol under single_name)."""

    access: Any  # fintel.environment.access.DataAccess
    symbol: str
    decision_date: Date
    config: EvidenceConfig = field(default_factory=EvidenceConfig)
    # Point-in-time company name for `symbol` resolved from the constituents
    # dataset at `decision_date`. When set it overrides the static map so
    # web-search queries use the contemporary entity name per period.
    company_name: str | None = None

    def _lookback(self, kind: str, default: int = 365) -> int:
        """The strategy's lookback for this kind (single source of truth)."""
        caps = getattr(self.access.policy, "lookback_cap_map", {})
        return int(caps.get(kind, default))

    def _render_cap(self, kind: str, param: str, default: int) -> int:
        """A per-kind render cap (e.g. web_search.snippet_max_chars).

        Defaults live in the markets catalog; a strategy may override via
        ``[[data]].params``. The policy surfaces the resolved value; this
        falls back to ``default`` when neither declares one.
        """
        rc = getattr(self.access.policy, "render_cap_map", {}) or {}
        per_kind = rc.get(kind) or {}
        return int(per_kind.get(param, default))

    def _assemble_budgeted(
        self, sections: list[tuple[str, str]], budget: int
    ) -> str:
        """Render every section in priority order; truncate only at the budget.

        The lookback already filtered what's relevant, so each section is kept
        whole. When the cumulative size would exceed ``budget``, the section
        that crosses the line is truncated to fit and a visible note is appended;
        any later sections are dropped (also noted). With a generous budget this
        almost never fires — it is a safety net for the context window, not a
        per-kind cap.
        """
        out: list[str] = []
        used = 0
        notes: list[str] = []
        for label, text in sections:
            if not text:
                continue
            if used >= budget:
                notes.append(f"{label}: dropped (budget reached)")
                continue
            remaining = budget - used
            if len(text) <= remaining:
                out.append(text)
                used += len(text) + 2
            else:
                kept = text[: max(0, remaining)].rstrip()
                out.append(
                    kept
                    + f"\n…truncated at evidence budget ({len(text) - len(kept)} "
                    f"of {len(text)} chars dropped in {label})"
                )
                notes.append(f"{label}: partial")
                used = budget
        if notes:
            out.append("…evidence budget note: " + "; ".join(notes))
        return "\n\n".join(out)

    def quantitative_block(self) -> str:
        sections = [
            ("fundamentals", self._fundamentals_section()),
            ("valuation", self._valuation_section()),
            ("prices", self._price_section()),
            ("macro", self._macro_section()),
            ("news_sentiment", self._news_sentiment_section()),
        ]
        return self._assemble_budgeted(
            sections, self.config.evidence_budget_chars
        )

    def qualitative_block(self) -> str:
        sections = [
            ("news", self._company_news_section()),
            ("web_search", self._web_context_section()),
            ("filing_text", self._filing_narrative()),
        ]
        return self._assemble_budgeted(
            sections, self.config.evidence_budget_chars
        )

    # ── reads ───────────────────────────────────────────────────────────────

    # ── quantitative ────────────────────────────────────────────────────────

    def _fundamentals_section(self) -> str:
        as_of = self.decision_date.isoformat()
        lookback = self._lookback("fundamentals", 720)
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
        hist = panel if panel else raw  # all records in the lookback window
        latest = hist[0]
        keys = [
            "revenue", "net_income", "eps_diluted", "gross_profit",
            "operating_income", "ebitda", "total_assets", "total_equity",
            "total_debt", "cash", "operating_cash_flow", "capex",
            "free_cash_flow", "operating_margin", "net_margin", "roe",
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
                f"\nquarterly trajectory (newest first, n={len(hist)} "
                f"— all in window):\n"
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
        return out

    def _growth_block(self, panel: list[dict]) -> str:
        if not panel:
            return "Growth (QoQ / YoY / TTM): n/a (no quarterly panel)."
        latest = panel[0]
        keys = (
            "revenue", "eps_diluted", "net_income", "gross_profit",
            "operating_cash_flow", "free_cash_flow",
        )

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

        ttm_keys = (
            "revenue", "net_income", "gross_profit", "operating_income",
            "operating_cash_flow", "free_cash_flow",
        )
        if len(panel) < 4:
            ttm_line = "TTM: n/a (need ≥4 quarters)"
        else:
            last4 = panel[:4]
            ok, detail = quarters_contiguous(last4)
            fps = ",".join(str(r.get("fiscal_period") or "?") for r in last4)
            if not ok:
                # Do NOT print a summed "TTM" — that was the AAPL false-TTM bug.
                ttm_line = (
                    f"TTM: incomplete — not 4 contiguous quarters "
                    f"[{fps}; {detail}]. Do not treat a sum of the last 4 "
                    f"available quarters as trailing twelve months."
                )
            else:
                ttm_bits = []
                for k in ttm_keys:
                    vals = [_f(r, k) for r in last4]
                    if any(v is None for v in vals):
                        ttm_bits.append(f"{k}=n/a")
                    else:
                        ttm_bits.append(f"{k}={_fmt(sum(vals))}")  # type: ignore[arg-type]
                ttm_line = (
                    f"TTM (sum of 4 contiguous quarters [{detail}]): "
                    + (" ".join(ttm_bits) if ttm_bits else "n/a")
                )

        return "\n".join(
            [
                f"Growth panel (latest period_end={latest.get('period_end')} "
                f"{latest.get('fiscal_period')}, n_quarters={len(panel)}, "
                f"lookback={self._lookback('fundamentals', 720)}d):",
                "  QoQ: " + (" ".join(qoq_bits) if qoq_bits else "n/a"),
                "  YoY: "
                + (" ".join(yoy_bits) if yoy_bits else "n/a (need prior-year same fiscal_period)"),
                f"  {ttm_line}",
            ]
        )

    def _valuation_section(self) -> str:
        """Daily ratio history via fintel's computed ``ratios`` kind (Delorean shape)."""
        as_of = self.decision_date.isoformat()
        lookback = self._lookback("ratios", 365)
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
        lookback = self._lookback("prices", 365)
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

    def _macro_section(self) -> str:
        """FRED macro bundle — all observations in the window, compact.

        The lookback (90d) is the filter; every observation in the window is
        rendered as ``date:value`` so the agent can read the trend directly.
        The source's ``change`` field is the window-length trend. The budget
        ceiling truncates if a long lookback ever makes this huge.
        """
        if "macro" not in self.access.kinds:
            return ""
        as_of = self.decision_date.isoformat()
        lookback = self._lookback("macro", 90)
        reading = self.access.read("macro", lookback_days=lookback)
        if reading.status != "ok" or not reading.data:
            detail = reading.detail or reading.status
            return f"Macro (PIT < {as_of}, lookback {lookback}d): none ({detail})."
        series = reading.data.get("series") or [] if isinstance(reading.data, dict) else []
        if not series:
            return f"Macro (PIT < {as_of}, lookback {lookback}d): no series returned."
        lines = [f"Macro (PIT < {as_of}, lookback {lookback}d, {len(series)} series):"]
        for s in series:
            sid = s.get("series_id", "?")
            title = (s.get("title") or sid)[:48]
            units = s.get("units") or ""
            freq = (s.get("frequency") or "")[:18]
            latest = s.get("latest_date")
            val = s.get("latest_value")
            chg = s.get("change")
            chg_pct = s.get("change_pct")
            obs = s.get("observations") or []
            lines.append(
                f"  {sid} ({freq}, {units}) latest={latest}={val} "
                f"Δ={chg} ({chg_pct}%) n={len(obs)} — {title}"
            )
            if obs:
                # All observations in the window, compact date:value.
                pts = " ".join(f"{o.get('date')}:{o.get('value')}" for o in obs)
                lines.append(f"    series: {pts}")
        return "\n".join(lines)

    def _news_sentiment_section(self) -> str:
        """Computed daily sentiment series — all points in the window.

        The quantitative replacement for reading hundreds of raw headlines:
        one score + count per day over the lookback, so the agent reads the
        sentiment trend (improving/deteriorating, bull/bear share) directly.
        """
        if "news_sentiment" not in self.access.kinds:
            return ""
        as_of = self.decision_date.isoformat()
        lookback = self._lookback("news_sentiment", 90)
        reading = self.access.read("news_sentiment", symbol=self.symbol, lookback_days=lookback)
        if reading.status != "ok" or not reading.data:
            detail = reading.detail or reading.status
            return f"News sentiment ({self.symbol}, PIT < {as_of}, lookback {lookback}d): none ({detail})."
        d = reading.data
        n_art = d.get("n_articles")
        n_scored = d.get("n_scored")
        mean = d.get("mean_score")
        series = d.get("series") or []
        if not series:
            return (
                f"News sentiment ({self.symbol}, PIT < {as_of}, lookback {lookback}d): "
                f"no scored days (n_articles={n_art})."
            )
        lines = [
            f"News sentiment ({self.symbol}, PIT < {as_of}, lookback {lookback}d, "
            f"{len(series)} days; n_articles={n_art} n_scored={n_scored} mean={mean}):"
        ]
        # All daily points, oldest→newest: date:score(n=count)
        pts = " ".join(
            f"{p.get('date')}:{p.get('score')}(n={p.get('n')})" for p in series
        )
        lines.append(f"  series: {pts}")
        return "\n".join(lines)

    # ── qualitative ─────────────────────────────────────────────────────────

    def _company_name(self) -> str:
        if self.company_name:
            return self.company_name
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
        since = self._lookback("news", 14)
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
        if not pit:
            detail = "" if reading.status == "ok" else f" ({reading.detail or reading.status})"
            return (
                f"Company news ({self.symbol}, last {since}d, "
                f"PIT {window_start} ≤ published_at < {ceil_iso}): none in the feed."
                f"{detail}"
            )
        lines: list[str] = []
        cap = self._render_cap("news", "summary_max_chars", 640)
        for a in pit:
            pub = (a.get("published_at") or "")[:10]
            title = (a.get("title") or "").strip()
            publisher = (a.get("publisher") or "").strip()
            line = f"  {pub} — {publisher}: {title}" if publisher else f"  {pub} — {title}"
            summary = (a.get("summary") or "").strip()
            if summary:
                # Per-summary char cap (default 640, from the markets catalog;
                # strategy may override via [[data]].params).
                line += f"\n    summary: {summary[:cap]}"
            lines.append(line)
        return (
            f"Company news ({self.symbol}, last {since}d, "
            f"PIT {window_start} ≤ published_at < {ceil_iso}, "
            f"newest first, n={len(pit)} — all in window):\n"
            + "\n".join(lines)
        )

    def _web_context_section(self) -> str:
        """Two-tier web search: structural (30d) + updates (last week).

        Brave's LLM-context endpoint returns ``sources.grounding.generic[]``,
        each item ``{url, title, snippets: [...]}``. Snippets are shown in full
        (no per-snippet char cap) — the evidence budget governs total size.
        Results are deduped by URL across all queries; overlaps between tiers
        and with the ``news`` kind are intentional.
        """
        if "web_search" not in self.access.kinds:
            return ""
        ceil_iso = self.decision_date.isoformat()
        # Structural tier honours the strategy binding (default 30d); the update
        # tier is a fixed short window. Both are clamped by the binding's cap.
        struct_since = self._lookback("web_search", self.config.web_structural_lookback_days)
        update_since = self.config.web_update_lookback_days
        max_per_query = self.config.web_snippets_per_query

        plan = [
            ("structural", struct_since, WEB_STRUCTURAL_QUERIES),
            ("updates", update_since, WEB_UPDATE_QUERIES),
        ]

        blocks: list[str] = []
        seen_urls: set[str] = set()
        for tier, since, queries in plan:
            for label, template in queries:
                query = self._format_web_query(template)
                reading = self.access.read(
                    "web_search",
                    query=query,
                    lookback_days=since,
                    max_results=max_per_query,
                )
                # Brave LLM-context shape: sources.grounding.generic[]
                generic: list[dict] = []
                if reading.status == "ok" and isinstance(reading.data, dict):
                    srcs = reading.data.get("sources") or {}
                    if isinstance(srcs, dict):
                        generic = (srcs.get("grounding") or {}).get("generic") or []
                    elif isinstance(srcs, list):
                        generic = srcs  # backward-compat with a flat-list provider
                kept: list[dict] = []
                for s in generic:
                    if not isinstance(s, dict):
                        continue
                    url = (s.get("url") or "").strip()
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    kept.append(s)
                    if len(kept) >= max_per_query:
                        break
                window_start = (self.decision_date - timedelta(days=since)).isoformat()
                if not kept:
                    detail = "" if reading.status == "ok" else f" [{reading.status}]"
                    blocks.append(
                        f"Web context [{tier}/{label}] (search \"{query}\", "
                        f"last {since}d, PIT {window_start} ≤ < {ceil_iso}): none.{detail}"
                    )
                    continue
                from urllib.parse import urlparse
                cap = self._render_cap(
                    "web_search", "snippet_max_chars", 640
                )
                slines = []
                for s in kept:
                    url = (s.get("url") or "").strip()
                    host = urlparse(url).hostname or "" if url else ""
                    title = (s.get("title") or "").strip()
                    snips = s.get("snippets") or []
                    # Per-snippet char cap (default 640, from the markets
                    # catalog; strategy may override via [[data]].params). The
                    # full text is still fetched and cached; only the rendered
                    # evidence is bounded, so the agent's context isn't blown.
                    body = "\n    ".join(
                        str(sn).strip()[:cap] for sn in snips if sn
                    )
                    slines.append(
                        f"  {host} — {title}\n    {body}" if body else f"  {host} — {title}"
                    )
                blocks.append(
                    f"Web context [{tier}/{label}] (search \"{query}\", "
                    f"last {since}d, PIT {window_start} ≤ < {ceil_iso}, "
                    f"n={len(kept)} — all in window):\n"
                    + "\n".join(slines)
                )
        return "\n\n".join(blocks)

    def _filing_narrative(self) -> str:
        if "filing_text" not in self.access.kinds:
            return ""  # filing left out of this package; section omitted entirely
        label = "Latest 10-K business section"
        lookback = self._lookback("filing_text", 400)
        reading = self.access.read(
            "filing_text",
            symbol=self.symbol,
            lookback_days=lookback,
            forms=["10-K"],
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
            text = (r.get("text") or "")  # full section; budget truncates if needed
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
                f"  filing_date={fd} section={sec} (text_len={len(text)}){age_note}\n{text}"
            )
        return f"{label}:\n" + "\n\n".join(blocks)





# ── Adapter ──────────────────────────────────────────────────────────────────

_BOOL = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
_INT_FIELDS = {
    "evidence_budget_chars",
    "fundamentals_max_quarters",
    "filing_narrative_max_chars",
    "news_max_articles",
    "news_summary_max_chars",
    "web_snippets_per_query",
    "web_structural_lookback_days",
    "web_update_lookback_days",
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


