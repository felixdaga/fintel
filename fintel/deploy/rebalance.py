"""Rebalance: compute target holdings, trades, and rebalancing guidance.

Reads a finished run + deploy config, computes the target book from the
latest decision's scores using the configured holdings rule, compares
against the current book (previous decision's holdings drifted by price
moves), and produces trade instructions for a given capital amount.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from pathlib import Path

from fintel.deploy.config import DeployConfig, job_dir
from fintel.deploy.holdings import get_rule
from fintel.evaluate.prices import ensure_job_prices, mark_as_of, price_lookup_for
from fintel.evaluate.read import load_job
from fintel.evaluate.signals import build_signals
from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.strategy import ScoringSpec


@dataclass
class Holding:
    symbol: Symbol
    target_weight: float
    current_weight: float
    target_shares: float
    current_shares: float
    trade_shares: float
    price: float
    target_notional: float
    action: str  # "buy", "sell", "hold"


@dataclass
class RebalanceReport:
    decision_date: str
    as_of: str
    capital: float
    n_holdings: int
    holdings: list[Holding] = field(default_factory=list)
    total_turnover: float = 0.0
    n_buys: int = 0
    n_sells: int = 0

    def to_dict(self) -> dict:
        return {
            "decision_date": self.decision_date,
            "as_of": self.as_of,
            "capital": self.capital,
            "n_holdings": self.n_holdings,
            "holdings": [asdict(h) for h in self.holdings],
            "total_turnover": round(self.total_turnover, 4),
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
        }


def compute_rebalance(
    config: DeployConfig,
    *,
    capital: float | None = None,
) -> RebalanceReport:
    """Compute the rebalancing report from the latest decision in the run."""
    jdir = job_dir(config)
    capital = capital if capital is not None else config.capital

    cfg = json.loads((jdir / "r1" / "config.json").read_text())
    scoring = ScoringSpec.model_validate(cfg["scoring"])
    signals = build_signals(load_job(jdir), signal=scoring.signal, transform=scoring.transform)
    universe = sorted(signals.universe)
    ensure_job_prices(jdir, universe)
    # Use close prices for rebalance sizing — this is what the agent saw.
    # The backtest NAV (build_strategy_data.py) still uses open-to-open
    # for forward returns; only the rebalance report prices change here.
    prices = price_lookup_for(jdir)
    prices_close = PriceLookup(store=prices.store, price_field="close")
    as_of = mark_as_of(prices_close, signals)

    dates = sorted(signals.ensemble)
    latest = dates[-1]
    prev = dates[-2] if len(dates) >= 2 else None

    # Target weights from latest decision
    rule = get_rule(config.holdings.rule)
    rule_params = {
        **config.holdings.params,
        "threshold": config.holdings.threshold,
        "active_budget": config.holdings.active_budget,
    }
    target_w = rule(signals.ensemble[latest], rule_params)

    # Current weights: previous decision's book, drifted by price moves
    if prev is not None:
        prev_w = rule(signals.ensemble[prev], rule_params)
        current_w = _drift_weights(prev_w, prices_close, prev, as_of or latest)
    else:
        current_w = {}

    # Compute holdings + trades
    holdings: list[Holding] = []
    all_symbols = sorted(set(target_w) | set(current_w))
    total_turnover = 0.0
    n_buys = n_sells = 0

    for sym in all_symbols:
        tw = target_w.get(sym, 0.0)
        cw = current_w.get(sym, 0.0)
        px = prices_close.price_at(sym, as_of or latest)
        if px is None or px <= 0:
            continue
        target_shares = (capital * tw) / px
        current_shares = (capital * cw) / px
        trade = target_shares - current_shares
        action = "buy" if trade > 0.5 else ("sell" if trade < -0.5 else "hold")
        if action == "buy":
            n_buys += 1
        elif action == "sell":
            n_sells += 1
        total_turnover += abs(tw - cw)
        holdings.append(
            Holding(
                symbol=sym,
                target_weight=round(tw, 4),
                current_weight=round(cw, 4),
                target_shares=round(target_shares, 2),
                current_shares=round(current_shares, 2),
                trade_shares=round(trade, 2),
                price=round(px, 2),
                target_notional=round(capital * tw, 2),
                action=action,
            )
        )

    # Sort: trades first (by abs magnitude), then non-trades by weight
    holdings.sort(key=lambda h: (-(abs(h.trade_shares)), -h.target_weight))

    return RebalanceReport(
        decision_date=latest.isoformat(),
        as_of=(as_of or latest).isoformat(),
        capital=capital,
        n_holdings=len(target_w),
        holdings=holdings,
        total_turnover=round(total_turnover, 4),
        n_buys=n_buys,
        n_sells=n_sells,
    )


def write_rebalance(report: RebalanceReport, config: DeployConfig) -> dict[str, Path]:
    """Write rebalance.json + rebalance.md under <job_dir>/deploy/."""
    deploy_dir = job_dir(config) / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    json_path = deploy_dir / "rebalance.json"
    md_path = deploy_dir / "rebalance.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md_path.write_text(_render_markdown(report))
    return {"json": json_path, "markdown": md_path}


def _drift_weights(
    prev_w: dict[Symbol, float], prices: PriceLookup, prev_date: Date, as_of: Date
) -> dict[Symbol, float]:
    """Drift previous weights by price moves from prev_date to as_of."""
    if not prev_w:
        return {}
    drifted = {}
    for sym, w in prev_w.items():
        p0 = prices.price_at(sym, prev_date)
        p1 = prices.price_at(sym, as_of)
        if p0 and p0 > 0 and p1 and p1 > 0:
            drifted[sym] = w * (p1 / p0)
    total = sum(drifted.values()) or 1.0
    return {s: w / total for s, w in drifted.items()}


def _render_markdown(r: RebalanceReport) -> str:
    lines = [
        f"# Rebalance — {r.decision_date}",
        "",
        f"Capital: ${r.capital:,.0f}  |  Holdings: {r.n_holdings}  |  "
        f"Buys: {r.n_buys}  Sells: {r.n_sells}  |  Turnover: {r.total_turnover:.2f}",
        "",
        "| Symbol | Action | Target Wt | Curr Wt | Trade Shares | Price | Notional |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in r.holdings:
        lines.append(
            f"| {h.symbol} | {h.action} | {h.target_weight:.1%} | {h.current_weight:.1%} | "
            f"{'+' if h.trade_shares > 0 else ''}{h.trade_shares:.1f} | "
            f"${h.price:.2f} | ${h.target_notional:,.0f} |"
        )
    lines.append("")
    return "\n".join(lines)
