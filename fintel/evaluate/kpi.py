"""The KPI protocol + builtins — the strategy-defined "what good means".

The strategy owns the KPI (`ScoringSpec.kpi` -> a callable
`kpi_fn(signal_by_date, price_lookup, horizons, params) -> dict`). This module
resolves it and runs it over the ensemble signal (and per-run signals for
stochasticity). It never inspects what's inside `kpi_fn`.

Builtin `single_name_ir` is the current package's KPI: **raw IC** (per-date
Spearman and Pearson IC of signal vs forward return, plus mean / raw_icir /
t_stat / icir_ann / ic_values) over the decision-date grid. `t_stat` is
`raw_icir × √n` (one-sample t of mean IC vs 0). `icir_ann` is
`raw_icir × √ppy` when `params["ppy"]` is set (cadence from the report
layer); mean IC is a rank/linear correlation and is never annualized. The
"net returns" half of "raw IC and net" comes from the holdings layer
(`evaluate/holdings.py`, platform mechanics), combined in the report —
keeping the KPI (the metric) and holdings (the mechanics) separate per the
abstraction in docs/architecture.md §1.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as Date

from fintel.market.realized import PriceLookup
from fintel.models.common import Symbol
from fintel.models.evaluate import Signals
from fintel.utils.import_path import resolve

# A KPI function: signal series + prices + horizons + params -> a result dict.
KpiFn = Callable[
    [dict[Date, dict[Symbol, float]], PriceLookup, list[int], dict],
    dict,
]

KPI_BUILTINS: dict[str, str] = {
    "single_name_ir": "fintel.evaluate.kpi:single_name_ir",
}


def resolve_kpi(name: str) -> KpiFn:
    """Resolve a builtin KPI name or a `module:Callable`."""
    spec = KPI_BUILTINS.get(name, name)
    if ":" not in spec:
        raise ValueError(
            f"unknown kpi {name!r}; expected one of {sorted(KPI_BUILTINS)} or 'module:Callable'"
        )
    return resolve(spec)  # type: ignore[return-value]


# --- IC primitives (pure python, no scipy) ------------------------------------


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, 1-indexed (Spearman convention)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-indexed average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def _spearman_ic(signal: dict[Symbol, float], fwd: dict[Symbol, float]) -> float | None:
    """Cross-sectional Spearman rank IC over the symbols present in both."""
    common = sorted(set(signal) & set(fwd))
    if len(common) < 2:
        return None
    sig_vals = [signal[s] for s in common]
    fwd_vals = [fwd[s] for s in common]
    rx = _ranks(sig_vals)
    ry = _ranks(fwd_vals)
    return _pearson(rx, ry)


def _pearson_ic(signal: dict[Symbol, float], fwd: dict[Symbol, float]) -> float | None:
    """Cross-sectional Pearson IC over the symbols present in both."""
    common = sorted(set(signal) & set(fwd))
    if len(common) < 2:
        return None
    return _pearson([signal[s] for s in common], [fwd[s] for s in common])


def _mean_icir(vals: list[float]) -> tuple[float | None, float | None]:
    """Return (mean, raw ICIR = mean/sample-std). ICIR is None when n<2 or std=0."""
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, None
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    std = var**0.5
    return mu, (mu / std if std > 0 else None)


def _t_stat(icir: float | None, n: int) -> float | None:
    """One-sample t of mean IC vs 0: ``ICIR_raw × √n``. None when ICIR is undefined."""
    if icir is None or n < 2:
        return None
    return icir * (n**0.5)


def _forward_returns(
    dates: list[Date], universe: list[Symbol], prices: PriceLookup, horizon: int
) -> dict[Date, dict[Symbol, float]]:
    """P[t+h]/P[t]-1 on the decision-date grid, per date."""
    out: dict[Date, dict[Symbol, float]] = {}
    for i, d in enumerate(dates):
        if i + horizon >= len(dates):
            break  # no future date on the grid for the last h
        end = dates[i + horizon]
        fwd: dict[Symbol, float] = {}
        for s in universe:
            r = prices.forward_return(s, d, end)
            if r is not None:
                fwd[s] = r
        if fwd:
            out[d] = fwd
    return out


# --- the builtin KPI ----------------------------------------------------------


def single_name_ir(
    signal_by_date: dict[Date, dict[Symbol, float]],
    prices: PriceLookup,
    horizons: list[int],
    params: dict,
) -> dict:
    """Raw IC for a single-name strategy.

    Per horizon: per-date Spearman and Pearson IC of signal vs forward return,
    then mean / raw_icir (mean/std, not annualized) / t_stat (`raw_icir × √n`) /
    icir_ann (`raw_icir × √ppy` when `params["ppy"]` is set). Mean IC is never
    annualized. `ic_values` stays Spearman for backward compatibility. The
    net-returns half of the package's "raw IC and net" KPI is produced by the
    holdings layer.
    """
    dates = sorted(signal_by_date)
    universe = sorted({s for sig in signal_by_date.values() for s in sig})
    ppy = params.get("ppy")
    try:
        sqrt_ppy = float(ppy) ** 0.5 if ppy else None
    except (TypeError, ValueError):
        sqrt_ppy = None

    def _ann(raw: float | None) -> float | None:
        if raw is None or sqrt_ppy is None:
            return None
        return round(raw * sqrt_ppy, 6)

    empty = {
        "mean_ic": None,
        "raw_icir": None,
        "t_stat": None,
        "icir_ann": None,
        "pearson_mean": None,
        "pearson_icir": None,
        "pearson_t": None,
        "pearson_icir_ann": None,
        "ic_values": [],
        "n_periods": 0,
    }
    per_horizon: dict[int, dict] = {}
    for h in horizons:
        fwd_by_date = _forward_returns(dates, universe, prices, h)
        sp_vals: list[float] = []
        pe_vals: list[float] = []
        for d in dates:
            if d not in fwd_by_date:
                continue
            sig, fwd = signal_by_date[d], fwd_by_date[d]
            sp = _spearman_ic(sig, fwd)
            pe = _pearson_ic(sig, fwd)
            if sp is not None:
                sp_vals.append(sp)
            if pe is not None:
                pe_vals.append(pe)
        if not sp_vals and not pe_vals:
            per_horizon[h] = dict(empty)
            continue
        sp_mu, sp_ir = _mean_icir(sp_vals)
        pe_mu, pe_ir = _mean_icir(pe_vals)
        n_sp, n_pe = len(sp_vals), len(pe_vals)
        sp_t = _t_stat(sp_ir, n_sp)
        pe_t = _t_stat(pe_ir, n_pe)
        per_horizon[h] = {
            "mean_ic": round(sp_mu, 6) if sp_mu is not None else None,
            "raw_icir": round(sp_ir, 6) if sp_ir is not None else None,
            "t_stat": round(sp_t, 6) if sp_t is not None else None,
            "icir_ann": _ann(sp_ir),
            "pearson_mean": round(pe_mu, 6) if pe_mu is not None else None,
            "pearson_icir": round(pe_ir, 6) if pe_ir is not None else None,
            "pearson_t": round(pe_t, 6) if pe_t is not None else None,
            "pearson_icir_ann": _ann(pe_ir),
            "ic_values": [round(v, 6) for v in sp_vals],
            "n_periods": n_sp,
        }
    return {
        "kpi": "single_name_ir",
        "metric_key": params.get("metric_key", "icir"),
        "horizons": horizons,
        "ppy": ppy,
        "per_horizon": per_horizon,
        "n_dates": len(dates),
    }


# --- orchestration ------------------------------------------------------------


def compute(
    signals: Signals, prices: PriceLookup, *, kpi: str, horizons: list[int], params: dict
) -> dict:
    """Run the strategy's KPI over the ensemble signal, plus per-run for
    stochasticity. Returns `{ensemble: ..., per_run: [...]}`."""
    fn = resolve_kpi(kpi)
    ensemble = fn(signals.ensemble, prices, horizons, params)
    per_run = [fn(sig, prices, horizons, params) for sig in signals.per_run]
    return {"ensemble": ensemble, "per_run": per_run}
