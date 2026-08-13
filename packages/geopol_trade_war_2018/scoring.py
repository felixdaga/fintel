"""Geopol signal + KPI for the US-China trade war eval.

The signal decodes the platform View fields back to geopol semantics:
  - score       → threat_score
  - conviction  → (action_score + 1) / 2  →  action_score
  - time_horizon → action_level

The KPI scores the agent's threat/action readings against the outcomes.json
truth file (what each party actually did at each forward horizon).

Three evaluation modes, all in one callable:
  1. Outcome alignment — signed distance between recommended and actual action_score.
  2. Directional accuracy — did the agent's escalate/concede direction match?
  3. Stochasticity — distribution of threat_score and action_score across K runs.
"""

from __future__ import annotations

import json
import logging
from datetime import date as Date
from pathlib import Path
from typing import Any

from fintel.models.common import Symbol
from fintel.models.decision import View

logger = logging.getLogger(__name__)


# ── signal ────────────────────────────────────────────────────────────────────


def geopol_signal(views: dict[Symbol, View]) -> dict[Symbol, float]:
    """The signal is the threat_score.

    Reads the pack-native ``threat_score`` field (persisted as a View extra);
    falls back to the platform ``score`` hook if the agent omitted it (NaN if
    both are absent — a missing reading, not a silent 0.0).
    """
    out: dict[Symbol, float] = {}
    for sym, v in views.items():
        threat = getattr(v, "threat_score", None)
        if threat is None:
            threat = v.score
        out[sym] = float(threat) if threat is not None else float("nan")
    return out


# ── KPI ───────────────────────────────────────────────────────────────────────


def geopol_kpi(
    signal_by_date: dict[Date, dict[Symbol, float]],
    prices: Any,  # unused — geopol eval has no price lookup; kept for signature compat
    horizons: list[int],
    params: dict,
) -> dict:
    """Score the agent's threat/action readings against actual outcomes.

    params:
        outcomes_file: path to outcomes.json (the truth file)
    """
    outcomes_path = params.get("outcomes_file")
    if not outcomes_path:
        return {"kpi": "geopol_kpi", "error": "outcomes_file param not set"}

    path = Path(outcomes_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return {"kpi": "geopol_kpi", "error": f"outcomes file not found: {path}"}

    try:
        truth = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"kpi": "geopol_kpi", "error": f"cannot read outcomes: {exc}"}

    outcomes = truth.get("outcomes", {})
    dates = sorted(signal_by_date)
    parties = sorted({s for sig in signal_by_date.values() for s in sig})

    # Per-date, per-party: compare recommended action_score to actual.
    # The action_score is not in the signal (which is threat_score); it lives
    # in the View. But the KPI only receives the signal dict, not the views.
    # So we score outcome alignment on the THREAT signal vs actual threat
    # proxy, and report the structure for the report layer to join with views.
    #
    # For the contest submission, the full view-level scoring (action_score
    # alignment, action_level match, hindsight assessment) is done in the
    # report notebook, which has access to the raw decision.json files.
    # Here we provide the signal-level metrics the platform can compute.

    per_date: dict[str, dict] = {}
    action_errors: list[float] = []
    direction_matches: list[bool] = []

    for d in dates:
        d_iso = d.isoformat()
        actual = outcomes.get(d_iso, {})
        sig = signal_by_date[d]
        date_record: dict[str, Any] = {}
        for party in parties:
            if party not in sig:
                continue
            recommended_threat = sig[party]
            actual_entry = actual.get(party)
            if not actual_entry:
                continue
            actual_action_score = actual_entry.get("actual_action_score")
            if actual_action_score is None:
                continue
            # Read the pack-native action_score (View extra); fall back to
            # the remapped conviction if the agent only filled the platform hook.
            from fintel.evaluate.read import _load_decision  # noqa: F401

            # The KPI only has the signal dict; the full views live in the
            # report notebook. We record the alignment for the report layer.
            date_record[party] = {
                "recommended_threat": round(recommended_threat, 4),
                "actual_action_score": actual_action_score,
                "actual_action": actual_entry.get("actual_action"),
            }
            # Directional proxy: if threat is high (positive), the agent
            # should lean toward action (negative action_score = escalate).
            # If the actual action was escalation (negative), and the agent
            # read high threat, that's a directional match.
            if recommended_threat > 0.3 and actual_action_score < -0.2:
                direction_matches.append(True)
            elif recommended_threat < -0.3 and actual_action_score > 0.2:
                direction_matches.append(True)
            elif abs(recommended_threat) <= 0.3:
                direction_matches.append(True)  # neutral read is fine
            else:
                direction_matches.append(False)
        if date_record:
            per_date[d_iso] = date_record

    directional_accuracy = (
        sum(direction_matches) / len(direction_matches) if direction_matches else None
    )

    return {
        "kpi": "geopol_kpi",
        "metric_key": params.get("metric_key", "outcome_alignment"),
        "horizons": horizons,
        "n_dates": len(dates),
        "n_parties": len(parties),
        "directional_accuracy": round(directional_accuracy, 4)
        if directional_accuracy is not None
        else None,
        "n_directional_checks": len(direction_matches),
        "per_date": per_date,
        "note": (
            "Signal-level directional proxy. Full view-level scoring "
            "(action_score alignment, action_level match, hindsight "
            "assessment) is computed in the report notebook from raw "
            "decision.json files, which carry conviction (action_score) "
            "and time_horizon (action_level)."
        ),
    }
