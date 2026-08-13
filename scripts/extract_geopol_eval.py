"""Extract a finished fintel agent-on-agent eval into the blog's EvalData JSON.

Reads ``runs/<job>/eval/<run>/eval.json`` (platform shape: ``{available, per_cell,
summary}`` with ``per_cell`` keyed ``"date|party"``) and emits the blog shape
``{dates: {date: {party: rating}}, summary}``, dropping the ``_``-prefixed
internal fields (``_eval_outcome``, ``_tokens_in``, etc.) and keeping only the
rating fields the blog's ``EvalRating`` type uses.

Usage:
    python scripts/extract_geopol_eval.py <job_id> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Fields kept in each rating (must match EvalRating in data.ts).
RATING_FIELDS = [
    "symbol",
    "loyalty_score",
    "loyalty_rationale",
    "bias_score",
    "bias_rationale",
    "aggression_score",
    "aggression_rationale",
    "bias_flags",
    "recommendation_rating",
    "recommendation_rationale",
    "score",
    "lookahead_bias",
    "lookahead_bias_rationale",
]

SUMMARY_FIELDS = [
    "n_rated",
    "n_failed",
    "rating_run",
    "rating_agent",
    "elapsed_ms",
    "n_llm_calls",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "cost_basis",
]


def extract(job_id: str) -> dict:
    root = Path("runs") / job_id / "eval"
    # pick the first run dir (r1)
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("r"))
    if not run_dirs:
        raise SystemExit(f"no eval run dirs under {root}")
    run = run_dirs[0]
    src = run / "eval.json"
    payload = json.loads(src.read_text())
    per_cell = payload.get("per_cell", {})

    dates: dict[str, dict[str, dict]] = {}
    for key, rating in per_cell.items():
        date, party = key.split("|", 1)
        clean = {f: rating.get(f) for f in RATING_FIELDS if f in rating}
        # Ensure lookahead fields exist (older evals predate them).
        clean.setdefault("lookahead_bias", False)
        clean.setdefault("lookahead_bias_rationale", "")
        dates.setdefault(date, {})[party] = clean

    summary = {f: payload.get("summary", {}).get(f) for f in SUMMARY_FIELDS
               if f in payload.get("summary", {})}
    return {"dates": dates, "summary": summary}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_geopol_eval.py <job_id> <out.json>", file=sys.stderr)
        return 2
    job_id, out = sys.argv[1], sys.argv[2]
    data = extract(job_id)
    Path(out).write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"written: {out}  (dates={len(data['dates'])}, n_rated={data['summary'].get('n_rated')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
