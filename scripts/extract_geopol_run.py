"""Extract a finished geopol run into the site-friendly JSON the blog consumes.

Reads the on-disk run artifacts (decision.json per trial, access.jsonl per cell,
outcomes.json from the pack) and emits the ``{runs, outcomes}`` shape the blog
components expect — one file per config so the site can compare ablations.

Usage:
    python scripts/extract_geopol_run.py <job_id> [--out NAME] [--label LABEL]

The output lands in site/src/content/blogs/geopol-trade-war-2018/<name>.json.
Local job folders are untracked; pass --out to write the published name.

Published configs (site JSON names, not local job ids):
    geopol-abl-mimo-llm.json
    geopol-abl-mimo-oc.json
    geopol-abl-deepseek-oc.json
    geopol-abl-grok-oc.json

    python scripts/extract_geopol_run.py <job_id> --out geopol-abl-mimo-oc.json

The outcomes (truth file) is shared across configs and copied only if missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PACK = ROOT / "packages" / "geopol_trade_war_2018"
OUT_DIR = ROOT / "site" / "src" / "content" / "blogs" / "geopol-trade-war-2018"

PARTIES = ("USA", "CHN")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _access_reads(session_dir: Path) -> tuple[int, int, list[str], dict[str, int]]:
    """Return (n_reads, n_searches, search_queries, read_kinds) from access.jsonl."""
    access = session_dir / "access.jsonl"
    if not access.is_file():
        return 0, 0, [], {}
    kinds: dict[str, int] = {}
    queries: list[str] = []
    for line in access.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") != "read" or not e.get("kind"):
            continue
        k = e["kind"]
        kinds[k] = kinds.get(k, 0) + 1
        if k == "web_search":
            q = (e.get("query") or {}).get("query")
            if q:
                queries.append(q)
    n_reads = sum(kinds.values())
    return n_reads, kinds.get("web_search", 0), queries, kinds


def _extract_cell(decision: dict, party: str) -> dict | None:
    v = decision.get(party)
    if not v:
        return None
    return {
        "threat_score": v.get("threat_score"),
        "action_score": v.get("action_score"),
        "action_level": v.get("action_level"),
        "score": v.get("score"),
        "rationale": v.get("rationale", ""),
        "key_factors": list(v.get("key_factors", [])),
        "sources_cited": list(v.get("sources_cited", [])),
        # n_reads / n_searches / search_queries / read_kinds / elapsed filled below
        "n_reads": 0,
        "n_searches": 0,
        "search_queries": [],
        "read_kinds": {},
        "elapsed_ms": 0,
    }


def extract(job_id: str) -> dict:
    job = RUNS / job_id
    if not job.is_dir():
        raise SystemExit(f"run not found: {job}")
    runs: dict[str, dict] = {}
    for k in (1, 2, 3):
        rk = f"r{k}"
        rd = job / rk
        if not rd.is_dir():
            continue
        run_id = f"{job_id}-{rk}"
        dates: dict[str, dict] = {}
        for trial in sorted((rd / "trials").iterdir()):
            if not trial.is_dir():
                continue
            date = trial.name
            decision = _load_json(trial / "decision.json")
            cells: dict[str, dict] = {}
            for party in PARTIES:
                cell = _extract_cell(decision, party)
                if cell is None:
                    continue
                # reads from the session access log
                sess = rd / "sessions" / run_id / date / party
                n_reads, n_searches, queries, kinds = _access_reads(sess)
                cell["n_reads"] = n_reads
                cell["n_searches"] = n_searches
                cell["search_queries"] = queries
                cell["read_kinds"] = kinds
                # elapsed from the cell record
                rec = _load_json(trial / "cells" / f"{party}.json")
                cell["elapsed_ms"] = int(rec.get("elapsed_ms", 0) or 0)
                cells[party] = cell
            dates[date] = cells
        runs[rk] = {"dates": dates}

    # outcomes are pack-owned (shared across configs)
    outcomes_raw = _load_json(PACK / "outcomes.json").get("outcomes", {})
    outcomes: dict[str, dict] = {}
    for date, parties in outcomes_raw.items():
        outcomes[date] = {}
        for party in PARTIES:
            p = parties.get(party, {})
            outcomes[date][party] = {
                "actual_action": p.get("actual_action", ""),
                "actual_action_score": p.get("actual_action_score", 0),
                "rationale": p.get("rationale", ""),
            }
    return {"runs": runs, "outcomes": outcomes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--out", default=None, help="output filename (default: <job_id>.json)")
    ap.add_argument("--label", default="", help="human label for this config")
    args = ap.parse_args()

    data = extract(args.job_id)
    name = args.out or f"{args.job_id}.json"
    out = OUT_DIR / name
    out.write_text(json.dumps(data, indent=2) + "\n")
    n_cells = sum(len(d) for r in data["runs"].values() for d in r["dates"].values())
    print(
        f"wrote {out.relative_to(ROOT)} — runs={len(data['runs'])} cells={n_cells} dates={len(data.get('outcomes', {}))}"
    )


if __name__ == "__main__":
    main()
