"""``fintel report <job_id>`` — evaluate a finished job.

Reads the strategy's `ScoringSpec` from the job's run config (the strategy
declares the KPI/signal/transform/horizons; the platform runs the mechanics),
runs the full evaluation pipeline, writes `report.json` + `report.md`, and
prints the markdown to stdout.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from fintel.models.paths import JobPaths
from fintel.models.strategy import EvalSpec, ScoringSpec


def _load_scoring(job_dir: Path) -> ScoringSpec:
    """The scoring spec is frozen into each run's config (`rK/config.json`)."""
    paths = JobPaths(root=job_dir)
    for run_dir in paths.run_dirs():
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        scoring = cfg.get("scoring")
        if scoring:
            # `signal` defaults to "single_name" for runs frozen before the
            # signal hook existed — ScoringSpec's own default handles it.
            return ScoringSpec.model_validate(scoring)
    raise SystemExit(
        f"no scoring spec found under {job_dir}/r*/config.json — "
        "the job may be incomplete or pre-dates the evaluation layer"
    )


def _load_eval_spec(job_dir: Path) -> tuple[EvalSpec | None, Path | None]:
    """Load the [eval] section + strategy root from the job's run config.

    The eval spec is frozen into each run's config alongside the scoring spec.
    The strategy root is needed to load rating_prompt.md / rating_schema.json.
    """
    paths = JobPaths(root=job_dir)
    for run_dir in paths.run_dirs():
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        eval_cfg = cfg.get("eval")
        if eval_cfg:
            strategy_path = cfg.get("strategy", "")
            if isinstance(strategy_path, dict):
                strategy_path = strategy_path.get("path", "")
            return EvalSpec.model_validate(eval_cfg), Path(strategy_path) if strategy_path else None
    return None, None


def _load_strategy_root(job_dir: Path) -> Path | None:
    """The pack directory from the job's run config — always present, even
    without an [eval] section, because the scoring signal/KPI import the
    pack's ``scoring.py`` by module path and need its parent on ``sys.path``.
    """
    paths = JobPaths(root=job_dir)
    for run_dir in paths.run_dirs():
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        strategy_path = cfg.get("strategy", "")
        if isinstance(strategy_path, dict):
            strategy_path = strategy_path.get("path", "")
        if strategy_path:
            return Path(strategy_path)
    return None


def run_report(args: Namespace) -> int:
    from fintel.evaluate.report import report, write_report

    root = Path(args.output_root)
    job_dir = root / args.job_id
    if not job_dir.is_dir():
        raise SystemExit(f"job not found: {job_dir}")
    scoring = _load_scoring(job_dir)
    eval_spec, strategy_root = _load_eval_spec(job_dir)
    # strategy_root is needed for the pack scoring import even without an eval,
    # so fall back to the always-present strategy path.
    if strategy_root is None:
        strategy_root = _load_strategy_root(job_dir)
    payload = report(
        job_dir,
        scoring=scoring,
        cache_root=args.cache_root,
        eval_spec=eval_spec,
        strategy_root=strategy_root,
        shared_concurrency=getattr(args, "shared_concurrency", None),
    )
    paths = write_report(payload, job_dir)
    print(render(payload))
    print(f"\n(written: {paths['json']} , {paths['markdown']})")
    return 0


def render(payload) -> str:
    from fintel.evaluate.report import render_markdown

    return render_markdown(payload)
