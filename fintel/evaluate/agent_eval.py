"""Agent-on-agent evaluation: a rating agent reviews finished run outputs.

This is the ``[eval]`` half of the report pipeline. When a strategy pack
declares an ``[eval]`` section, ``fintel report`` deploys a rating agent
with current knowledge (no PIT cutoff) to rate specific output fields
from each cell of a finished run.

The default rater is the ``llm`` agent (``channel='pack'``) — it gets the
full decisions fed in and judges with its own knowledge. The pack may add
tools (``rating_tools``) to feed extra context; for this run we add the
full event timeline. The rater's cell is stamped with today's date so the
PIT cutoff doesn't clip historical data, and lookback caps are removed so
the full timeline is visible.

The platform provides the plumbing (agent adapters, MCP, cell execution,
invoke); the pack specifies *what* to rate (``rating_prompt.md``), *what
shape* the rating takes (``rating_schema.json``), and *which agent* plays
the rater. This mirrors the simulation contract: the pack owns the
strategy, the platform owns the mechanics.

Key difference from simulation: the rating agent has **no PIT cutoff** —
it may use current knowledge to assess whether the original agent's
recommendation was good or biased. It is not replaying history; it is
judging it with hindsight.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date
from pathlib import Path
from typing import Any

from fintel.agents import factory as agent_factory
from fintel.agents.run import invoke
from fintel.environment.cell import Cell
from fintel.environment.factory import RuntimeConfig, build_environment
from fintel.environment.progress import NullProgress, Progress
from fintel.market.data.base import DataSource
from fintel.market.factory import build_data_sources
from fintel.market.settings import MarketConfig
from fintel.models.agent import AgentSpec, ModelSpec
from fintel.models.common import Symbol
from fintel.models.strategy import EvalSpec
from fintel.models.paths import JobPaths

logger = logging.getLogger(__name__)

# The rater's cell is stamped with today so the PIT cutoff (strictly-before
# midnight of decision_date) doesn't clip any historical data — the rater
# sees the full timeline including events after the original decision date.
_RATER_DECISION_DATE = Date.today

# Lookback cap override: 100 years. The pack's strategy.toml declares
# lookback_days (e.g. 365 for event_timeline), which is correct for the
# simulation agent but would clip the 2018-2019 timeline when the rater's
# decision date is today. We override to a very large value so the full
# declared lookback is visible.
_NO_LOOKBACK_CAP = 36500


def evaluate(
    job_dir: Path,
    *,
    eval_spec: EvalSpec,
    strategy_root: Path,
    cache_root: str | Path | None = None,
    market_config: MarketConfig | None = None,
    progress: Progress | None = None,
    shared_concurrency: int | None = None,
) -> dict[str, Any]:
    """Run agent-on-agent evaluation over a finished job.

    For each cell in the selected run, deploy the rating agent with the
    pack's rating prompt + schema, give it the original cell's output,
    and collect its ratings. Returns ``{per_cell: {...}, summary: {...}}``.

    The rating agent reuses the same MCP sources as the simulation but
    without PIT cutoff — it has current knowledge. Only the tool kinds
    listed in ``eval_spec.rating_tools`` are bound; the rest are dropped.

    ``shared_concurrency`` parallelizes cell ratings via a thread pool,
    same semantics as the simulation's ``--shared-concurrency``. ``None``
    or ``1`` runs sequentially.
    """
    progress = progress or NullProgress()
    started = time.perf_counter()

    # Load the rating prompt + schema from the pack.
    prompt_path = strategy_root / eval_spec.rating_prompt_file
    schema_path = strategy_root / eval_spec.rating_schema_file
    if not prompt_path.is_file():
        logger.warning("agent_eval: rating prompt not found at %s — skipping", prompt_path)
        return {"available": False, "summary": {"note": f"rating prompt not found: {prompt_path}"}}
    rating_prompt = prompt_path.read_text()
    rating_schema = ""
    if schema_path.is_file():
        rating_schema = schema_path.read_text()

    # Select which run to rate.
    run_dirs = JobPaths(root=job_dir).run_dirs()
    if not run_dirs:
        return {"available": False, "summary": {"note": "no runs found"}}
    if eval_spec.rating_run == "worst":
        run_dir = run_dirs[-1]
    else:
        run_dir = run_dirs[0]

    # Build the rating agent spec.
    agent_spec = AgentSpec(
        name=eval_spec.rating_agent,
        model=ModelSpec(),
        options=dict(eval_spec.rating_agent_opt),
    )

    # Build data sources for the rater — only the declared rating_tools,
    # from the pack's strategy.toml bindings, with lookback caps removed
    # so the full timeline is visible from today's vantage point.
    if market_config is None:
        market_config = MarketConfig.from_env(
            cache_root=cache_root or (job_dir.parent / "cache")
        )
    sources = _build_rating_sources(strategy_root, eval_spec, market_config)

    # Walk the selected run's cells and collect (date, cell_name, output) triples.
    trials_dir = run_dir / "trials"
    if not trials_dir.is_dir():
        return {"available": False, "summary": {"note": "no trials directory"}}

    tasks: list[tuple[Date, str, dict]] = []
    for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
        try:
            d = Date.fromisoformat(trial_dir.name)
        except ValueError:
            continue
        cells_dir = trial_dir / "cells"
        if not cells_dir.is_dir():
            continue
        for cell_path in sorted(cells_dir.glob("*.json")):
            rec = json.loads(cell_path.read_text())
            name = rec.get("cell", cell_path.stem)
            output = rec.get("views", {}).get(name)
            if output:
                tasks.append((d, name, output))

    if not tasks:
        return {"available": False, "summary": {"note": "no cells with views found"}}

    # Eval output lives inside the job it evaluates, as a separate folder:
    #   runs/<job>/eval/<run_id>/<date>/<cell>.json   — per-cell rating
    #   runs/<job>/eval/<run_id>/eval.json             — summary (all ratings)
    eval_dir = job_dir / "eval" / run_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Rate cells — sequentially or in parallel via a thread pool.
    bound = shared_concurrency or 1
    results: list[dict | None] = [None] * len(tasks)

    def _rate(task_idx: int) -> dict | None:
        d, name, output = tasks[task_idx]
        progress.emit("cell_start", cell=name, decision_date=d.isoformat())
        cell_started = time.perf_counter()
        rating = _rate_one_cell(
            cell_name=name,
            original_decision_date=d,
            original_output=output,
            rating_prompt=rating_prompt,
            rating_schema=rating_schema,
            agent_spec=agent_spec,
            sources=sources,
            market_config=market_config,
            job_dir=job_dir,
            progress=progress,
        )
        elapsed_ms = int((time.perf_counter() - cell_started) * 1000)

        # Persist to disk (thread-safe: each cell writes its own file).
        date_eval_dir = eval_dir / d.isoformat()
        date_eval_dir.mkdir(parents=True, exist_ok=True)

        if rating is not None:
            (date_eval_dir / f"{name}.json").write_text(
                json.dumps(rating, indent=2, default=str)
            )
            progress.emit(
                "cell_done",
                cell=name,
                decision_date=d.isoformat(),
                outcome="ok",
                elapsed_ms=elapsed_ms,
                n_llm_calls=rating.get("_n_llm_calls", 0),
                tokens_in=rating.get("_tokens_in", 0),
                tokens_out=rating.get("_tokens_out", 0),
                cost_usd=rating.get("_cost_usd"),
                cost_basis=rating.get("_cost_basis", "unknown"),
            )
        else:
            progress.emit(
                "cell_done",
                cell=name,
                decision_date=d.isoformat(),
                outcome="failed",
                elapsed_ms=elapsed_ms,
            )
        return rating

    if bound <= 1:
        for i in range(len(tasks)):
            results[i] = _rate(i)
    else:
        with ThreadPoolExecutor(max_workers=bound) as pool:
            future_to_idx = {pool.submit(_rate, i): i for i in range(len(tasks))}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

    # Assemble results + accumulate usage totals.
    per_cell: dict[str, dict] = {}
    n_rated = 0
    n_failed = 0
    total_tokens_in = 0
    total_tokens_out = 0
    total_n_llm_calls = 0
    total_cost_usd: float | None = None
    cost_basis = "unknown"

    for i, (d, name, _) in enumerate(tasks):
        rating = results[i]
        key = f"{d.isoformat()}|{name}"
        if rating is not None:
            per_cell[key] = rating
            n_rated += 1
            total_tokens_in += int(rating.get("_tokens_in", 0) or 0)
            total_tokens_out += int(rating.get("_tokens_out", 0) or 0)
            total_n_llm_calls += int(rating.get("_n_llm_calls", 0) or 0)
            cell_cost = rating.get("_cost_usd")
            if cell_cost is not None:
                total_cost_usd = (total_cost_usd or 0.0) + float(cell_cost)
            cb = rating.get("_cost_basis", "unknown")
            if cb == "reported" or (total_cost_usd is not None and cb in ("reported", "estimated")):
                cost_basis = cb if cost_basis != "reported" else "reported"
        else:
            n_failed += 1

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    summary = {
        "available": n_rated > 0,
        "n_rated": n_rated,
        "n_failed": n_failed,
        "rating_run": run_dir.name,
        "rating_agent": eval_spec.rating_agent,
        "elapsed_ms": elapsed_ms,
        "n_llm_calls": total_n_llm_calls,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "cost_usd": total_cost_usd,
        "cost_basis": cost_basis,
    }
    result = {"available": n_rated > 0, "per_cell": per_cell, "summary": summary}

    # Write the summary file.
    (eval_dir / "eval.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    return result


def _rate_one_cell(
    *,
    cell_name: str,
    original_decision_date: Date,
    original_output: dict,
    rating_prompt: str,
    rating_schema: str,
    agent_spec: AgentSpec,
    sources: dict[str, DataSource],
    market_config: MarketConfig,
    job_dir: Path,
    progress: Progress,
) -> dict | None:
    """Deploy the rating agent for one cell's output.

    Builds a fresh environment stamped with today's date (so PIT cutoff
    doesn't clip), composes the rating prompt with the original cell's
    output, invokes the agent, and returns its rating.
    """
    # The rater's cell is stamped with today — the PIT cutoff is today,
    # so no historical data is clipped. The original decision date is
    # carried in the instruction text, not in the cell identity. The
    # run_id includes the original decision date so concurrent cells
    # don't collide on the same session directory.
    rating_cell = Cell(
        run_id=f"eval-{job_dir.name}-{original_decision_date.isoformat()}",
        decision_date=_RATER_DECISION_DATE(),
        symbols=(Symbol(cell_name),),
        scope="single_name",
    )

    # Runtime: write rating sessions under <job>/eval/sessions/
    eval_session_root = job_dir / "eval" / "sessions"
    eval_session_root.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(
        session_root=eval_session_root,
        trace=True,
        reset_sessions=True,
    )

    env = build_environment(
        cell=rating_cell,
        sources=sources,
        universe=[Symbol(cell_name)],
        kinds=tuple(sources) if sources else (),
        runtime=runtime,
        market_config=market_config,
        nerve=progress,
    )

    # Compose the instruction: rating prompt + original output + schema.
    instruction = _compose_rating_instruction(
        rating_prompt=rating_prompt,
        original_output=original_output,
        rating_schema=rating_schema,
        cell_name=cell_name,
        original_decision_date=original_decision_date,
    )

    # Build the agent. The eval always controls the mission (rating prompt +
    # original output) and the output schema (rating schema) — these are
    # not overridable by pack agent options. The pack may configure channel,
    # rules_text, model, max_rounds, etc. via rating_agent_opt.
    #
    # For the llm agent: channel='pack' (render evidence up front) and
    # rules_text='' (suppress PIT rules — the rater has hindsight) are the
    # eval defaults; the pack can override either via rating_agent_opt.
    #
    # Model pin: extracted from AgentSpec.model.id, same as simulate.cell.build_agent.
    params = dict(agent_spec.options)
    if agent_spec.model.id:
        params.setdefault("model", agent_spec.model.id)
    params["mission_text"] = instruction
    params["output_schema_text"] = rating_schema
    params.setdefault("channel", "pack")
    params.setdefault("rules_text", "")
    try:
        agent = agent_factory.build(agent_spec.name, **params)
    except TypeError:
        # Agent doesn't accept some of these params — fall back to bare options.
        agent = agent_factory.build(agent_spec.name, **agent_spec.options)

    response = invoke(agent, env)
    env.close()

    if response.outcome != "ok" or not response.views:
        logger.warning(
            "eval: rating agent returned %s for %s@%s: %s",
            response.outcome, cell_name, original_decision_date, response.detail,
        )
        return None

    # Extract the rating from the agent's view.
    view = response.views.get(Symbol(cell_name))
    if view is None:
        # Take the first view if the symbol doesn't match.
        view = next(iter(response.views.values()), None)
    if view is None:
        return None

    # Return the rating as a plain dict (all view fields, including extras)
    # plus the usage/cost data from the agent response.
    rating = view.model_dump()
    rating["_eval_outcome"] = response.outcome
    rating["_eval_detail"] = response.detail
    rating["_n_llm_calls"] = response.usage.n_llm_calls
    rating["_tokens_in"] = response.usage.tokens_in
    rating["_tokens_out"] = response.usage.tokens_out
    rating["_cost_usd"] = response.usage.cost_usd
    rating["_cost_basis"] = response.usage.basis
    return rating


def _compose_rating_instruction(
    *,
    rating_prompt: str,
    original_output: dict,
    rating_schema: str,
    cell_name: str,
    original_decision_date: Date,
) -> str:
    """Compose the instruction the rating agent receives.

    The rater sees: the rating prompt (what to evaluate), the original
    agent's full output (what was recommended), and the output schema
    (what shape its rating should take). It does NOT see the original
    agent's identity — the rating is blind to which model produced the
    output, to avoid biasing the rater.
    """
    parts = [
        rating_prompt.strip(),
        "",
        "---",
        f"## Cell: {cell_name}  |  Original decision date: {original_decision_date.isoformat()}",
        "",
        "## Original agent output to evaluate",
        "",
        "```json",
        json.dumps(original_output, indent=2, default=str),
        "```",
    ]
    if rating_schema:
        parts.extend([
            "",
            "## Rating schema",
            "",
            "```json",
            rating_schema,
            "```",
        ])
    parts.append("")
    return "\n".join(parts)


def _build_rating_sources(
    strategy_root: Path,
    eval_spec: EvalSpec,
    market_config: MarketConfig,
) -> dict[str, DataSource]:
    """Build data sources for the rating agent.

    Only the tool kinds listed in ``eval_spec.rating_tools`` are bound.
    The sources are built from the pack's strategy.toml data bindings,
    so the same MCP server config is reused — but with lookback caps
    removed (overridden to 100 years) so the full timeline is visible
    from today's vantage point.
    """
    from fintel.strategy.load import load as load_strategy

    paths = load_strategy(strategy_root)
    manifest = paths.manifest

    # Filter data bindings to only the rating tools.
    rating_kinds = set(eval_spec.rating_tools)
    filtered_data = [b for b in manifest.data if b.kind in rating_kinds]
    if not filtered_data:
        logger.warning(
            "agent_eval: no data bindings match rating_tools=%s — rater gets no tools",
            sorted(rating_kinds),
        )
        return {}

    # Override lookback_days to a very large value so the full timeline
    # is visible from today. The pack's lookback_days (e.g. 365) is correct
    # for the simulation agent (whose decision date is in 2018-2019) but
    # would clip the timeline when the rater's decision date is today.
    widened = []
    for binding in filtered_data:
        extra = dict(binding.params)
        extra["lookback_days"] = _NO_LOOKBACK_CAP
        widened.append(binding.model_copy(update=extra))

    return build_data_sources(widened, config=market_config)
