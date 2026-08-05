"""The run echo: every input to a run, gathered into one snapshot and printed.

The point is to never go blind into a run. Before any cell executes, a reviewer
sees the whole world the run is about to act on: which agent and model, which
universe and schedule, which data bindings, the *exact* tool schemas the agent
will be offered (every param), the injected prompt (mission + output schema +
the composed instruction), the PIT policy, and the fingerprint that pins
reproducibility. The rendered block is emitted as a nerve ``run_echo`` event
(terminal + ``run.log``); the fingerprint digest is sealed into ``config.json``.

Everything gathered here already exists in memory at `run_run` time — RunConfig,
the manifest, the built sources, the tool descriptors, the mission/schema text,
the fingerprint. This module only collects it into one shape and renders it.
It does not fetch and does not run the agent; it is a snapshot, not a gate
(the probe is the gate).

The prompt-stitching helpers (`_render_tools`, `_compose_instruction`) are
inlined here rather than imported from `fintel.agents.prompts` because
`environment` (L4) is below `agents` (L5) on the layer ladder. They mirror the
agent's own composition so the echo shows the instruction the agent will
actually receive; if the agent's composition changes, this mirror should too.
"""

from __future__ import annotations

from typing import Any

from fintel.environment.tools import ToolSpec, spec_for, tool_name
from fintel.market import catalog
from fintel.models.run import RunConfig


def build_echo(
    *,
    run_config: RunConfig,
    strategy_description: str,
    sources: dict[str, Any],
    mission_text: str,
    output_schema_text: str,
    fingerprint: dict,
    probe: dict | None = None,
    prefetch: dict | None = None,
) -> dict:
    """Gather every run input into one dict.

    `sources` is the run's kind -> DataSource map (already built). Tool schemas
    are derived from the catalog for each binding whose source is known to it;
    a package-supplied source with no catalog entry is recorded as such (still
    readable, just not advertised as a typed tool).
    """
    decision_date = (
        run_config.schedule_dates[0] if run_config.schedule_dates else ""
    )
    tools = _tool_echo(run_config.data, sources, decision_date)
    instruction = _compose_instruction(
        mission_text=mission_text,
        output_schema_text=output_schema_text,
        decision_date=decision_date,
        symbols=tuple(sorted(run_config.universe_symbols)),
        tools=tools,
    )
    return {
        "run_id": run_config.run_id,
        "job_id": run_config.job_id,
        "k_index": run_config.k_index,
        "k_repeats": run_config.k_repeats,
        "agent": {
            "name": run_config.agent.name,
            "model": run_config.agent.model.id or None,
            "options": dict(run_config.agent.options),
            "pit_enforcement": fingerprint.get("adapter_params", {}).get(
                "pit_enforcement"
            ),
            "pit_deny": fingerprint.get("adapter_params", {}).get("pit_deny", []),
        },
        "strategy": {
            "name": run_config.strategy.name,
            "path": run_config.strategy.path,
            "digest": run_config.strategy.digest,
            "scope": run_config.scope,
            "description": strategy_description,
        },
        "universe": {
            "ref": run_config.universe.model_dump(),
            "resolved_symbols": list(run_config.universe_symbols),
        },
        "schedule": {
            "ref": run_config.schedule.model_dump(),
            "resolved_dates": list(run_config.schedule_dates),
        },
        "data": [
            {"kind": b.kind, "source": b.source} for b in run_config.data
        ],
        "tools": tools,
        "prompt": {
            "mission": mission_text,
            "output_schema": output_schema_text,
            "composed_instruction": instruction,
        },
        "fingerprint": fingerprint,
        "probe": probe,
        "prefetch": prefetch,
    }


def _tool_echo(
    bindings: list, sources: dict[str, Any], decision_date: str
) -> list[dict]:
    """One entry per bound kind: name, description, full JSON schema (the params)."""
    out: list[dict] = []
    for binding in bindings:
        kind = binding.kind
        source_name = binding.source
        if not catalog.has_source(source_name):
            out.append(
                {
                    "kind": kind,
                    "name": tool_name(kind),
                    "source": source_name,
                    "description": None,
                    "schema": None,
                    "note": "package-supplied source; no catalog entry, not advertised as a typed tool",
                }
            )
            continue
        spec: ToolSpec = spec_for(kind, source_name, decision_date=decision_date)
        out.append(
            {
                "kind": kind,
                "name": spec.name,
                "source": source_name,
                "description": spec.description,
                "schema": spec.schema,
                "required": list(spec.required),
            }
        )
    return out


def _render_tools_manual(tools: list[dict]) -> str:
    """Mirror of `agents.prompts.render_tools` for the echo. Kept inline to
    respect the layer ladder (environment may not import from agents)."""
    typed = [t for t in tools if t.get("schema") is not None]
    if not typed:
        return "No data tools are bound for this cell."
    parts = [
        "Every tool below enforces point-in-time semantics: nothing dated on or "
        "after the decision date is ever returned, and the boundary cannot be "
        "passed, widened, or bypassed."
    ]
    for t in typed:
        req = t.get("required", [])
        req_line = f"  required: {req}" if req else ""
        parts.append(f"- {t['name']}:{req_line}\n{t['description']}")
    return "\n\n".join(parts)


def _compose_instruction(
    *,
    mission_text: str,
    output_schema_text: str,
    decision_date: str,
    symbols: tuple[str, ...],
    tools: list[dict],
) -> str:
    """Mirror of `agents.prompts.compose_instruction` for the echo. Uses the
    run's first decision date and resolved universe; the real per-cell
    instruction is identical in shape (symbols may narrow to one for a
    single_name cell)."""
    listed = ", ".join(symbols) or "the assigned symbols"
    parts: list[str] = []
    if mission_text.strip():
        parts.append(mission_text.strip())
    parts.append(f"Decision date: {decision_date or 'UNKNOWN'}. You are deciding on: {listed}.")
    manual = _render_tools_manual(tools)
    if manual and "No data tools" not in manual:
        parts.append(f"## Tools\n{manual.strip()}")
        parts.append(
            "Gather the evidence you need with the tools above, then call "
            "submit_views exactly once with your answer."
        )
    if output_schema_text and output_schema_text.strip():
        parts.append(f"## Output schema (from the strategy pack)\n{output_schema_text.strip()}")
    return "\n\n".join(parts)


def render_echo(echo: dict) -> str:
    """A human-readable block for the terminal — the 'print everything' view.

    Compact but complete: agent, universe, schedule, data, tools (with params),
    prompt lengths, PIT policy, fingerprint digest, probe/prefetch outcomes.
    Full mission/schema/instruction text lives in the strategy pack; the
    terminal shows lengths plus a snippet so the line stays scannable. The
    block is also recorded on the ``run_echo`` nerve event in ``run.log``.
    """
    lines: list[str] = []
    a = echo["agent"]
    s = echo["strategy"]
    u = echo["universe"]
    sch = echo["schedule"]
    lines.append("== run echo")
    lines.append(
        f"   agent:    {a['name']}  model={a.get('model') or '(profile default)'}  "
        f"pit={a.get('pit_enforcement')}"
    )
    if a.get("pit_deny"):
        lines.append(f"   pit deny:  {', '.join(a['pit_deny'])}")
    lines.append(
        f"   strategy: {s['name']}  scope={s['scope']}  digest={_short(s.get('digest'))}"
    )
    lines.append(
        f"   universe: {u['ref']}  symbols={u['resolved_symbols']}"
    )
    lines.append(
        f"   schedule: {sch['ref']}  dates={sch['resolved_dates']}"
    )
    lines.append(f"   data:     {echo['data']}")
    lines.append("   tools:")
    for t in echo["tools"]:
        params = sorted((t.get("schema") or {}).get("properties", {}).keys())
        req = t.get("required", [])
        req_mark = " (required)" if req else ""
        if t.get("schema") is None:
            lines.append(f"     - {t['name']:16} [{t['source']}] {t.get('note','')}")
        else:
            lines.append(
                f"     - {t['name']:16} params={params}{req_mark}"
            )
    p = echo["prompt"]
    lines.append(
        f"   prompt:   mission={len(p['mission'])}c  schema={len(p['output_schema'])}c  "
        f"instruction={len(p['composed_instruction'])}c"
    )
    if echo.get("probe"):
        pr = echo["probe"]
        lines.append(
            f"   probe:    ok={pr['ok']}  kinds={len(pr['kinds'])}  elapsed={pr.get('elapsed_ms')}ms"
        )
    if echo.get("prefetch"):
        pf = echo["prefetch"]
        lines.append(
            f"   prefetch: warmed={pf.get('n_warmed')}  failed={pf.get('n_failed')}  "
            f"elapsed={pf.get('elapsed_ms')}ms"
        )
    lines.append(f"   fingerprint: {_short(echo['fingerprint'].get('digest'))}")
    return "\n".join(lines)


def _short(value: Any, n: int = 12) -> str:
    s = str(value or "")
    return s[:n] + ("…" if len(s) > n else "")
