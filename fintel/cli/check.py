"""``fintel check`` — wired vs default vs issues for a pack × agents.

Reads the central lists in ``fintel.strategy.inspect.PACK_FEATURES`` and
``fintel.agents.contract.PACK_CONTEXT_FIELDS``. Does not run cells.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from fintel.agents.contract import ABLATION_AGENT, inspect_agent
from fintel.agents.factory import names as agent_names
from fintel.strategy.inspect import PackReport, inspect_pack


def run_check(args: Namespace) -> int:
    package = Path(args.package).expanduser()
    if not package.is_dir():
        print(f"package not found: {package}")
        return 2

    if not getattr(args, "no_bootstrap", False):
        from fintel.utils.secrets import bootstrap_env

        bootstrap_env()

    pack = inspect_pack(package)
    agents = _agent_names(args)
    surfaces = [inspect_agent(name) for name in agents]
    issues = list(pack.problems)
    for feat in pack.features:
        if feat.id == "eval" and feat.status == "wired" and "missing" in feat.detail:
            issues.append(f"pack [eval] is wired but files are missing: {feat.detail}")
    for surface in surfaces:
        issues.extend(surface.issues)
        issues.extend(surface.preflight)
        issues.extend(_cross_issues(pack, surface))

    if args.json:
        print(
            json.dumps(
                {
                    "pack": pack.to_dict(),
                    "agents": [s.to_dict() for s in surfaces],
                    "issues": issues,
                },
                indent=2,
            )
        )
    else:
        print(_render(pack, surfaces, issues))

    return 1 if issues else 0


def _agent_names(args: Namespace) -> list[str]:
    if getattr(args, "all_agents", False):
        return list(agent_names())
    raw: list[str] = list(getattr(args, "agent", None) or [])
    out: list[str] = []
    for item in raw:
        for part in item.split(","):
            name = part.strip()
            if name and name not in out:
                out.append(name)
    return out


def _cross_issues(pack: PackReport, agent) -> list[str]:
    if not agent.known:
        return []
    issues: list[str] = []
    prefix = f"agent {agent.name!r}"
    if pack.has_alpha_view and agent.subagents and not agent.accepts.get("alpha_view_text"):
        issues.append(
            f"{prefix}: pack has an alpha view and this adapter declares sub-agents "
            "but does not accept alpha_view_text; specialists that never see "
            "mission.md will miss the thesis"
        )
    if pack.ablation_search_query:
        if agent.name != ABLATION_AGENT:
            issues.append(
                f"{prefix}: pack sets [ablation].search_query but the platform "
                f"only injects it for the {ABLATION_AGENT!r} agent; this run would ignore it"
            )
        elif not agent.accepts.get("search_query"):
            issues.append(
                f"{prefix}: pack sets [ablation].search_query but the adapter "
                "does not accept search_query"
            )
    return issues


def _render(pack: PackReport, surfaces: list, issues: list[str]) -> str:
    lines: list[str] = [
        f"strategy  {pack.name}",
        f"path      {pack.path}",
        "",
        "features",
    ]
    for feat in pack.features:
        lines.append(_row(feat.id, feat.status, feat.detail))
    if pack.kinds:
        lines.append(_row("data", "wired", ", ".join(pack.kinds)))
    else:
        lines.append(_row("data", "default", "no [[data]] bindings"))
    lines.append("")
    lines.append("scoring")
    for feat in pack.scoring_defaults:
        lines.append(_row(feat.id, feat.status, feat.detail))

    if pack.warnings:
        lines.append("")
        lines.append("warnings")
        for w in pack.warnings:
            lines.append(f"  - {w}")

    for surface in surfaces:
        lines.append("")
        lines.append(f"agent     {surface.name}")
        if not surface.known:
            lines.append(_row("status", "missing", "not in the registry"))
            continue
        lines.append(
            _row(
                "pit",
                "wired" if surface.pit_enforcement else "missing",
                surface.pit_enforcement or "undeclared",
            )
        )
        lines.append(
            _row(
                "subagents",
                "wired" if surface.subagents else "default",
                "declared — forward alpha_view_text into inner prompts"
                if surface.subagents
                else "single prompt; composed mission_text is enough",
            )
        )
        for field, ok in surface.accepts.items():
            if ok:
                extra = ""
                if field == "alpha_view_text" and pack.has_alpha_view:
                    extra = (
                        "pack thesis present"
                        if not surface.subagents
                        else "pack thesis present; forward to sub-agents"
                    )
                lines.append(_row(field, "wired", extra or "accepted"))
            else:
                lines.append(_row(field, "default", "not declared"))

    if issues:
        lines.append("")
        lines.append("issues")
        for item in issues:
            lines.append(f"  - {item}")
    else:
        lines.append("")
        lines.append("issues    none")
    return "\n".join(lines) + "\n"


def _row(name: str, status: str, detail: str) -> str:
    body = f"  {name:<22} {status:<8}"
    if detail:
        return f"{body} {detail}"
    return body.rstrip()
