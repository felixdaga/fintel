"""Mission + tools manual + output schema — one composer, for tool-calling agents.

The old repo hand-wrote a "skills" block per adapter (`delorean/agents/prompts.py`)
describing every MCP tool's params, defaults and ranges — accurate the day it was
written, and free to drift the moment a tool's real behaviour changed underneath
it. Here the manual is rendered from `ToolSurface.descriptors()`, the same
catalog-derived source the tools themselves are built from, so it cannot describe
a tool that doesn't exist or omit one that does.

Only a tool-calling delivery needs the manual. A pack-channel agent (or any
non-tool-calling adapter — the old repo's `optimizedagent`) never sees it: it has
no tools to call, so listing them would be noise at best and a false promise at
worst.
"""

from __future__ import annotations

from fintel.environment.tools import ToolSpec


def render_tools(descriptors: tuple[ToolSpec, ...]) -> str:
    """The tools manual: one framing line, then each tool's own description.

    Each `ToolSpec.description` already carries the PIT boundary note and field
    list (built in `environment/tools.py` from the catalog), so this only adds
    the required-args line and stitches them together — nothing here can
    contradict what `ToolSurface.call()` actually accepts.
    """
    if not descriptors:
        return "No data tools are bound for this cell."
    parts = [
        "Every tool below enforces point-in-time semantics: nothing dated on or "
        "after the decision date is ever returned, and the boundary cannot be "
        "passed, widened, or bypassed."
    ]
    for spec in descriptors:
        required = ", ".join(spec.required) or "none"
        parts.append(f"### {spec.name}  (required: {required})\n{spec.description}")
    return "\n\n".join(parts)


def compose_instruction(
    *,
    mission: str = "",
    decision_date: str,
    symbols: tuple[str, ...],
    tools_manual: str | None = None,
    output_schema: str | None = None,
    submit_tool: str = "submit_views",
) -> str:
    """Stitch one instruction: strategy mission, task framing, tools, output schema.

    `tools_manual=None` means this delivery is not tool-calling — the block is
    omitted rather than shown empty, so a non-tool-calling agent isn't told about
    tools it has no way to call (mirrors the old repo's split between the
    tool-calling `OpenclawAgent`, which got the skills block, and
    `optimizedagent`, which didn't need it).
    """
    listed = ", ".join(symbols) or "the assigned symbols"
    parts: list[str] = []
    if mission.strip():
        parts.append(mission.strip())
    parts.append(f"Decision date: {decision_date}. You are deciding on: {listed}.")
    if tools_manual is not None:
        parts.append(f"## Tools\n{tools_manual.strip()}")
        parts.append(
            f"Gather the evidence you need with the tools above, then call "
            f"{submit_tool} exactly once with your answer."
        )
    if output_schema and output_schema.strip():
        parts.append(f"## Output schema (from the strategy pack)\n{output_schema.strip()}")
    return "\n\n".join(parts)


__all__ = ["compose_instruction", "render_tools"]
