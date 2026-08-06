"""PIT tool policy — what an adapter must strip before a cell can run.

A backtest is only honest if the agent cannot reach uncontrolled knowledge:
live web, free filesystem, shell egress, non-PIT memory. Those are *threat*
channels. Sub-agents, planning, session status, and the fintel MCP surface are
*ability* channels — they stay on.

Every adapter declares how it meets this requirement via `pit_enforcement`:

  * ``"access"`` — in-process hosts (``llm``, scripted). No native tool surface;
    every read goes through ``Environment.access`` (already PIT-clamped).
  * ``"cli_deny"`` — subprocess CLIs (OpenClaw, Claude Code). The adapter must
    apply ``OPENCLAW_DENY`` / ``CLAUDE_CODE_DENY`` at launch and expose only the
    ``fintel`` MCP server for the cell. Preflight fails closed if it cannot.

The deny lists are the platform's; adapters only map them into the CLI's native
config shape. Importing a personal OpenClaw/Claude profile must not silently
keep web/fs — the adapter re-asserts this every cell, not once at setup.
"""

from __future__ import annotations

from typing import Any, Literal

PitEnforcement = Literal["access", "cli_deny"]

# The only MCP server a fintel cell may expose. Other servers (old delorean,
# the operator's GitHub/browser plugins, …) are uncontrolled knowledge channels
# and are stashed for the duration of the cell.
FINTEL_MCP_SERVER = "fintel"

# OpenClaw native tools that can break PIT. Mirrors delorean's PIT_TOOL_DENY.
# Deliberately does NOT include sessions_spawn / sessions_yield — those are
# agent ability (sub-agents), not knowledge egress.
OPENCLAW_DENY: tuple[str, ...] = (
    "group:fs",
    "group:runtime",
    "group:web",
    "group:memory",
    "group:ui",
    "group:media",
    "cron",
    "sessions_list",
    "sessions_history",
    "sessions_send",
)

# Claude Code native tools that can break PIT. Same threat classes, different
# names. Sub-agents (`Agent` / `Task`) stay available.
CLAUDE_CODE_DENY: tuple[str, ...] = (
    "WebSearch",
    "WebFetch",
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
)


def merge_deny(existing: list[str] | None, required: tuple[str, ...] | list[str]) -> list[str]:
    """Union, order-preserving: existing first, then any missing required entries."""
    out: list[str] = list(existing or [])
    seen = set(out)
    for item in required:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def missing_deny(existing: list[str] | None, required: tuple[str, ...] | list[str]) -> list[str]:
    have = set(existing or [])
    return [item for item in required if item not in have]


def apply_openclaw_tools(data: dict[str, Any]) -> dict[str, Any]:
    """Merge PIT denies into an openclaw.json `tools` block. Mutates and returns."""
    tools = data.setdefault("tools", {})
    if not isinstance(tools, dict):
        tools = {}
        data["tools"] = tools
    tools["deny"] = merge_deny(tools.get("deny"), OPENCLAW_DENY)
    # coding profile keeps session/plan ability; deny list is the safety net.
    tools.setdefault("profile", "coding")
    return data


def isolate_fintel_mcp(servers: dict[str, Any], fintel_entry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fintel server; return the stashed others for later restore."""
    stashed = {k: v for k, v in servers.items() if k != FINTEL_MCP_SERVER}
    servers.clear()
    servers[FINTEL_MCP_SERVER] = fintel_entry
    return stashed


def restore_mcp(servers: dict[str, Any], stashed: dict[str, Any]) -> None:
    """Put non-fintel servers back. Leaves the current fintel entry in place."""
    for name, cfg in stashed.items():
        if name == FINTEL_MCP_SERVER:
            continue
        servers.setdefault(name, cfg)


def openclaw_deny_ok(data: dict[str, Any]) -> list[str]:
    """Findings if an openclaw.json is missing required PIT denies."""
    tools = data.get("tools") if isinstance(data.get("tools"), dict) else {}
    missing = missing_deny(tools.get("deny"), OPENCLAW_DENY)
    return [f"openclaw tools.deny missing PIT entry {m!r}" for m in missing]


__all__ = [
    "CLAUDE_CODE_DENY",
    "FINTEL_MCP_SERVER",
    "OPENCLAW_DENY",
    "PitEnforcement",
    "apply_openclaw_tools",
    "isolate_fintel_mcp",
    "merge_deny",
    "missing_deny",
    "openclaw_deny_ok",
    "restore_mcp",
]
