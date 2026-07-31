"""OpenClaw adapter.

OpenClaw is a local agent harness that spawns an MCP server as a stdio
subprocess. The fintel MCP server is that subprocess, so every tool the agent
calls goes through the one clamped, recorded path.

The adapter is thin: it declares how to build the OpenClaw argv and how to
write the MCP server config in OpenClaw's native format. Everything else —
session setup, result collection, error classification — is on the base.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass

from fintel.agents.installed.base import CliFlag, SubprocessAgent


@dataclass
class OpenClawAgent(SubprocessAgent):
    """Runs `openclaw agent --local --json` with the fintel MCP server."""

    profile: str = "default"
    model: str = "claude-sonnet-4-20250514"
    message: str = ""
    binary: str = "openclaw"
    name: str = "openclaw"
    version: str = "1"
    timeout_s: float = 600.0

    CLI_FLAGS = [
        CliFlag("model", "--model", "str"),
    ]

    def build_command(self, env, mcp_server_cmd: list[str]) -> list[str]:
        instruction = self._instruction(env)
        return [
            self.binary,
            "agent",
            "--local",
            "--json",
            "--model",
            shlex.quote(self.model),
            "--message",
            shlex.quote(instruction),
        ]

    def mcp_config(self, env, mcp_server_cmd: list[str]) -> dict | None:
        """OpenClaw reads MCP servers from its config file."""
        return {
            "mcp": {
                "servers": {
                    "fintel": {
                        "command": mcp_server_cmd[0],
                        "args": mcp_server_cmd[1:],
                    }
                }
            }
        }

    @staticmethod
    def _instruction(env) -> str:
        symbols = ", ".join(sorted(env.policy.decidable))
        return (
            f"Decision date: {env.cell.decision_date.isoformat()}. "
            f"Score: {symbols}. Use the fintel tools to gather evidence, "
            f"then call {json.dumps('submit_views')} with your answer."
        )
