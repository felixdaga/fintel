"""Claude Code adapter.

Claude Code is a CLI that reads MCP server config from `~/.claude.json` and takes
its instruction via stdin. Same host as OpenClaw; different config format and
delivery channel.
"""

from __future__ import annotations

from dataclasses import dataclass

from fintel.agents.installed.base import CliFlag, SubprocessAgent


@dataclass
class ClaudeCodeAgent(SubprocessAgent):
    """Runs `claude --print --output-format=stream-json` with stdin instruction."""

    model: str = "claude-sonnet-4-20250514"
    binary: str = "claude"
    name: str = "claude-code"
    version: str = "1"
    timeout_s: float = 600.0
    max_turns: int = 20

    CLI_FLAGS = [
        CliFlag("max_turns", "--max-turns", "int"),
    ]

    def build_command(self, env, mcp_server_cmd: list[str]) -> list[str]:
        return [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(self.max_turns),
        ]

    def mcp_config(self, env, mcp_server_cmd: list[str]) -> dict | None:
        """Claude Code reads MCP from `~/.claude.json` under `mcpServers`."""
        return {
            "mcpServers": {
                "fintel": {
                    "type": "stdio",
                    "command": mcp_server_cmd[0],
                    "args": mcp_server_cmd[1:],
                }
            }
        }

    @staticmethod
    def _instruction(env) -> str:
        symbols = ", ".join(sorted(env.policy.decidable))
        return (
            f"Decision date: {env.cell.decision_date.isoformat()}. "
            f"Score: {symbols}. Use the fintel MCP tools to gather evidence, "
            f"then call submit_views with your answer."
        )
