"""The MCP stdio server: a subprocess agent's window onto one cell.

A CLI agent (OpenClaw, Claude Code) talks to its tools over MCP. This server is
that transport — and it is the only transport in the tree, so the old repo's
duplication of an MCP server and a LangChain Toolkit reimplementing the same
tools does not recur.

The server reads its cell from the session directory at startup and rebuilds the
environment from the bindings stored alongside it. It serves exactly that cell's
tools and nothing else, then writes the agent's answer to `result.json` and exits.

One cell per process is structural, not policed: the server is a stdio subprocess
that dies when the CLI exits, so a reused gateway process cannot keep serving the
first cell it ever loaded — the failure the old slot pool masked. If a long-lived
gateway is ever introduced, the rule it must carry is in `architecture.md`: refuse
a second cell, fail loudly, let the adapter restart.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from fintel.environment.session import read_cell, session_dir_from_env

logger = logging.getLogger(__name__)

BINDINGS_FILE = "bindings.json"


def load_bindings(session_path: Path) -> dict:
    path = session_path / BINDINGS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {BINDINGS_FILE} in {session_path}")
    return json.loads(path.read_text())


def rebuild_environment(session_path: Path):
    """Rebuild the Environment from the session directory.

    The orchestrator wrote `cell.json` and `bindings.json`; here we read them and
    call the same factory the orchestrator used. The sources are rebuilt from the
    catalog, so the server sees exactly the tools the strategy declared.
    """
    from fintel.environment.factory import build_environment
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding

    cell_data = read_cell(session_path)
    from datetime import date as Date

    from fintel.environment.cell import Cell

    cell = Cell(
        run_id=cell_data["run_id"],
        decision_date=Date.fromisoformat(cell_data["decision_date"]),
        symbols=tuple(cell_data["symbols"]),
        scope=cell_data.get("scope", "single_name"),
    )
    bindings_data = load_bindings(session_path)
    bindings = [DataBinding(**b) for b in bindings_data["bindings"]]
    config = MarketConfig(**bindings_data.get("config", {}))
    sources = build_data_sources(bindings, config=config)
    universe = bindings_data.get("universe", list(cell.symbols))
    peers = bindings_data.get("peers", False)
    return build_environment(
        cell=cell,
        sources=sources,
        universe=universe,
        kinds=tuple(bindings_data.get("kinds", [])) or None,
        peers=peers,
    )


def write_result(session_path: Path, payload: dict) -> None:
    (session_path / "result.json").write_text(json.dumps(payload, default=str, indent=2))


def serve() -> None:
    """Run the MCP stdio server. Entry point for `python -m fintel.agents.installed.mcp_server`."""
    from mcp.server.fastmcp import FastMCP

    session_path = session_dir_from_env()
    env = rebuild_environment(session_path)
    symbols = tuple(sorted(env.policy.decidable))

    server = FastMCP(name="fintel")

    @server.tool()
    def submit_views(views: list[dict], abstain: bool = False, abstain_reason: str = "") -> str:
        """Submit your final answer. Call exactly once when done."""
        payload = {"views": views, "abstain": abstain, "abstain_reason": abstain_reason}
        write_result(session_path, payload)
        return "recorded"

    for spec in env.tools.descriptors():
        _register_data_tool(server, spec, env)

    logger.info("fintel mcp server serving %s for %s", env.cell.name, symbols)
    server.run(transport="stdio")


def _register_data_tool(server, spec, env) -> None:
    """Register one catalog-derived tool on the MCP server."""

    @server.tool(name=spec.name, description=spec.description)
    def _tool(**kwargs: Any) -> dict:
        return env.tools.call(spec.name, kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    serve()


if __name__ == "__main__":
    main()
