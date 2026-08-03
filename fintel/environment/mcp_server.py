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

    Critical: this process **attaches** to the orchestrator's ``access.jsonl``
    rather than creating a new session. Otherwise tool reads live only in the
    MCP process's memory and the parent's health audit sees ``n_reads: 0``.
    """
    from datetime import date as Date

    from fintel.environment.access import DataAccess
    from fintel.environment.base import Environment
    from fintel.environment.cell import Cell
    from fintel.environment.factory import build_policy
    from fintel.environment.session import TRACE_FILE
    from fintel.environment.tools import ToolSurface
    from fintel.environment.trace import AccessLog
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding

    cell_data = read_cell(session_path)
    cell = Cell(
        run_id=cell_data["run_id"],
        decision_date=Date.fromisoformat(cell_data["decision_date"]),
        symbols=tuple(cell_data["symbols"]),
        scope=cell_data.get("scope", "single_name"),
    )
    bindings_data = load_bindings(session_path)
    bindings = [DataBinding(**b) for b in bindings_data["bindings"]]
    config = MarketConfig.from_dict(bindings_data.get("config"))
    sources = build_data_sources(bindings, config=config)
    universe = bindings_data.get("universe", list(cell.symbols))
    peers = bindings_data.get("peers", False)
    kinds = tuple(bindings_data.get("kinds", [])) or tuple(sources)

    policy = build_policy(
        cell=cell, kinds=kinds, universe=universe, peers=peers
    )
    log = AccessLog(cell=cell, path=session_path / TRACE_FILE, attach=True)
    access = DataAccess(cell=cell, sources=sources, policy=policy, on_read=log.record)
    return Environment(
        cell=cell,
        access=access,
        policy=policy,
        log=log,
        tools=ToolSurface(
            access=access,
            bound={kind: getattr(src, "name", "") for kind, src in sources.items()},
        ),
        session=None,
        market_config=config,
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
    """Register one catalog-derived tool with an explicit parameter signature.

    FastMCP builds the JSON schema from the Python signature. A bare
    ``**kwargs`` collapses to a single ``kwargs`` property — which is exactly
    the failure the openclaw agent hit (``requires ['symbol']; got ['kwargs']``).
    We synthesize keyword-only params from ``ToolSpec.schema`` instead.
    """
    import inspect

    props = spec.schema.get("properties") or {}
    required = set(spec.schema.get("required") or [])
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}

    parameters: list[inspect.Parameter] = []
    for name, prop in props.items():
        ann = type_map.get(prop.get("type", "string"), Any)
        if name in required:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=ann)
            )
        else:
            default = prop["default"] if "default" in prop else None
            parameters.append(
                inspect.Parameter(
                    name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=ann
                )
            )

    def _impl(**kwargs: Any) -> dict:
        return env.tools.call(spec.name, kwargs)

    _impl.__name__ = spec.name
    _impl.__doc__ = spec.description
    _impl.__signature__ = inspect.Signature(parameters)
    server.add_tool(_impl, name=spec.name, description=spec.description)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    serve()


if __name__ == "__main__":
    main()
