"""Structural guard: every catalog-declared strategy param must survive the
factory → bindings → MCP rebuild round-trip.

The "Tool not found" failure we hit when adding event_timeline/country_health
was caused by `_write_bindings` dropping strategy-owned params (event_file,
lookback_days): the MCP rebuild used catalog defaults, the source failed to
construct, and the tools never registered. That was patched for the specific
case; these tests guard the *class* — if a future factory function or catalog
entry drops or forgets a param, the round-trip breaks here in CI, not at a
smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fintel.market import catalog
from fintel.market.factory import build_data_source
from fintel.market.settings import MarketConfig
from fintel.models.market import DataBinding


def _event_file(tmp_path: Path) -> Path:
    p = tmp_path / "event.md"
    p.write_text(
        "| entry_date | event_date | actors | headline | summary |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 2018-01-01 | 2018-01-01 | USA | Test | Test event |\n"
    )
    return p


def _build(source_name: str, kind: str, *, config: MarketConfig, tmp_path: Path) -> object:
    params: dict = {}
    if source_name == "event_timeline":
        params["event_file"] = str(_event_file(tmp_path))
    binding = DataBinding(kind=kind, source=source_name, **params)
    return build_data_source(binding, config=config)


def _strategy_params(info: catalog.SourceInfo) -> list[catalog.Param]:
    """All declared params — per_call or not. The strategy's binding sets the
    default for every param; per_call only means the agent may override at call
    time. All of them must round-trip through bindings for MCP rebuild."""
    return list(info.params)


@pytest.fixture(autouse=True)
def _register():
    catalog.register_builtins()


@pytest.mark.parametrize(
    "name,kind",
    [
        (s.name, s.kind)
        for s in catalog.sources()
        if not s.is_computed and s.name != "synthetic_prices"
    ],
)
def test_every_strategy_default_param_is_an_attribute_on_the_source(name, kind, tmp_path):
    """A catalog param with a non-None default is one the factory bakes into the
    source instance. If it's not exposed as an attribute, `_write_bindings`
    silently skips it and the MCP rebuild uses the catalog default instead of
    the strategy's value — the 'Tool not found' failure class.

    Params with ``default=None`` are pure call-time knobs (the agent passes them
    per call); the factory doesn't store them and there's nothing to persist.
    """
    config = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    source = _build(name, kind, config=config, tmp_path=tmp_path)
    for param in _strategy_params(catalog.source(name)):
        if param.default is None:
            continue  # call-time knob; not strategy-owned
        assert hasattr(source, param.name), (
            f"source {name!r} is missing attribute {param.name!r} declared in "
            f"the catalog with default {param.default!r}; the factory likely "
            "dropped it from its `keep` set, which means _write_bindings won't "
            "persist it and the MCP rebuild will use the default — the "
            "'Tool not found' failure class."
        )


def test_bindings_roundtrip_rebuilds_the_same_tools(tmp_path):
    """The invariant end-to-end: orchestrator builds sources → writes bindings
    → MCP rebuild reads bindings → reconstructs the same tool set. If any
    strategy param is lost on the way, the rebuilt tools differ."""
    from fintel.agents.adapters.base import BINDINGS_FILE, SubprocessAgent
    from fintel.environment import Cell, RuntimeConfig, build_environment
    from fintel.environment.mcp_server import rebuild_environment
    from fintel.market.factory import build_data_sources

    config = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    bindings = [
        DataBinding(
            kind="event_timeline",
            source="event_timeline",
            event_file=str(_event_file(tmp_path)),
            lookback_days=90,
        ),
        DataBinding(kind="country_health", source="country_health", lookback_days=180),
    ]
    sources = build_data_sources(bindings, config=config)
    env = build_environment(
        cell=Cell(
            run_id="r1", decision_date=__import__("datetime").date(2018, 7, 5), symbols=("USA",)
        ),
        sources=sources,
        universe=["USA", "CHN"],
        kinds=tuple(sources),
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
        market_config=config,
    )

    class _Script(SubprocessAgent):
        name = "t"
        version = "1"

        def build_command(self, env, mcp_server_cmd):
            return ["true"]

        def enforce_pit_policy(self, env, mcp_server_cmd):
            return None

    _Script()._write_bindings(env)
    payload = json.loads((env.session.path / BINDINGS_FILE).read_text())
    bound_kinds = sorted(b["kind"] for b in payload["bindings"])
    assert bound_kinds == ["country_health", "event_timeline"]

    # MCP rebuild sees the same kinds + tools
    env2 = rebuild_environment(env.session.path)
    assert sorted(env2.tools.bound) == bound_kinds

    # And the strategy params survived
    et = next(b for b in payload["bindings"] if b["kind"] == "event_timeline")
    assert et["event_file"] == str(_event_file(tmp_path))
    assert et["lookback_days"] == 90
    ch = next(b for b in payload["bindings"] if b["kind"] == "country_health")
    assert ch["lookback_days"] == 180
