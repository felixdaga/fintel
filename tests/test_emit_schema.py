"""Pack output_schema.json → submit_views tool schema + runtime validation."""

from __future__ import annotations

import json

from fintel.agents import emit

GEOPOL_ITEM = {
    "type": "object",
    "title": "GeopolView",
    "$defs": {
        "SourceRef": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string"},
                "source_id": {"type": "string"},
            },
            "required": ["source_type", "source_id"],
        }
    },
    "properties": {
        "symbol": {"type": "string"},
        "score": {"type": "number", "minimum": -1, "maximum": 1},
        "conviction": {"type": "number", "minimum": 0, "maximum": 1},
        "time_horizon": {"type": "string"},
        "rationale": {"type": "string"},
        "sources_cited": {"type": "array", "items": {"$ref": "#/$defs/SourceRef"}},
    },
    "required": ["symbol", "score", "conviction", "time_horizon", "rationale"],
}


def test_pack_native_extras_survive_into_view():
    """A pack schema may declare native fields (e.g. geopol's threat_score).
    parse_views must pass them through so decision.json follows the pack
    output_schema, not just the platform View keys."""
    views, notes = emit.parse_views(
        {
            "views": [
                {
                    "symbol": "USA",
                    "score": -0.3,
                    "conviction": 0.35,
                    "time_horizon": "retaliate",
                    "rationale": "x",
                    "threat_score": -0.3,
                    "action_score": -0.3,
                    "action_level": "retaliate",
                }
            ]
        },
        decidable=frozenset({"USA"}),
    )
    assert notes == []
    v = views["USA"]
    assert v.score == -0.3
    assert v.threat_score == -0.3  # type: ignore[attr-defined]
    assert v.action_score == -0.3  # type: ignore[attr-defined]
    assert v.action_level == "retaliate"  # type: ignore[attr-defined]
    dumped = v.model_dump(mode="json", exclude_none=True)
    assert dumped["threat_score"] == -0.3
    assert dumped["action_score"] == -0.3
    assert dumped["action_level"] == "retaliate"


def test_missing_score_falls_back_to_none_not_zero():
    """A pack that omits `score` (e.g. declares only threat_score) must not
    get a silent 0.0 neutral reading — the signal layer surfaces NaN."""
    views, notes = emit.parse_views(
        {"views": [{"symbol": "USA", "threat_score": -0.3, "rationale": "x"}]},
        decidable=frozenset({"USA"}),
    )
    v = views["USA"]
    assert v.score is None
    assert v.threat_score == -0.3  # type: ignore[attr-defined]
    # single_name_signal surfaces NaN, not 0.0
    from fintel.evaluate.signals import single_name_signal

    sig = single_name_signal(views)
    assert sig["USA"] != sig["USA"]  # NaN is not equal to itself


def test_default_submit_schema_requires_symbol_score_rationale():
    schema = emit.submit_schema(("AAPL",))
    required = schema["properties"]["views"]["items"]["required"]
    assert required == ["symbol", "score", "rationale"]


def test_advertised_submit_schema_strips_numeric_bounds():
    """min/max on the tool schema collapse mimo tool-call negatives to -1."""
    default = emit.submit_schema(("AAPL",))
    score = default["properties"]["views"]["items"]["properties"]["score"]
    assert score["type"] == "number"
    assert "minimum" not in score
    assert "maximum" not in score

    packed = emit.submit_schema(("USA",), item_schema=GEOPOL_ITEM)
    props = packed["properties"]["views"]["items"]["properties"]
    for key in ("score", "conviction"):
        assert "minimum" not in props[key]
        assert "maximum" not in props[key]


def test_pack_item_schema_is_wrapped_as_views_items_with_hoisted_defs():
    schema = emit.submit_schema(("USA",), item_schema=GEOPOL_ITEM)
    items = schema["properties"]["views"]["items"]
    assert items["required"] == [
        "symbol",
        "score",
        "conviction",
        "time_horizon",
        "rationale",
    ]
    assert "SourceRef" in schema["$defs"]
    assert "$defs" not in items


def test_validate_rejects_invented_dimensions_without_symbol():
    schema = emit.submit_schema(("USA",), item_schema=GEOPOL_ITEM)
    payload = {
        "views": [
            {"dimension": "trade_war_risk", "score": 0.75, "time_horizon": "3m", "rationale": "x"}
        ]
    }
    errors = emit.validate_submit(payload, schema)
    assert errors
    assert any("symbol" in e for e in errors)


def test_validate_accepts_pack_shaped_view():
    schema = emit.submit_schema(("USA",), item_schema=GEOPOL_ITEM)
    payload = {
        "views": [
            {
                "symbol": "USA",
                "score": 0.6,
                "conviction": 0.3,
                "time_horizon": "escalate_tariffs",
                "rationale": "RECOMMENDATION: hold course\n\nSUPPORTING RATIONALE: tariffs live",
            }
        ]
    }
    assert emit.validate_submit(payload, schema) == []


def test_abstain_skips_item_validation():
    schema = emit.submit_schema(("USA",), item_schema=GEOPOL_ITEM)
    assert (
        emit.validate_submit({"views": [], "abstain": True, "abstain_reason": "thin"}, schema) == []
    )


def test_for_agent_text_strips_numeric_bounds_and_comments():
    from fintel.environment.submit_schema import for_agent_text

    text = json.dumps(
        {
            "properties": {
                "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            },
            "$comment": "internal",
        }
    )
    out = json.loads(for_agent_text(text))
    assert "$comment" not in out
    assert out["properties"]["score"] == {"type": "number"}


def test_item_schema_from_text_roundtrip():
    text = json.dumps(GEOPOL_ITEM)
    assert emit.item_schema_from_text(text)["title"] == "GeopolView"
    assert emit.item_schema_from_text("") is None
    assert emit.item_schema_from_text("{not json") is None


def test_bindings_persist_output_schema_for_mcp(tmp_path):
    """Subprocess host writes pack schema into bindings.json for MCP rebuild."""
    import stat

    from fintel import agents
    from fintel.agents.adapters import SubprocessAgent
    from fintel.environment import Cell, RuntimeConfig, build_environment
    from fintel.market.factory import build_data_sources
    from fintel.market.settings import MarketConfig
    from fintel.models.market import DataBinding
    from tests import fixtures
    from tests.test_environment import DAY

    fixtures.register_all()
    market_config = MarketConfig(cache_root=tmp_path / "cache", offline=True)
    sources = build_data_sources(
        [DataBinding(kind="prices", source="flat_prices")], config=market_config
    )
    env = build_environment(
        cell=Cell(run_id="test-r1", decision_date=DAY, symbols=("AAPL",)),
        sources=sources,
        universe=["AAPL"],
        kinds=("prices",),
        runtime=RuntimeConfig(session_root=tmp_path / "sessions"),
        market_config=market_config,
    )

    script = tmp_path / "fake.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(stat.S_IRWXU)

    class _Script(SubprocessAgent):
        name = "fake"
        version = "1"

        def build_command(self, env, mcp_server_cmd):
            return [str(script)]

        def enforce_pit_policy(self, env, mcp_server_cmd):
            return None

    agents.invoke(
        _Script(
            binary=str(script),
            timeout_s=10,
            output_schema_text=json.dumps(GEOPOL_ITEM),
        ),
        env,
    )
    payload = json.loads((env.session.path / "bindings.json").read_text())
    assert payload["output_schema"]["title"] == "GeopolView"
    assert payload["output_schema"]["required"] == GEOPOL_ITEM["required"]


def test_mcp_submit_views_rejects_before_writing_result(tmp_path):
    """Invalid submit must not create result.json; valid submit must."""
    from fintel.agents import emit as emit_mod
    from fintel.environment import mcp_server

    session = tmp_path / "session"
    session.mkdir()
    (session / "cell.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "decision_date": "2018-07-05",
                "symbols": ["USA"],
                "scope": "single_name",
            }
        )
    )
    (session / "bindings.json").write_text(
        json.dumps(
            {
                "bindings": [],
                "kinds": [],
                "universe": ["USA"],
                "peers": False,
                "config": {"cache_root": str(tmp_path / "cache"), "offline": True},
                "output_schema": GEOPOL_ITEM,
            }
        )
    )

    # Unit-test the submit closure logic without running the MCP stdio loop.
    symbols = ("USA",)
    submit_params = emit_mod.submit_schema(symbols, item_schema=GEOPOL_ITEM)

    def submit_views(views, abstain=False, abstain_reason=""):
        payload = {"views": views, "abstain": abstain, "abstain_reason": abstain_reason}
        errors = emit_mod.validate_submit(payload, submit_params)
        if errors:
            return "REJECTED: " + "; ".join(errors)
        mcp_server.write_result(session, payload)
        return "recorded"

    bad = submit_views([{"dimension": "x", "score": 1}])
    assert bad.startswith("REJECTED")
    assert not (session / "result.json").exists()

    ok = submit_views(
        [
            {
                "symbol": "USA",
                "score": 0.5,
                "conviction": 0.4,
                "time_horizon": "hold",
                "rationale": "ok",
            }
        ]
    )
    assert ok == "recorded"
    assert (session / "result.json").is_file()
