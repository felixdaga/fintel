"""The shared prompt composer: mission + tools manual + output schema.

`render_tools` must derive its manual only from `ToolSpec` — anything else
would let a tool-calling instruction drift from what `ToolSurface` actually
serves. `compose_instruction` must omit the tools block entirely for a
non-tool-calling delivery, mirroring the old repo's split between
`OpenclawAgent` (got the skills block) and `optimizedagent` (didn't).
"""

from __future__ import annotations

from fintel.agents.prompts import compose_instruction, render_tools
from fintel.environment.tools import ToolSpec


def _spec(name: str, required: tuple[str, ...] = ("symbol",)) -> ToolSpec:
    return ToolSpec(
        name=name,
        kind=name,
        description=f"Fetch {name}.",
        schema={"required": list(required)},
    )


def test_render_tools_is_empty_message_with_no_descriptors():
    assert "No data tools" in render_tools(())


def test_render_tools_lists_every_descriptor_by_name():
    manual = render_tools((_spec("get_prices"), _spec("get_news", required=())))
    assert "get_prices" in manual
    assert "get_news" in manual
    assert "Fetch get_prices." in manual
    assert "Fetch get_news." in manual


def test_render_tools_reports_required_args():
    manual = render_tools((_spec("get_prices", required=("symbol", "lookback_days")),))
    assert "required: symbol, lookback_days" in manual


def test_render_tools_reports_none_when_nothing_required():
    manual = render_tools((_spec("get_universe", required=()),))
    assert "required: none" in manual


def test_compose_instruction_includes_mission_when_given():
    text = compose_instruction(
        mission="Be a fundamental analyst.", decision_date="2025-01-02", symbols=("AAPL",)
    )
    assert "Be a fundamental analyst." in text


def test_compose_instruction_omits_mission_block_when_blank():
    text = compose_instruction(mission="", decision_date="2025-01-02", symbols=("AAPL",))
    assert text.startswith("Decision date:")


def test_compose_instruction_lists_symbols_and_date():
    text = compose_instruction(mission="", decision_date="2025-06-01", symbols=("AAPL", "MSFT"))
    assert "2025-06-01" in text
    assert "AAPL, MSFT" in text


def test_compose_instruction_without_tools_manual_has_no_tools_section():
    """The pack channel (and any non-tool-calling agent) must not be told about
    tools it cannot call."""
    text = compose_instruction(mission="m", decision_date="2025-01-02", symbols=("AAPL",))
    assert "## Tools" not in text


def test_compose_instruction_with_tools_manual_includes_it_and_the_call_to_action():
    text = compose_instruction(
        mission="m",
        decision_date="2025-01-02",
        symbols=("AAPL",),
        tools_manual=render_tools((_spec("get_prices"),)),
    )
    assert "## Tools" in text
    assert "get_prices" in text
    assert "submit_views exactly once" in text


def test_compose_instruction_respects_a_custom_submit_tool_name():
    text = compose_instruction(
        mission="m",
        decision_date="2025-01-02",
        symbols=("AAPL",),
        tools_manual=render_tools((_spec("get_prices"),)),
        submit_tool="finish",
    )
    assert "finish exactly once" in text


def test_compose_instruction_includes_output_schema_when_given():
    text = compose_instruction(
        mission="m", decision_date="2025-01-02", symbols=("AAPL",), output_schema="score in [-1,1]"
    )
    assert "## Output schema" in text
    assert "score in [-1,1]" in text


def test_compose_instruction_omits_output_schema_block_when_blank():
    text = compose_instruction(
        mission="m", decision_date="2025-01-02", symbols=("AAPL",), output_schema=""
    )
    assert "## Output schema" not in text


def test_compose_instruction_defaults_symbols_label_when_empty():
    text = compose_instruction(mission="", decision_date="2025-01-02", symbols=())
    assert "the assigned symbols" in text
