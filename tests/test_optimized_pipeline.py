"""Tests for the standalone optimized pipeline and its fintel-adapter contract.

These run without a network or OpenRouter key: a fake LLM client returns canned
completions per stage, proving the pipeline is host-agnostic (no fintel imports
in the agent) and that its raw ``submit_args`` round-trip through
``emit.parse_views`` into typed ``View`` / ``SourceRef`` objects — including the
provenance form of ``sources_cited`` (objects with source_type / source_id /
excerpt).
"""

from __future__ import annotations

from typing import Any

from fintel.agents import emit
from fintel.agents.installed.optimized_agent import OptimizedAgent


class FakeToolCall:
    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class FakeUsage:
    def __init__(self, tokens_in: int = 10, tokens_out: int = 20, cost: float = 0.001) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost
        self.basis = "reported"


class FakeCompletion:
    def __init__(
        self, *, text: str = "", tool_calls=(), finish: str = "stop", model: str = "fake/m"
    ) -> None:
        self.text = text
        self.tool_calls = list(tool_calls)
        self.finish_reason = finish
        self.model = model
        self.usage = FakeUsage()

    def call_named(self, name: str) -> FakeToolCall | None:
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None


class FakeLLM:
    """Returns a canned completion per stage, recognised by force_tool / prompt."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=(), force_tool=None, max_tokens=None) -> FakeCompletion:
        self.calls.append({"force_tool": force_tool, "max_tokens": max_tokens})
        user = messages[-1]["content"]
        if force_tool == "submit_verification":
            return FakeCompletion(
                tool_calls=[
                    FakeToolCall(
                        "submit_verification",
                        {
                            "recommendation": "KEEP",
                            "severity": "none",
                            "corrections": [],
                        },
                    )
                ]
            )
        if force_tool == "submit_views":
            return FakeCompletion(
                tool_calls=[
                    FakeToolCall(
                        "submit_views",
                        {
                            "views": [
                                {
                                    "symbol": "AAPL",
                                    "score": 0.4,
                                    "rationale": "cheap vs history",
                                    "key_factors": ["pe_diluted=36.2x"],
                                    "sources_cited": [
                                        {
                                            "source_type": "prices",
                                            "source_id": "2026-04-23",
                                            "excerpt": "close=177.19",
                                        },
                                        {
                                            "source_type": "fundamentals",
                                            "source_id": "2026-03-29",
                                            "excerpt": "revenue=215.9B",
                                        },
                                    ],
                                }
                            ],
                        },
                    )
                ]
            )
        if "Quantitative evidence" in user:
            return FakeCompletion(
                text="quant report\n## key evidence\nprices | 2026-04-23 | close=177.19"
            )
        return FakeCompletion(text="qual report\n## key evidence\nnews | 2026-04-22 | beat Q1")


def test_the_pipeline_has_no_fintel_imports():
    import fintel.agents.installed.optimized_agent as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "fintel" not in src.split('"""', 2)[2]  # docstring may mention it; code body must not


def test_decide_one_runs_the_four_stage_pipeline():
    stages: list[tuple[str, str]] = []
    llm = FakeLLM()
    agent = OptimizedAgent(llm=llm, on_stage=lambda s, sym: stages.append((s, sym)))
    res = agent.decide_one(
        symbol="AAPL", trade_date="2026-04-24", quant_evidence="Q", qual_evidence="N"
    )

    assert res.error is None
    assert res.submit_args is not None
    assert [c.stage for c in res.calls] == [
        "quantitative_specialist",
        "qualitative_specialist",
        "independent_verifier",
        "final_pm",
    ]
    assert stages == [
        ("quantitative_specialist", "AAPL"),
        ("qualitative_specialist", "AAPL"),
        ("independent_verifier", "AAPL"),
        ("final_pm", "AAPL"),
    ]


def test_decide_one_skips_the_verifier_when_disabled():
    llm = FakeLLM()
    agent = OptimizedAgent(llm=llm, enable_verification=False)
    res = agent.decide_one(
        symbol="AAPL", trade_date="2026-04-24", quant_evidence="Q", qual_evidence="N"
    )
    assert [c.stage for c in res.calls] == [
        "quantitative_specialist",
        "qualitative_specialist",
        "final_pm",
    ]


def test_submit_args_round_trip_into_typed_views_with_provenance():
    llm = FakeLLM()
    agent = OptimizedAgent(llm=llm)
    res = agent.decide_one(
        symbol="AAPL", trade_date="2026-04-24", quant_evidence="Q", qual_evidence="N"
    )
    views, notes = emit.parse_views(res.submit_args, decidable=frozenset({"AAPL"}))
    assert notes == []
    v = views["AAPL"]
    assert v.score == 0.4
    # Platform defaults — not solicited from the model in submit_views.
    assert v.time_horizon == "quarter"
    assert v.conviction == 0.5
    assert len(v.sources_cited) == 2
    assert v.sources_cited[0].source_type == "prices"
    assert v.sources_cited[0].source_id == "2026-04-23"
    assert v.sources_cited[0].excerpt == "close=177.19"
    assert v.sources_cited[1].source_type == "fundamentals"


def test_legacy_string_sources_cited_still_parse():
    payload = {
        "views": [
            {"symbol": "AAPL", "score": 0.3, "rationale": "ok", "sources_cited": ["prices", "news"]}
        ]
    }
    views, _ = emit.parse_views(payload, decidable=frozenset({"AAPL"}))
    v = views["AAPL"]
    assert [s.source_type for s in v.sources_cited] == ["prices", "news"]
    assert all(s.excerpt is None for s in v.sources_cited)


def test_a_missing_llm_is_a_clear_error_not_a_silent_crash():
    agent = OptimizedAgent(llm=None)
    try:
        agent.decide_one(
            symbol="AAPL", trade_date="2026-04-24", quant_evidence="Q", qual_evidence="N"
        )
    except RuntimeError as exc:
        assert "llm" in str(exc).lower()
    else:
        raise AssertionError("expected a RuntimeError when llm is not injected")


def test_pm_not_calling_submit_is_reported_as_an_error():
    class NoSubmit(FakeLLM):
        def complete(self, messages, *, tools=(), force_tool=None, max_tokens=None):
            if force_tool == "submit_views":
                return FakeCompletion(text="I forgot to call the tool", finish="stop")
            return super().complete(
                messages, tools=tools, force_tool=force_tool, max_tokens=max_tokens
            )

    agent = OptimizedAgent(llm=NoSubmit())
    res = agent.decide_one(
        symbol="AAPL", trade_date="2026-04-24", quant_evidence="Q", qual_evidence="N"
    )
    assert res.submit_args is None
    assert res.error is not None
    assert "submit_views" in res.error
