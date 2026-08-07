"""Single-turn LLM fed the same evidence surface as :class:`OptimizedFintelAgent`.

The generic ``llm`` pack channel uses ``env.evidence()``, which cannot
pre-render ``web_search`` (needs a query). The optimized host instead builds
role-split packs via :class:`FintelEvidence` — including the fixed two-tier
web_search plan (4 structural @ strategy lookback + 1 updates @ 7d).

This adapter reuses that exact pack builder, then asks one LLM call (mission
as system prompt, packs as user content, forced ``submit_views``). No
specialists, no verifier — same information surface, different agent shape.

Feeding is strategy-aligned to the DJIA stock-rate packages (quant/qual kind
partition + web query templates in ``fintel.agents.evidence``). The LLM itself
stays strategy-agnostic: it only sees mission + packs.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, ClassVar

from fintel.agents import emit
from fintel.agents.evidence import (
    QUAL_KINDS,
    QUANT_KINDS,
    EvidenceConfig,
    FintelEvidence,
    _pit_company_name,
)
from fintel.agents.llm import OpenRouter, as_tool_spec
from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse, View
from fintel.models.trace import ReasoningTrace, TraceStep, Usage

logger = logging.getLogger(__name__)

_INT_FIELDS = {
    "max_tokens",
    "evidence_budget_chars",
    "web_structural_lookback_days",
    "web_update_lookback_days",
    "web_snippets_per_query",
}

RULES = (
    "Rules:\n"
    "- Every number you cite must come from the evidence packs below. Do not "
    "recall prices, results or events from memory; your memory postdates the "
    "decision date.\n"
    "- Absence of data is not bad news. A lookup marked 'empty' genuinely has "
    "nothing; one marked 'failed' broke, and tells you nothing either way.\n"
    f"- Finish by calling {emit.SUBMIT_TOOL} exactly once.\n"
    "- Declining is a real answer. If the evidence does not support a view, "
    "abstain and say why rather than scoring something you don't believe.\n"
    "- OUTPUT SHAPE: respond with ONLY the submit_views tool call. Do NOT write "
    "any free text before the tool call — it is discarded and risks truncating "
    "the call. Your entire thesis goes inside the view's rationale and "
    "key_factors fields."
)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class DjiaStrategyAdapterForLlmAgent:
    """One-shot LLM over the optimized evidence packs (quant + qual + web plan)."""

    model: str = "xiaomi/mimo-v2.5-pro"
    temperature: float = 0.0
    max_tokens: int | None = None  # uncapped — match optimized PM

    evidence_budget_chars: int = 400_000
    web_structural_lookback_days: int = 30
    web_update_lookback_days: int = 7
    web_snippets_per_query: int = 5
    company_names: dict[str, str] = field(default_factory=dict)

    # Strategy-pack context (wired by simulate.cell.build_agent).
    mission_text: str = ""
    output_schema_text: str = ""

    name: str = "djia_strategy_adapter_for_llm_agent"
    version: str = "1"
    pit_enforcement: ClassVar[PitEnforcement] = "access"

    _llm: OpenRouter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, str) and f.name in _INT_FIELDS:
                setattr(self, f.name, int(val) if val.strip() else None)

    @staticmethod
    def preflight_checks(**params: Any) -> list[str]:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return [
                "OPENROUTER_API_KEY is not set; the djia_strategy_adapter_for_llm_agent agent cannot call OpenRouter"
            ]
        return []

    def _client(self) -> OpenRouter:
        if self._llm is None:
            self._llm = OpenRouter.from_env(self.model, temperature=self.temperature)
        return self._llm

    def _evidence_config(self) -> EvidenceConfig:
        return EvidenceConfig(
            evidence_budget_chars=self.evidence_budget_chars,
            valuation_history_points=12,
            web_structural_lookback_days=self.web_structural_lookback_days,
            web_update_lookback_days=self.web_update_lookback_days,
            web_snippets_per_query=self.web_snippets_per_query,
            company_names=dict(self.company_names or {}),
        )

    def decide(self, env: Environment) -> AgentResponse:
        symbols = tuple(sorted(env.policy.decidable))
        if not symbols:
            return AgentResponse(views={}, outcome="empty", detail="no decidable symbols")

        decidable = frozenset(env.policy.decidable)
        views: dict[str, View] = {}
        steps: list[TraceStep] = []
        errors: list[str] = []
        usage = Usage()

        cell_name = env.cell.name
        decision_date = env.cell.decision_date.isoformat()
        bound = set(env.access.kinds)
        quant_kinds = tuple(k for k in QUANT_KINDS if k in bound)
        qual_kinds = tuple(k for k in QUAL_KINDS if k in bound)

        for sym in symbols:
            _emit(
                env.nerve,
                "agent_stage",
                cell=cell_name,
                decision_date=decision_date,
                stage="evidence",
                symbol=sym,
            )
            builder = FintelEvidence(
                access=env.access,
                symbol=sym,
                decision_date=env.cell.decision_date,
                config=self._evidence_config(),
                company_name=_pit_company_name(
                    getattr(env, "market_config", None), sym, env.cell.decision_date
                ),
            )
            _emit(
                env.nerve,
                "agent_stage",
                cell=cell_name,
                decision_date=decision_date,
                stage="evidence_quantitative",
                symbol=sym,
            )
            quant = builder.quantitative_block()
            _emit(
                env.nerve,
                "agent_stage",
                cell=cell_name,
                decision_date=decision_date,
                stage="evidence_qualitative",
                symbol=sym,
            )
            qual = builder.qualitative_block()

            submit = as_tool_spec(
                emit.SUBMIT_TOOL,
                emit.submit_description((sym,)),
                emit.submit_schema((sym,)),
            )
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(
                        symbol=sym,
                        trade_date=decision_date,
                        quant=quant,
                        qual=qual,
                        quant_kinds=quant_kinds,
                        qual_kinds=qual_kinds,
                    ),
                },
            ]

            _emit(
                env.nerve,
                "agent_stage",
                cell=cell_name,
                decision_date=decision_date,
                stage="one_shot",
                symbol=sym,
            )
            started = _now()
            try:
                completion = self._client().complete(
                    messages,
                    tools=(submit,),
                    force_tool=emit.SUBMIT_TOOL,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 — classified upstream by invoke
                errors.append(f"{sym}: {exc}")
                logger.warning(
                    "djia_strategy_adapter_for_llm_agent: %s LLM call failed — %s", sym, exc
                )
                continue

            step = TraceStep(
                step_id=uuid.uuid4().hex[:12],
                kind="llm_call",
                started_at=started,
                duration_ms=int((_now() - started).total_seconds() * 1000),
                model=completion.model or self.model,
                tokens_in=completion.usage.tokens_in or None,
                tokens_out=completion.usage.tokens_out or None,
                cost_usd=completion.usage.cost_usd,
                payload={
                    "ticker": sym,
                    "stage": "one_shot",
                    "finish_reason": completion.finish_reason,
                    "tool_calls": [c.name for c in completion.tool_calls],
                    "cost_basis": completion.usage.basis,
                    "evidence_surface": "fintel_evidence_quant_qual",
                },
            )
            steps.append(step)
            usage = usage.merge(
                Usage(
                    n_llm_calls=1,
                    tokens_in=completion.usage.tokens_in or 0,
                    tokens_out=completion.usage.tokens_out or 0,
                    cost_usd=completion.usage.cost_usd,
                    basis=completion.usage.basis or "unknown",
                )
            )

            call = completion.call_named(emit.SUBMIT_TOOL)
            if call is None:
                errors.append(
                    f"{sym}: model was required to call {emit.SUBMIT_TOOL} and did not "
                    f"(finish_reason={completion.finish_reason!r})"
                )
                continue

            reason = emit.abstained(call.arguments)
            parsed, notes = emit.parse_views(call.arguments, decidable=decidable)
            view = parsed.get(sym)
            if reason is not None and view is None:
                continue
            if view is None:
                errors.append(f"{sym}: PM submitted no view" + (f" ({notes})" if notes else ""))
                continue
            views[sym] = view

        outcome: Outcome
        detail = ""
        if views:
            outcome = "ok"
        elif errors:
            outcome = "parse_error"
            detail = "; ".join(errors)[:2000]
        else:
            outcome = "abstained"
            detail = "one-shot produced no views"

        return AgentResponse(
            views=views,
            outcome=outcome,
            detail=detail,
            usage=usage,
            trace=ReasoningTrace(
                final_explanation=(
                    f"djia_strategy_adapter_for_llm_agent: {len(views)}/{len(symbols)} views on "
                    f"{decision_date} via {self.model}; "
                    "FintelEvidence quant‖qual → one-shot submit_views."
                ),
                steps=steps,
                metadata={
                    "agent": self.name,
                    "version": self.version,
                    "model": self.model,
                    "evidence_surface": "fintel_evidence_quant_qual",
                    "errors": errors,
                    "mission_digest_present": bool(self.mission_text.strip()),
                },
            ),
        )

    def _system_prompt(self) -> str:
        body = self.mission_text.strip() or (
            "You are an equity analyst. Judge the security on the evidence "
            "available at the decision date, and score it."
        )
        return f"{body}\n\n{RULES}"

    def _schema_block(self) -> str:
        if not self.output_schema_text.strip():
            return ""
        return f"\n\n## Output schema (from the strategy pack)\n{self.output_schema_text.strip()}"

    def _user_prompt(
        self,
        *,
        symbol: str,
        trade_date: str,
        quant: str,
        qual: str,
        quant_kinds: tuple[str, ...],
        qual_kinds: tuple[str, ...],
    ) -> str:
        q_list = ", ".join(quant_kinds) if quant_kinds else "(none)"
        l_list = ", ".join(qual_kinds) if qual_kinds else "(none)"
        return (
            f"The instrument is {symbol}. The decision date is {trade_date}; all "
            f"evidence is strictly point-in-time before that date.\n\n"
            f"## Quantitative evidence\n"
            f"Kinds in this pack: {q_list}.\n"
            f"{quant}\n\n"
            f"## Qualitative evidence\n"
            f"Kinds in this pack: {l_list}.\n"
            f"{qual}"
            f"{self._schema_block()}"
        )


def _emit(nerve: Any, kind: str, **fields: Any) -> None:
    if nerve is None or not hasattr(nerve, "emit"):
        return
    try:
        nerve.emit(kind, **fields)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["DjiaStrategyAdapterForLlmAgent"]
