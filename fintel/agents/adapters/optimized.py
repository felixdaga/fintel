"""Fintel adapter for the standalone :mod:`fintel.agents.installed.optimized_agent`.

This is the fintel plumbing only. It does not run the LLM pipeline itself — that
lives in the platform-agnostic :class:`optimized_agent.OptimizedAgent`, which
has no fintel imports and can be hosted by any backtesting platform. The
adapter's job is the fintel coupling:

* render evidence packs from ``env.access`` via :class:`FintelEvidence`
  (PIT-clamped, role-partitioned into quantitative vs qualitative);
* feed the pipeline and relay its stage events onto the nerve so the live
  dashboard tracks ``quantitative_specialist`` / ``qualitative_specialist`` /
  ``independent_verifier`` / ``final_pm``;
* hand the PM a submit tool whose schema is pinned to the symbol being decided
  (via :mod:`fintel.agents.emit`);
* parse the PM's raw ``submit_views`` payload into fintel ``View`` objects
  (``emit.abstained`` + ``emit.parse_views``) and assemble the
  ``AgentResponse`` / ``ReasoningTrace`` / ``Usage``;
* persist the per-symbol dossier into the cell session.

Boundary (architecture §1 + §8): all reads go through ``env.access``
(``pit_enforcement='access'``); cell fan-out is the platform's.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from fintel.agents import emit
from fintel.agents.evidence import (
    QUAL_KINDS,
    QUANT_KINDS,
    EvidenceConfig,
    FintelEvidence,
    _pit_company_name,
)
from fintel.agents.installed.optimized_agent import (
    AgentResult,
    CallRecord,
)
from fintel.agents.installed.optimized_agent import (
    OptimizedAgent as OptimizedPipeline,
)
from fintel.agents.llm import OpenRouter
from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse, View
from fintel.models.trace import ReasoningTrace, TraceStep, Usage

logger = logging.getLogger(__name__)

_INT_FIELDS = {"specialist_max_tokens", "synthesis_max_tokens", "evidence_budget_chars",
               "web_structural_lookback_days", "web_update_lookback_days", "web_snippets_per_query"}
_BOOL_FIELDS = {"enable_verification"}


@dataclass
class OptimizedFintelAgent:
    """In-process host for the standalone optimized pipeline on fintel cells."""

    model: str = "xiaomi/mimo-v2.5-pro"
    temperature: float = 0.0
    # See optimized_agent.py — 6000 truncated CAT's qualitative report; 16000
    # is generous for a concise report + key-evidence block while bounding a
    # runaway specialist short of the 131k model max.
    specialist_max_tokens: int | None = 16000
    # Uncapped (None) — see optimized_agent.py. The PM writes a reasoning preamble
    # before the forced submit_views call; any cap truncates it mid-stream.
    synthesis_max_tokens: int | None = None
    enable_verification: bool = True

    evidence_budget_chars: int = 400_000
    web_structural_lookback_days: int = 30
    web_update_lookback_days: int = 7
    web_snippets_per_query: int = 5
    company_names: dict[str, str] = field(default_factory=dict)

    # Strategy-pack context (wired by simulate.cell.build_agent; optional).
    mission_text: str = ""
    output_schema_text: str = ""

    name: str = "optimized"
    version: str = "1.0.0"
    pit_enforcement: ClassVar[PitEnforcement] = "access"

    _pipeline: OptimizedPipeline | None = field(default=None, init=False, repr=False)
    _llm: OpenRouter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, str) and f.name in _INT_FIELDS:
                setattr(self, f.name, int(val))
            elif isinstance(val, str) and f.name in _BOOL_FIELDS:
                setattr(self, f.name, val.strip().lower() in ("1", "true", "yes", "on"))

    @staticmethod
    def preflight_checks(**params: Any) -> list[str]:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return ["OPENROUTER_API_KEY is not set; the optimized agent cannot call OpenRouter"]
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

    def _pipeline_for(self, env: Environment) -> OptimizedPipeline:
        """Build the standalone pipeline wired to this cell's nerve."""
        nerve = env.nerve
        cell_name = env.cell.name
        decision_date = env.cell.decision_date.isoformat()

        def _on_stage(stage: str, symbol: str) -> None:
            _emit(nerve, "agent_stage", cell=cell_name, decision_date=decision_date,
                  stage=stage, symbol=symbol)

        return OptimizedPipeline(
            model=self.model,
            temperature=self.temperature,
            specialist_max_tokens=self.specialist_max_tokens,
            synthesis_max_tokens=self.synthesis_max_tokens,
            enable_verification=self.enable_verification,
            mission_text=self.mission_text,
            output_schema_text=self.output_schema_text,
            llm=self._client(),
            on_stage=_on_stage,
        )

    def decide(self, env: Environment) -> AgentResponse:
        symbols = tuple(sorted(env.policy.decidable))
        if not symbols:
            return AgentResponse(views={}, outcome="empty", detail="no decidable symbols")

        decidable = frozenset(env.policy.decidable)
        views: dict[str, View] = {}
        steps: list[TraceStep] = []
        dossiers: dict[str, dict] = {}
        errors: list[str] = []
        usage = Usage()

        nerve = env.nerve
        cell_name = env.cell.name
        decision_date = env.cell.decision_date.isoformat()
        pipeline = self._pipeline_for(env)

        for sym in symbols:
            _emit(nerve, "agent_stage", cell=cell_name, decision_date=decision_date,
                  stage="evidence", symbol=sym)

            builder = FintelEvidence(
                access=env.access,
                symbol=sym,
                decision_date=env.cell.decision_date,
                config=self._evidence_config(),
                company_name=_pit_company_name(
                    getattr(env, "market_config", None), sym, env.cell.decision_date
                ),
            )
            with _stage(nerve, "evidence_quantitative", cell_name, decision_date, sym):
                quant = builder.quantitative_block()
            with _stage(nerve, "evidence_qualitative", cell_name, decision_date, sym):
                qual = builder.qualitative_block()

            # Which kinds are actually bound for this cell, intersected with
            # the agent's quant/qual partition. Passed to the pipeline so the
            # specialist user prompt declares what's in the pack rather than
            # hardcoding a specific strategy's bindings.
            bound = set(env.access.kinds)
            quant_kinds = tuple(k for k in QUANT_KINDS if k in bound)
            qual_kinds = tuple(k for k in QUAL_KINDS if k in bound)

            submit_tool = {
                "type": "function",
                "function": {
                    "name": emit.SUBMIT_TOOL,
                    "description": emit.submit_description((sym,)),
                    "parameters": emit.submit_schema((sym,)),
                },
            }

            result = pipeline.decide_one(
                symbol=sym, trade_date=decision_date,
                quant_evidence=quant, qual_evidence=qual,
                submit_tool=submit_tool,
                quant_kinds=quant_kinds,
                qual_kinds=qual_kinds,
            )

            for idx, c in enumerate(result.calls):
                steps.append(_trace_step(sym, idx, c))
                usage = usage.merge(_usage(c))

            dossier = _dossier(sym, result)
            self._persist_dossier(env, sym, dossier)
            dossiers[sym] = dossier

            view, error = self._resolve(result, decidable)
            if error:
                errors.append(f"{sym}: {error}")
                logger.warning("optimized native: %s failed — %s", sym, error)
            elif view is not None:
                views[sym] = view

        outcome: Outcome
        detail = ""
        if views:
            outcome = "ok"
        elif errors:
            # All symbols failed to produce a view. These are PM output
            # failures (didn't call submit_views, no view for symbol,
            # malformed payload) — model behavior a retry may fix, not a
            # code bug. "parse_error" is retryable; "crashed" is not.
            outcome = "parse_error"
            detail = "; ".join(errors)[:2000]
        else:
            outcome = "abstained"
            detail = "pipeline produced no views (PM abstained or empty submit)"

        steps.append(
            TraceStep(
                step_id=f"optimized-summary-{cell_name}",
                kind="other",
                started_at=datetime.now(UTC),
                model=self.model,
                payload={
                    "n_views": len(views),
                    "n_symbols": len(symbols),
                    "pipeline": (
                        "quant‖qual → verify → final_pm"
                        if self.enable_verification
                        else "quant‖qual → final_pm"
                    ),
                    "errors": errors,
                },
            )
        )

        return AgentResponse(
            views=views,
            outcome=outcome,
            detail=detail,
            usage=usage,
            trace=ReasoningTrace(
                final_explanation=(
                    f"OptimizedAgent (native): {len(views)}/{len(symbols)} views on "
                    f"{env.cell.decision_date} via {self.model}; "
                    + (
                        "quant/qual → structured-verify → final-PM."
                        if self.enable_verification
                        else "quant/qual → final-PM (verification off)."
                    )
                ),
                steps=steps,
                metadata={
                    "agent": self.name,
                    "version": self.version,
                    "model": self.model,
                    "enable_verification": self.enable_verification,
                    "errors": errors,
                    "dossiers": dossiers,
                    "mission_digest_present": bool(self.mission_text.strip()),
                },
            ),
        )

    def _resolve(self, result: AgentResult, decidable: frozenset[str]) -> tuple[View | None, str | None]:
        """Map the pipeline's raw submit_args into a fintel View (or an error)."""
        if result.error:
            return None, result.error
        if result.submit_args is None:
            return None, f"PM did not call {emit.SUBMIT_TOOL}"
        reason = emit.abstained(result.submit_args)
        if reason:
            return None, None  # abstention is not an error; no view
        views, notes = emit.parse_views(result.submit_args, decidable=decidable)
        view = views.get(result.symbol)
        if view is None:
            return None, f"PM submitted no view for {result.symbol}"
        return view, None

    def _persist_dossier(self, env: Environment, symbol: str, dossier: dict) -> None:
        session = getattr(env, "session", None)
        if session is None or not getattr(session, "path", None):
            return
        try:
            path = Path(session.path)
            path.mkdir(parents=True, exist_ok=True)
            payload = {
                "symbol": symbol,
                "decision_date": env.cell.decision_date.isoformat(),
                "cell": env.cell.name,
                "agent": self.name,
                "version": self.version,
                "model": self.model,
                **dossier,
            }
            (path / "dossier.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            logger.debug("failed to persist dossier for %s", symbol, exc_info=True)


def _dossier(sym: str, result: AgentResult) -> dict[str, Any]:
    return {
        "ticker": sym,
        "pipeline": "quant‖qual → verify → final_pm",
        "quantitative_report": result.quant_report,
        "qualitative_report": result.qual_report,
        "verification": result.verification,
        "verification_flag": result.verification_flag,
        "pm_emit_path": "submit_views",
        "final_decision": result.decision_md[:5000],
        "submit_args": result.submit_args,
        "error": result.error,
    }


def _trace_step(sym: str, index: int, c: CallRecord) -> TraceStep:
    return TraceStep(
        step_id=f"optimized-llm-{sym}-{index}",
        kind="llm_call",
        started_at=c.started_at,
        duration_ms=c.duration_ms,
        model=c.model or "",
        tokens_in=c.tokens_in or None,
        tokens_out=c.tokens_out or None,
        cost_usd=c.cost_usd,
        payload={
            "ticker": sym,
            "stage": c.stage,
            "finish_reason": c.finish_reason,
            "tool": c.tool_name,
        },
    )


def _usage(c: CallRecord) -> Usage:
    return Usage(
        n_llm_calls=1,
        tokens_in=c.tokens_in,
        tokens_out=c.tokens_out,
        cost_usd=c.cost_usd,
        basis="reported" if c.cost_usd is not None else "unknown",
    )


def _emit(nerve: Any, kind: str, **fields: Any) -> None:
    if nerve is None or not hasattr(nerve, "emit"):
        return
    try:
        nerve.emit(kind, **fields)
    except Exception:  # noqa: BLE001
        pass


class _stage:
    """Context manager that emits an `agent_stage` nerve event on enter
    (evidence-fetching stages only; pipeline stages come from the
    pipeline's on_stage hook)."""

    def __init__(self, nerve: Any, name: str, cell: str, decision_date: str, symbol: str) -> None:
        self.nerve = nerve
        self.name = name
        self.cell = cell
        self.decision_date = decision_date
        self.symbol = symbol

    def __enter__(self) -> None:
        _emit(self.nerve, "agent_stage", cell=self.cell, decision_date=self.decision_date,
              stage=self.name, symbol=self.symbol)

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = ["OptimizedFintelAgent"]
