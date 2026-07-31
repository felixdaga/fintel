"""The one place an agent is called.

`invoke` never raises. A cell that fails is a recorded outcome, not a lost run:
the old platform let an adapter exception abort the whole date, so one bad
subprocess erased every other symbol's work for that day.

It is also the only place an exception becomes an `Outcome`, so "was that a
refusal or a crash" has one answer rather than one per adapter.
"""

from __future__ import annotations

import logging
import time

from fintel.agents.base import Agent, AgentError
from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse

logger = logging.getLogger(__name__)


def classify(exc: BaseException) -> tuple[Outcome, str]:
    """Map an exception to an outcome. Unknown failures are crashes, never
    silent empties — an unrecognised error is a bug to fix, not a data point."""
    if isinstance(exc, AgentError):
        return exc.outcome, str(exc) or exc.__class__.__name__
    if isinstance(exc, TimeoutError):
        return "timeout", str(exc) or "timed out"
    return "crashed", f"{exc.__class__.__name__}: {exc}"


def invoke(agent: Agent, env: Environment) -> AgentResponse:
    started = time.perf_counter()
    try:
        response = agent.decide(env)
    except BaseException as exc:  # noqa: BLE001 - deliberately total
        outcome, detail = classify(exc)
        # A KeyboardInterrupt must still stop the run, but only after the cell's
        # failure is on record.
        logger.warning("%s on %s: %s — %s", agent.name, env.cell.name, outcome, detail)
        env.log.append("agent_failed", outcome=outcome, detail=detail)
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        return AgentResponse(views={}, outcome=outcome, detail=detail)
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        env.log.append("agent_returned", agent=agent.name, elapsed_ms=round(elapsed, 1))

    return _check(agent, env, response)


def _check(agent: Agent, env: Environment, response: AgentResponse) -> AgentResponse:
    """Hold the adapter to the cell's terms.

    Enforced here rather than trusted per adapter: a view on a symbol this cell
    may not decide is dropped, because the alternative is a strategy silently
    scoring an opinion it never asked for.
    """
    allowed = set(env.policy.decidable)
    stray = sorted(set(response.views) - allowed)
    if stray:
        logger.warning(
            "%s returned view(s) on %s, which %s may not decide — dropped",
            agent.name,
            stray,
            env.cell.name,
        )
        kept = {s: v for s, v in response.views.items() if s in allowed}
        response = response.model_copy(
            update={
                "views": kept,
                "outcome": response.outcome if kept else "empty",
                "detail": response.detail or f"dropped out-of-scope views: {stray}",
            }
        )
    env.log.submitted(n_views=len(response.views), dropped=stray)
    return response
