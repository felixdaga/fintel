"""The agent contract, and the failures an agent is allowed to have.

One method: given an `Environment`, produce an `AgentResponse`. A return value
rather than a `submit()` callback, so an adapter cannot finish having produced
nothing and the platform can check what came back.

An adapter never fetches data. Everything it may read is already on the
environment, behind one PIT-clamped, recorded path — which is what makes two
agents on the same strategy comparable rather than merely both plausible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fintel.environment import Environment
from fintel.models.common import Outcome
from fintel.models.decision import AgentResponse

# What an adapter may be handed. Declared, not guessed: a text-only agent asked
# for tools should fail before a run starts, not after it has spent money.
Channel = str  # "tools" | "pack" | "direct"


@runtime_checkable
class Agent(Protocol):
    """A decision-maker for one cell."""

    name: str
    version: str

    def decide(self, env: Environment) -> AgentResponse: ...


class AgentError(Exception):
    """A failure the platform knows how to classify.

    Subclasses carry the retry decision with them, so it is made once here
    rather than re-derived by every caller from a log line.
    """

    outcome: Outcome = "crashed"


class AgentTimeout(AgentError):
    outcome: Outcome = "timeout"


class RateLimited(AgentError):
    outcome: Outcome = "rate_limited"


class ProviderUnavailable(AgentError):
    """5xx, overloaded, or the connection dropped mid-response."""

    outcome: Outcome = "transient"


class ContextOverflow(AgentError):
    """The prompt did not fit. A configuration bug — retrying reproduces it."""

    outcome: Outcome = "context_overflow"


class SafetyRefusal(AgentError):
    """The provider declined on policy grounds.

    Deliberately not retryable. Re-rolling until the model says something else
    does not recover a lost answer, it manufactures a different one.
    """

    outcome: Outcome = "refused"


class MalformedOutput(AgentError):
    """The agent answered, unintelligibly."""

    outcome: Outcome = "parse_error"


class Abstained(AgentError):
    """The agent declined to take a position.

    An error type only because raising is the convenient way to unwind; it is a
    legitimate result and is recorded as one.
    """

    outcome: Outcome = "abstained"
