from __future__ import annotations

from typing import Literal

Symbol = str

# The agent's thesis horizon, decoupled from the platform's rebalance cadence.
# Left open: a package's output_schema.json owns its own vocabulary.
TimeHorizon = str

DecisionScope = Literal["single_name", "portfolio"]
Status = Literal["ok", "partial", "empty", "failed", "timeout", "skipped"]

# Harness / environment grade — orthogonal to agent outcome and Status.
# A cell can be Status=ok (agent returned views) while HealthStatus=broken
# (every tool call failed). Scoring should treat broken as non-comparable.
HealthStatus = Literal["ok", "degraded", "broken"]

PORTFOLIO_CELL = "__portfolio__"

# How one agent invocation ended. Zero views is not one state but several, and
# collapsing them loses the difference between an agent that declined and an
# agent that fell over — which is what made the old platform retry blindly.
Outcome = Literal[
    "ok",  # produced at least one view
    "abstained",  # declined on purpose: a real answer, and often the right one
    "refused",  # the provider blocked the request on policy grounds
    "empty",  # returned nothing and gave no reason
    "timeout",
    "rate_limited",
    "transient",  # provider 5xx, overload, dropped connection
    "context_overflow",  # the prompt did not fit
    "parse_error",  # answered, unintelligibly
    "crashed",
]

# Retrying anything outside this set is wrong, not merely wasteful.
# `refused` and `abstained` are genuine results — re-rolling them until the
# agent says something else manufactures a different answer. `context_overflow`
# and `crashed` are bugs in the run's configuration or code, so a retry burns
# money to reproduce them.
RETRYABLE: frozenset[str] = frozenset(
    {"empty", "timeout", "rate_limited", "transient", "parse_error"}
)
