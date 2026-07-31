"""What the agent is allowed to ask for. The constraints half of the environment.

The old server gated the *universe* only at submission time: "data tools accept
any symbol". So an agent scored on the Dow could read anything in the cache,
including names that had already left the index. That silently changes the
experiment, and nothing in the run record showed it happened.

Here the universe is enforced on reads, and widening it is a decision a strategy
has to make out loud via `peers`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fintel.models.common import Symbol


class AccessDenied(Exception):
    """The agent asked for something outside its allowance.

    Denial is never silent and never rendered as absence: an agent that reads a
    forbidden symbol must learn it was refused, not conclude the company has no
    history.
    """


@dataclass(frozen=True)
class AccessPolicy:
    """Derived from the strategy's bindings and the universe at the decision date.

    `decidable` is what the cell may submit views on. `readable` is what it may
    fetch data about — the same set unless the strategy opted into peers.
    """

    kinds: frozenset[str]
    decidable: frozenset[Symbol]
    peers: frozenset[Symbol] = frozenset()
    max_lookback_days: int = 3650
    max_results: int = 500

    @property
    def readable(self) -> frozenset[Symbol]:
        return self.decidable | self.peers

    def check_kind(self, kind: str) -> None:
        if kind not in self.kinds:
            raise AccessDenied(
                f"kind {kind!r} is not available to this run; declared kinds: "
                f"{sorted(self.kinds)}. Add a [[data]] block to the strategy to enable it."
            )

    def check_symbol(self, symbol: Symbol) -> None:
        if symbol not in self.readable:
            hint = (
                f"readable symbols: {sorted(self.readable)}"
                if len(self.readable) <= 12
                else f"{len(self.readable)} readable symbols"
            )
            raise AccessDenied(f"symbol {symbol!r} is outside this cell's universe; {hint}")

    def check_decidable(self, symbol: Symbol) -> None:
        if symbol not in self.decidable:
            raise AccessDenied(
                f"symbol {symbol!r} is readable but not decidable in this cell; "
                f"views may only be submitted for {sorted(self.decidable)}"
            )

    def clamp_query(self, query: dict) -> dict:
        """Bound the size of a request without failing it.

        A lookback wider than the cap is trimmed rather than refused: the agent
        asked a legitimate question slightly too greedily, and an error here
        would cost a whole cell. Refusal is reserved for the boundaries that
        actually change the experiment — kinds and symbols.
        """
        out = dict(query)
        caps = (("lookback_days", self.max_lookback_days), ("max_results", self.max_results))
        for key, cap in caps:
            value = out.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = max(1, min(int(value), cap))
        return out


@dataclass
class PolicyBuilder:
    """Assembles a policy from the three inputs the environment brings together."""

    kinds: tuple[str, ...] = ()
    decidable: tuple[Symbol, ...] = ()
    peers: tuple[Symbol, ...] = ()
    limits: dict = field(default_factory=dict)

    def build(self) -> AccessPolicy:
        return AccessPolicy(
            kinds=frozenset(self.kinds),
            decidable=frozenset(self.decidable),
            peers=frozenset(self.peers) - frozenset(self.decidable),
            **{
                k: v
                for k, v in self.limits.items()
                if k in {"max_lookback_days", "max_results"}
            },
        )
