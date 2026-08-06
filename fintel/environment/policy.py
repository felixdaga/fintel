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
    # Per-kind lookback cap (from the strategy binding). A caller may request
    # less, never more. `max_lookback_days` is the fallback for kinds without a
    # binding-declared lookback (e.g. a custom `module:Callable` source).
    lookback_caps: frozenset[tuple[str, int]] = frozenset()
    # Per-kind render caps carried from the binding/catalog (e.g.
    # web_search.snippet_max_chars, news.summary_max_chars). These bound
    # how the evidence pack is rendered to the agent, not what is fetched.
    render_caps: frozenset[tuple[str, tuple[tuple[str, int], ...]]] = frozenset()
    max_lookback_days: int = 3650
    max_results: int = 500

    @property
    def readable(self) -> frozenset[Symbol]:
        return self.decidable | self.peers

    @property
    def lookback_cap_map(self) -> dict[str, int]:
        return dict(self.lookback_caps)

    @property
    def render_cap_map(self) -> dict[str, dict[str, int]]:
        return {kind: dict(caps) for kind, caps in self.render_caps}

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

    def clamp_query(self, kind: str, query: dict) -> dict:
        """Bound the size of a request without failing it.

        A lookback wider than the cap is trimmed rather than refused: the agent
        asked a legitimate question slightly too greedily, and an error here
        would cost a whole cell. Refusal is reserved for the boundaries that
        actually change the experiment — kinds and symbols. The lookback cap
        is the strategy's `lookback_days` for that kind, so a caller cannot
        exceed the range the prefetch warmed.
        """
        out = dict(query)
        cap = self.lookback_cap_map.get(kind, self.max_lookback_days)
        lb = out.get("lookback_days")
        if isinstance(lb, (int, float)) and not isinstance(lb, bool):
            out["lookback_days"] = max(1, min(int(lb), cap))
        mr = out.get("max_results")
        if isinstance(mr, (int, float)) and not isinstance(mr, bool):
            out["max_results"] = max(1, min(int(mr), self.max_results))
        return out


@dataclass
class PolicyBuilder:
    """Assembles a policy from the three inputs the environment brings together."""

    kinds: tuple[str, ...] = ()
    decidable: tuple[Symbol, ...] = ()
    peers: tuple[Symbol, ...] = ()
    lookback_caps: dict = field(default_factory=dict)
    render_caps: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def build(self) -> AccessPolicy:
        caps = dict(self.lookback_caps)
        # `limits` may also supply lookback_caps/max_lookback_days for tests.
        caps.update(self.limits.get("lookback_caps", {}))
        rcaps = {kind: dict(c) for kind, c in self.render_caps.items()}
        rcaps.update(self.limits.get("render_caps", {}))
        return AccessPolicy(
            kinds=frozenset(self.kinds),
            decidable=frozenset(self.decidable),
            peers=frozenset(self.peers) - frozenset(self.decidable),
            lookback_caps=frozenset(caps.items()),
            render_caps=frozenset((kind, tuple(c.items())) for kind, c in rcaps.items()),
            **{k: v for k, v in self.limits.items() if k in {"max_lookback_days", "max_results"}},
        )
