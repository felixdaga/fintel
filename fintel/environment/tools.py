"""The agent's tool surface, generated from the run's bindings. Transport-free.

The old server registered twenty tools unconditionally at import, then filtered
them per decision with allow/deny lists. So the tool list was a hand-maintained
constant that had to be kept in agreement with the catalog, the bundle builder
and the prompt text — and it drifted: a tool's docstring promised 300-character
news summaries the code never truncated, another advertised a window its filter
didn't apply, and one accepted an `as_of` argument no tool exposed.

Here the surface is derived. A run that binds three kinds gets three tools, each
described by the catalog entry of the source actually bound. There is nothing to
keep in sync, and a kind that isn't bound is not merely denied — it doesn't exist
to the agent, which is a much clearer signal than a tool that always errors.

No MCP or transport dependency: `descriptors()` yields schemas and `call()`
dispatches. Binding those to a protocol belongs with the agent adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fintel.environment.access import DataAccess
from fintel.market import catalog

# JSON Schema types for the catalog's dtypes.
_JSON_TYPE = {
    "number": "integer",
    "text": "string",
    "date": "string",
    "bool": "boolean",
    "list": "array",
}

_SUBJECT_DOC = {
    "symbol": "Ticker to look up. Must be within this cell's universe.",
    "query": "Search text.",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str
    description: str
    schema: dict

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.schema.get("required", ()))


def tool_name(kind: str) -> str:
    """`prices` → `get_prices`; `web_search` stays itself.

    A kind that already reads as a verb keeps its name, since `get_web_search`
    describes nothing.
    """
    return kind if kind.endswith("_search") else f"get_{kind}"


def describe(info: catalog.SourceInfo, cutoff_note: str) -> str:
    """Tool help built from the catalog, so it cannot contradict the source."""
    lines = [info.description.strip() or f"Return {info.kind} data.", "", cutoff_note]
    if info.fields:
        shown = ", ".join(f.name for f in info.fields[:14])
        more = "" if len(info.fields) <= 14 else f", and {len(info.fields) - 14} more"
        lines += ["", f"Fields: {shown}{more}."]
    lines += [
        "",
        "Returns {status, data}. status='empty' means there genuinely is none; "
        "status='failed' means the lookup broke and the absence means nothing.",
    ]
    return "\n".join(lines)


def spec_for(
    kind: str,
    source_name: str,
    *,
    decision_date: str,
    lookback_default: int | None = None,
) -> ToolSpec:
    info = catalog.source(source_name)
    properties: dict[str, dict] = {}
    required: list[str] = []

    if info.subject != "none":
        properties[info.subject] = {
            "type": "string",
            "description": _SUBJECT_DOC.get(info.subject, ""),
        }
        required.append(info.subject)

    for param in info.call_params:
        entry: dict[str, Any] = {
            "type": _JSON_TYPE.get(param.dtype, "string"),
            "description": param.description or f"{param.name} for this lookup.",
        }
        # The strategy's lookback (baked into the source) is the default the agent
        # sees — one knob, not a catalog default that disagrees with the binding.
        if param.name == "lookback_days" and lookback_default is not None:
            entry["default"] = lookback_default
        elif param.default is not None:
            entry["default"] = param.default
        properties[param.name] = entry

    cutoff_note = (
        f"Point-in-time: only information available strictly before {decision_date} "
        f"is returned. The boundary is enforced and cannot be passed, widened, "
        f"or bypassed."
    )
    return ToolSpec(
        name=tool_name(kind),
        kind=kind,
        description=describe(info, cutoff_note),
        schema={"type": "object", "properties": properties, "required": required},
    )


@dataclass
class ToolSurface:
    """The tools this cell has, and the dispatcher behind them.

    Deliberately transport-free: `descriptors()` yields schemas and `call()`
    dispatches, so MCP, an LLM's native function-calling, and an in-process
    framework wrapper are all thin adapters over the same surface rather than
    three implementations of it. The old repo had exactly that duplication — an
    MCP server and a LangChain `Toolkit` reimplementing the same tools over the
    same session, each with its own PIT filters to keep in agreement.
    """

    access: DataAccess
    bound: dict[str, str]  # kind -> source name
    only: tuple[str, ...] | None = None  # restrict to these kinds

    def descriptors(self) -> tuple[ToolSpec, ...]:
        decision_date = self.access.cell.decision_date.isoformat()
        out = []
        for kind in self.access.kinds:
            if self.only is not None and kind not in self.only:
                continue
            source_name = self.bound.get(kind)
            # A package-supplied source has no catalog entry to describe; it is
            # still readable, just not advertised as a typed tool.
            if source_name and catalog.has_source(source_name):
                src = self.access.sources.get(kind)
                lb = getattr(src, "lookback_days", None) if src is not None else None
                if lb is None and src is not None:
                    spec = getattr(src, "spec", None)
                    if spec is not None and hasattr(spec, "lookback_days"):
                        lb = spec.lookback_days
                out.append(
                    spec_for(
                        kind,
                        source_name,
                        decision_date=decision_date,
                        lookback_default=lb,
                    )
                )
        return tuple(out)

    def subset(self, kinds: tuple[str, ...]) -> ToolSurface:
        """A narrower surface over the same access, for one role.

        A multi-role desk gives its fundamentals analyst prices and filings and
        its narrative analyst news and filing text. Both still read through the
        one clamped, recorded path, so a role boundary can't become a second
        data path with its own PIT rules.
        """
        unknown = sorted(set(kinds) - set(self.access.kinds))
        if unknown:
            raise ValueError(
                f"cannot build a tool subset for {unknown}: not available to this cell "
                f"(has {list(self.access.kinds)})"
            )
        return ToolSurface(access=self.access, bound=self.bound, only=tuple(kinds))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.descriptors())

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Dispatch by tool name. Always returns a payload, never raises.

        Schema-level refusals are recorded as ``denied`` reads so the access
        log (and post-run health) sees them — not only successful fetches.
        """
        specs = {spec.name: spec for spec in self.descriptors()}
        spec = specs.get(name)
        args = dict(arguments or {})
        if spec is None:
            return self.access.deny(
                kind=name,
                query=args,
                detail=(
                    f"no tool named {name!r} in this run; available: {sorted(specs)}. "
                    f"Data kinds are declared by the strategy."
                ),
            ).payload()
        missing = [key for key in spec.required if not args.get(key)]
        if missing:
            return self.access.deny(
                kind=spec.kind,
                query=args,
                detail=f"{name} requires {missing}; got {sorted(args)}",
            ).payload()
        unknown = sorted(set(args) - set(spec.schema["properties"]))
        if unknown:
            return self.access.deny(
                kind=spec.kind,
                query=args,
                detail=(
                    f"{name} does not accept {unknown}; accepted: "
                    f"{sorted(spec.schema['properties'])}"
                ),
            ).payload()
        return self.access.read(spec.kind, **args).payload()
