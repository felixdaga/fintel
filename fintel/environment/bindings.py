"""Extract strategy-owned params from live sources for bindings.json.

Lives in the environment layer so the subprocess adapter (agents/) can persist
bindings without reaching down into market/ (forbidden by the layer ladder).
The catalog is the single source of truth for what params a source accepts;
this helper iterates it and reads the values off the live source instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fintel.market import catalog


def extract_bindings(
    bound: dict[str, str],
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the ``bindings`` array for ``bindings.json``.

    ``bound`` is ``{kind: source_name}`` (the tool surface). ``sources`` is the
    live source dict from ``DataAccess``. For each binding, persist ``kind``,
    ``source``, and every catalog-declared non-per-call param the live source
    instance carries — so the MCP rebuild constructs the same sources, not
    catalog defaults (the cause of the "Tool not found" failure class).
    """
    out: list[dict[str, Any]] = []
    for kind, source_name in bound.items():
        entry: dict[str, Any] = {"kind": kind, "source": source_name}
        src = sources.get(kind)
        if src is not None and catalog.has_source(source_name):
            for param in catalog.source(source_name).params:
                # Persist every catalog-declared param (per_call or not): the
                # strategy's binding sets the default, and the MCP rebuild needs
                # that default. per_call only means the agent may also override
                # it at call time — the default still travels through bindings.
                if not hasattr(src, param.name):
                    continue
                val = getattr(src, param.name)
                if val is None:
                    continue
                if isinstance(val, Path):
                    val = str(val)
                elif isinstance(val, tuple):
                    val = list(val)
                entry[param.name] = val
        out.append(entry)
    return out
