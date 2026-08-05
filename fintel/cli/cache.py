"""``fintel cache status`` — what's in the central cache, gap-aware.

Reads the coverage sidecars that the stores already maintain and prints, per
kind and symbol, the cached date intervals — and the gaps inside a requested
window. Read-only; never fetches.
"""

from __future__ import annotations

import os
from argparse import Namespace
from datetime import date as Date
from pathlib import Path

from fintel.market import catalog
from fintel.market.cache_view import coverage_for_kind


def _resolve_cache_root(args: Namespace) -> Path:
    if args.cache_root:
        return Path(args.cache_root).expanduser()
    env = os.environ.get("FINTEL_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(args.output_root).expanduser() / "cache"


def run_cache(args: Namespace) -> int:
    cache_root = _resolve_cache_root(args)

    if args.cache_command == "status":
        return _status(args, cache_root)
    return 2


def _status(args: Namespace, cache_root: Path) -> int:
    catalog.register_builtins()

    if args.source:
        sources = [args.source]
    else:
        sources = sorted(s.name for s in catalog.sources())

    if not cache_root.is_dir():
        print(f"(no cache at {cache_root})")
        return 0

    print(f"cache root: {cache_root}")
    print()

    any_data = False
    for name in sources:
        if not catalog.has_source(name):
            continue
        info = catalog.source(name)
        if info.is_computed:
            # No own cache; upstream kinds hold the data.
            print(
                f"{info.kind} ({name}) — computed from "
                f"{list(info.derives_from)}; see upstream kinds."
            )
            continue
        cov = coverage_for_kind(
            kind=info.kind, source_name=name, cache_root=cache_root, symbol=args.symbol
        )
        if not cov.symbols:
            continue
        any_data = True
        print(f"{info.kind} ({name})  @ {cov.cache_dir}")
        for sym, spans in sorted(cov.symbols.items()):
            if not spans:
                continue
            rendered = "  ".join(f"[{a} .. {b}]" for a, b in spans)
            line = f"  {sym:<6} {rendered}"
            # Show gaps inside the requested window, if given.
            if args.window:
                wf, wt = _parse_window(args.window)
                gaps = cov.gaps(sym, wf, wt)
                if gaps:
                    line += "   (gaps: " + ", ".join(f"{a}..{b}" for a, b in gaps) + ")"
            print(line)
        print()

    if not any_data:
        print("(cache empty)")

    return 0


def _parse_window(window: str) -> tuple[Date, Date]:
    if ":" in window:
        lo, hi = window.split(":", 1)
    elif ".." in window:
        lo, hi = window.split("..", 1)
    else:
        raise SystemExit(f"window must be FROM..TO or FROM:TO (ISO dates), got {window!r}")
    return Date.fromisoformat(lo.strip()), Date.fromisoformat(hi.strip())
