#!/usr/bin/env python3
"""Remove PIT-leaked Brave articles from raw ``web_search`` cache files.

Brave's ``freshness`` window is soft — cached payloads often retain results whose
``sources[url].age`` is outside ``[since, through]``. Runtime
``clamp_by_age`` drops them in memory; this script rewrites the on-disk blobs
so the raw cache matches what agents actually see.

Usage:
  python scripts/scrub_web_search_leaks.py [cache_root ...]
  # default: runs/cache cache
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as Date
from pathlib import Path

from fintel.market.data.store import atomic_write
from fintel.market.data.web import clamp_brave_by_age, parse_brave_age


def _window(path: Path, blob: dict) -> tuple[Date, Date] | None:
    """Prefer filename ``{through}_{since}_{hash}.json``, else search_window."""
    stem = path.stem  # through_since_hash
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            through = Date.fromisoformat(parts[0])
            since = Date.fromisoformat(parts[1])
            return since, through
        except ValueError:
            pass
    window = blob.get("search_window") or {}
    try:
        since = Date.fromisoformat(str(window["from"])[:10])
        through = Date.fromisoformat(str(window["to"])[:10])
    except (KeyError, TypeError, ValueError):
        return None
    return since, through


def _count_leaks(payload: dict, since: Date, through: Date) -> tuple[int, int]:
    """Return (n_generic, n_leaked) for a Brave sources payload."""
    grounding = payload.get("grounding") if isinstance(payload, dict) else None
    meta = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(grounding, dict) or not isinstance(meta, dict):
        return 0, 0
    generic = grounding.get("generic")
    if not isinstance(generic, list):
        return 0, 0
    leaked = 0
    for item in generic:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        entry = meta.get(url) if isinstance(url, str) else None
        age = entry.get("age") if isinstance(entry, dict) else None
        ymd = parse_brave_age(age)
        if ymd is not None and not (since <= ymd <= through):
            leaked += 1
    return len(generic), leaked


def scrub_file(path: Path, *, dry_run: bool) -> tuple[int, int] | None:
    """Scrub one cache file. Returns (before, after) generic counts, or None if skipped."""
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skip corrupt {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(blob, dict):
        return None
    window = _window(path, blob)
    if window is None:
        print(f"skip no window {path}", file=sys.stderr)
        return None
    since, through = window
    payload = blob.get("sources")
    if not isinstance(payload, dict):
        return None
    before, n_leaked = _count_leaks(payload, since, through)
    if n_leaked == 0:
        return before, before
    clamped = clamp_brave_by_age(payload, since, through)
    after = len((clamped.get("grounding") or {}).get("generic") or [])
    if dry_run:
        return before, after
    out = dict(blob)
    out["sources"] = clamped
    # Keep search_window aligned with the filename / clamp window.
    out["search_window"] = {"from": since.isoformat(), "to": through.isoformat()}
    atomic_write(path, json.dumps(out, indent=2) + "\n")
    return before, after


def scrub_root(root: Path, *, dry_run: bool) -> dict[str, int]:
    web = root / "web_search" if root.name != "web_search" else root
    if not web.is_dir():
        return {"files": 0, "scrubbed": 0, "dropped": 0, "clean": 0, "skipped": 0}
    stats = {"files": 0, "scrubbed": 0, "dropped": 0, "clean": 0, "skipped": 0}
    for path in sorted(web.glob("*.json")):
        if path.name.endswith(".meta.json") or path.name.endswith(".lock"):
            continue
        # Skip sidecar-style names if any
        if ".meta." in path.name:
            continue
        stats["files"] += 1
        result = scrub_file(path, dry_run=dry_run)
        if result is None:
            stats["skipped"] += 1
            continue
        before, after = result
        if after < before:
            stats["scrubbed"] += 1
            stats["dropped"] += before - after
            print(f"{'dry-run ' if dry_run else ''}{path}: {before} -> {after} (-{before - after})")
        else:
            stats["clean"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("runs/cache"), Path("cache")],
        help="Cache roots (or …/web_search). Default: runs/cache cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report leaks without rewriting files",
    )
    args = parser.parse_args(argv)

    total = {"files": 0, "scrubbed": 0, "dropped": 0, "clean": 0, "skipped": 0}
    for root in args.roots:
        stats = scrub_root(root, dry_run=args.dry_run)
        print(f"\n{root}: {stats}")
        for k, v in stats.items():
            total[k] += v
    print(f"\ntotal: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
