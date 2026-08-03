"""The `Progress` protocol + a null sink.

Owned by the environment module because the environment owns the emit
surface (`Nerve`). `simulate/` and `agents/` import `Progress`/`NullProgress`
from here — they are producers, not owners. Keeping the protocol in the
environment layer keeps the import ladder one-way (environment -> simulate is
legal; the reverse is not).

`NullProgress` is the test/dry-run sink: a run that should be silent passes it
(or passes nothing and lets `run_job` construct a `Nerve`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Progress(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...


@dataclass
class NullProgress:
    """No-op sink — the default when a test wants a run to be silent.

    The production path constructs a `Nerve` when no `Progress` is passed to
    `run_job`; tests pass `NullProgress()` explicitly to keep runs quiet and
    log-file-free.
    """

    def emit(self, event: str, **fields: Any) -> None:
        return None

