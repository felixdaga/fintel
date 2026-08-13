"""Tail the agent's own session transcript live and capture tool errors.

The agent sees tool results (including ``MCP error -32001``) that our MCP
server never observes — the server executes the read, but the result may
not round-trip over stdio. This module tails the CLI's own JSONL transcript
in real time so tool errors are:

  * emitted as progress events (visible in the terminal as they happen);
  * logged to ``access.jsonl`` as ``tool_error`` events (health = broken);
  * copied into the cell's session dir for post-run audit.

If ``fail_fast_errors`` consecutive tool results are errors, the catcher
sets a flag so the caller can kill the subprocess instead of burning the
full timeout.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fintel.environment.progress import Progress
from fintel.models.trace import Usage

logger = logging.getLogger(__name__)

# Patterns that mark a tool result as a harness error (not the agent's judgment).
_ERROR_MARKERS = (
    "mcp error",
    "request timed out",
    "connection closed",
    "not connected",
    'status": "error',
    '"iserror": true',
    "econnreset",
    "pipe broken",
)


class _TranscriptCatcher:
    """Tails one JSONL transcript file in a background thread."""

    def __init__(
        self,
        *,
        transcript_path: Path | None,
        access_log: Any,
        fail_fast_errors: int = 3,
        nerve: Progress | None = None,
        cell: str = "",
        decision_date: str = "",
    ) -> None:
        self.transcript_path = transcript_path
        self.access_log = access_log
        self.nerve = nerve
        self.cell = cell
        # Carried on every staging emit so a multi-date run's per-cell events
        # can be told apart (NVDA@Jan vs NVDA@Apr) — the dashboard keys cells
        # by (cell, decision_date) and shows the date.
        self.decision_date = decision_date
        self.fail_fast_errors = max(0, fail_fast_errors)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []
        self.tool_errors: list[dict] = []
        self._consecutive_errors = 0
        self.fail_fast_triggered = False
        self._round = 0
        # Usage rollup accumulated from each assistant turn's `message.usage`.
        # OpenClaw writes real token counts per turn (input/cacheRead/output) and
        # a `cost` block the provider surfaced (often $0 / not surfaced). The
        # adapter's job is to feed these into the platform's nerve + AgentResponse
        # so a run's cost is tracked, not silently zeroed.
        self._tok_in = 0
        self._tok_out = 0
        self._n_llm_calls = 0
        self._cost_usd: float | None = None
        self._cost_reported = False

    def start(self) -> None:
        if self.transcript_path is None:
            return
        self._thread = threading.Thread(target=self._tail, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def finalize(self, session_path: Path) -> None:
        """Copy the transcript into the cell dir and log any tool errors."""
        if self.transcript_path is None or not self.transcript_path.is_file():
            return
        dest = session_path / "agent_transcript.jsonl"
        try:
            shutil.copy2(self.transcript_path, dest)
        except OSError as exc:
            logger.warning("could not copy transcript to %s: %s", dest, exc)

        if self.tool_errors:
            self.access_log.append(
                "tool_errors",
                n=len(self.tool_errors),
                errors=[
                    {
                        "tool": e.get("tool"),
                        "error": str(e.get("error", ""))[:200],
                    }
                    for e in self.tool_errors[:20]
                ],
            )

    def _tail(self) -> None:
        """Poll the transcript file, parse each new line, emit progress."""
        path = self.transcript_path
        if path is None:
            return

        # Wait for the file to appear (the CLI creates it on first turn).
        # Under high concurrency (shared_concurrency), MCP attach alone can
        # take minutes — a 30s wait bails before the transcript ever lands,
        # and the cell looks silent to the nerve (false "stalled"). Wait the
        # full cell lifetime instead; the subprocess timeout is the real
        # backstop for a cell that never starts.
        deadline = time.monotonic() + 600.0
        while not path.is_file():
            if self._stop.is_set() or time.monotonic() > deadline:
                return
            time.sleep(0.2)

        with path.open("r") as handle:
            while not self._stop.is_set():
                line = handle.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                self._process_line(line.strip())

    def _accumulate_usage(self, raw: Any) -> None:
        """Fold one assistant turn's `message.usage` into the running rollup.

        OpenClaw's usage shape:
            {input, output, cacheRead, cacheWrite, totalTokens,
             cost: {input, output, cacheRead, cacheWrite, total}}

        `input` is cache-miss input; `cacheRead`/`cacheWrite` are the cached
        input sides. All three are input tokens, so they roll into `tokens_in`.
        `cost.total` is the provider-surfaced charge — often 0 (not surfaced),
        in which case `cost_usd` stays None and `basis` stays "unknown" rather
        than stamping a misleading $0.
        """
        if not isinstance(raw, dict):
            return
        self._n_llm_calls += 1
        self._tok_in += int(raw.get("input", 0) or 0) + int(raw.get("cacheRead", 0) or 0)
        self._tok_out += int(raw.get("output", 0) or 0)
        cost = raw.get("cost") or {}
        total = cost.get("total") if isinstance(cost, dict) else None
        if isinstance(total, (int, float)) and total > 0:
            self._cost_usd = (self._cost_usd or 0.0) + float(total)
            self._cost_reported = True
        if self.nerve is not None:
            self.nerve.emit(
                "agent_usage",
                cell=self.cell,
                decision_date=self.decision_date,
                tokens_in=self._tok_in,
                tokens_out=self._tok_out,
                n_llm_calls=self._n_llm_calls,
                cost_usd=self._cost_usd,
            )

    def usage(self) -> Usage:
        """The accumulated token/cost rollup for this cell, as a `Usage`."""
        return Usage(
            n_llm_calls=self._n_llm_calls,
            tokens_in=self._tok_in,
            tokens_out=self._tok_out,
            reasoning_tokens=0,
            cost_usd=self._cost_usd,
            basis="reported" if self._cost_reported else "unknown",
        )

    def _process_line(self, line: str) -> None:
        if not line:
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return

        msg = ev.get("message") or {}
        role = msg.get("role")
        ts = ev.get("timestamp", "")

        if role == "assistant":
            self._round += 1
            self._accumulate_usage(msg.get("usage"))
            for c in msg.get("content") or []:
                ctype = c.get("type")
                if ctype == "toolCall":
                    call = {
                        "ts": ts,
                        "tool": c.get("name", ""),
                        "args": c.get("arguments", {}),
                    }
                    self.tool_calls.append(call)
                    self.access_log.append(
                        "agent_tool_call",
                        tool=call["tool"],
                        args=str(call["args"])[:120],
                    )
                    if self.nerve is not None:
                        self.nerve.emit(
                            "agent_tool_call",
                            cell=self.cell,
                            decision_date=self.decision_date,
                            tool=call["tool"],
                            args=str(call["args"])[:120],
                        )
                elif ctype in ("text", "thinking"):
                    # A reasoning turn: emit a stage line so an operator can see
                    # the agent thinking live, not just calling tools. OpenClaw's
                    # thinking items carry the text under `thinking`.
                    if self.nerve is not None:
                        text = str(
                            c.get("text", "") or c.get("thinking", "") or c.get("content", "") or ""
                        )
                        self.nerve.emit(
                            "agent_stage",
                            cell=self.cell,
                            decision_date=self.decision_date,
                            stage="reasoning",
                            round=self._round,
                            text=text[:80],
                        )
        elif role == "toolResult":
            text = (msg.get("content") or [{}])[0].get("text", "")
            tool = msg.get("toolName", "")
            is_error = msg.get("isError", False) or _is_error_text(text)
            result = {
                "ts": ts,
                "tool": tool,
                "ok": not is_error,
                "text": text[:300],
            }
            self.tool_results.append(result)
            self.access_log.append(
                "agent_tool_result",
                tool=tool,
                ok=not is_error,
                text=text[:120] if is_error else "",
            )
            if self.nerve is not None:
                self.nerve.emit(
                    "agent_tool_result",
                    cell=self.cell,
                    decision_date=self.decision_date,
                    tool=tool,
                    ok=not is_error,
                    text=text[:80] if is_error else "",
                )
            if is_error:
                self.tool_errors.append({"tool": tool, "error": text[:300]})
                self._consecutive_errors += 1
                if self.fail_fast_errors and self._consecutive_errors >= self.fail_fast_errors:
                    self.fail_fast_triggered = True
            else:
                self._consecutive_errors = 0


def _is_error_text(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _ERROR_MARKERS)
