"""The catcher must see what the agent sees — including MCP tool errors
that never round-trip to our server.
"""

from __future__ import annotations

import json
from pathlib import Path

from fintel.agents.adapters.catcher import _TranscriptCatcher


def _wait_for(condition, *, timeout: float = 2.0, interval: float = 0.05) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)


class _FakeLog:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def append(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})


def _write_transcript(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_catcher_flags_mcp_timeout_errors(tmp_path: Path) -> None:
    """The exact -32001 errors from run 0005 must be flagged as tool errors."""
    transcript = tmp_path / "fintel-test.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "timestamp": "2026-08-02T01:15:30.478Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "name": "fintel__get_fundamentals", "arguments": {"symbol": "AAPL"}},
                        {"type": "toolCall", "name": "fintel__get_prices", "arguments": {"symbol": "AAPL"}},
                    ],
                },
            },
            {
                "timestamp": "2026-08-02T01:16:30.542Z",
                "message": {
                    "role": "toolResult",
                    "toolName": "fintel__get_fundamentals",
                    "content": [{"text": '{"status": "error", "tool": "fintel__get_fundamentals", "error": "MCP error -32001: Request timed out"}'}],
                },
            },
            {
                "timestamp": "2026-08-02T01:16:30.542Z",
                "message": {
                    "role": "toolResult",
                    "toolName": "fintel__get_prices",
                    "content": [{"text": '{"status": "error", "error": "MCP error -32000: Connection closed"}'}],
                },
            },
        ],
    )

    log = _FakeLog()
    catcher = _TranscriptCatcher(
        transcript_path=transcript, access_log=log, fail_fast_errors=2
    )
    catcher.start()
    # Wait for the tail thread to process the pre-written file.
    _wait_for(lambda: len(catcher.tool_results) >= 2, timeout=2.0)
    catcher.stop()

    assert len(catcher.tool_calls) == 2
    assert len(catcher.tool_results) == 2
    assert len(catcher.tool_errors) == 2, "both -32001/-32000 results are errors"
    assert catcher.fail_fast_triggered, "2 consecutive errors should trip fail-fast"

    # finalize logs a tool_errors event the health audit reads
    catcher.finalize(tmp_path)
    err_events = [e for e in log.events if e["event"] == "tool_errors"]
    assert len(err_events) == 1
    assert err_events[0]["n"] == 2

    # transcript copied into session dir
    assert (tmp_path / "agent_transcript.jsonl").is_file()


def test_catcher_distinguishes_ok_results(tmp_path: Path) -> None:
    transcript = tmp_path / "fintel-ok.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "timestamp": "t1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "fintel__get_prices", "arguments": {"symbol": "AAPL"}}],
                },
            },
            {
                "timestamp": "t2",
                "message": {
                    "role": "toolResult",
                    "toolName": "fintel__get_prices",
                    "content": [{"text": '{"status": "ok", "rows": 252}'}],
                },
            },
        ],
    )

    log = _FakeLog()
    catcher = _TranscriptCatcher(
        transcript_path=transcript, access_log=log, fail_fast_errors=3
    )
    catcher.start()
    _wait_for(lambda: len(catcher.tool_results) >= 1, timeout=2.0)
    catcher.stop()

    assert len(catcher.tool_errors) == 0
    assert not catcher.fail_fast_triggered


def test_catcher_handles_missing_transcript(tmp_path: Path) -> None:
    """A CLI with no transcript path should not crash."""
    log = _FakeLog()
    catcher = _TranscriptCatcher(transcript_path=None, access_log=log)
    catcher.start()
    catcher.stop()
    catcher.finalize(tmp_path)
    assert catcher.tool_errors == []
