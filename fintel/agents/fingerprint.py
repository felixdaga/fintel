"""Fingerprint: what makes a run reproducible.

The old repo fingerprinted OpenClaw only — `build_fingerprint()` returned
`mission_text: None` for plugin agents, so the LangGraph desks' prompts and
flags weren't hashed at all. Two runs could differ in configuration and the
platform couldn't tell.

Here the fingerprint covers every adapter uniformly: the adapter name and version,
the model pin, the channel, and a hash of the prompt text. A strategy package
supplies the prompt; the platform supplies everything else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Fingerprint:
    """A digest of everything that should make two runs identical."""

    agent_name: str
    agent_version: str
    model: str
    channel: str
    prompt_hash: str
    data_kinds: tuple[str, ...]
    adapter_params: dict[str, Any]

    @property
    def digest(self) -> str:
        blob = json.dumps(
            {
                "agent": self.agent_name,
                "version": self.agent_version,
                "model": self.model,
                "channel": self.channel,
                "prompt": self.prompt_hash,
                "kinds": list(self.data_kinds),
                "params": self.adapter_params,
            },
            sort_keys=True,
        )
        return _hash(blob)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "channel": self.channel,
            "prompt_hash": self.prompt_hash,
            "data_kinds": list(self.data_kinds),
            "adapter_params": self.adapter_params,
            "digest": self.digest,
        }


def fingerprint(
    *,
    agent_name: str,
    agent_version: str,
    model: str,
    channel: str,
    prompt: str,
    data_kinds: tuple[str, ...],
    adapter_params: dict[str, Any] | None = None,
) -> Fingerprint:
    return Fingerprint(
        agent_name=agent_name,
        agent_version=agent_version,
        model=model,
        channel=channel,
        prompt_hash=_hash(prompt),
        data_kinds=data_kinds,
        adapter_params=adapter_params or {},
    )
