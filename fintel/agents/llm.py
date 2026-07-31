"""Talking to a model, and naming what goes wrong.

Two jobs. First, one shape for a completion so an adapter never parses provider
JSON. Second — the reason this module earns its place — turning provider failures
into the platform's typed outcomes. A rate limit, a refusal on policy grounds and
a prompt that didn't fit are three different events, and the old pipeline reported
all of them as an agent that produced no views.

Cost is reported only when the provider states it. There is no rate card here, so
`basis` stays `unknown` rather than quietly becoming an estimate that later gets
summed with a measured one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from fintel.agents.base import (
    AgentError,
    ContextOverflow,
    MalformedOutput,
    ProviderUnavailable,
    RateLimited,
    SafetyRefusal,
)
from fintel.models.trace import Usage

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_OPENROUTER_KEY = "OPENROUTER_API_KEY"

# Body patterns, checked after status. Order matters: the specific ones first,
# since a provider often wraps a real refusal in a generic error envelope.
BODY_PATTERNS: tuple[tuple[str, type[AgentError]], ...] = (
    ("context length", ContextOverflow),
    ("maximum context", ContextOverflow),
    ("too many tokens", ContextOverflow),
    ("prompt is too long", ContextOverflow),
    ("content filter", SafetyRefusal),
    ("content_policy", SafetyRefusal),
    ("safety", SafetyRefusal),
    ("blocked by", SafetyRefusal),
    ("rate limit", RateLimited),
    ("too many requests", RateLimited),
    ("quota", RateLimited),
    ("overloaded", ProviderUnavailable),
    ("temporarily unavailable", ProviderUnavailable),
)

TRANSIENT_STATUS = frozenset({500, 502, 503, 504, 522, 524})


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Completion:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""

    def call_named(self, name: str) -> ToolCall | None:
        return next((c for c in self.tool_calls if c.name == name), None)


class LLM(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict],
        *,
        tools: tuple[dict, ...] = (),
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...


def classify_status(status: int, body: str) -> AgentError:
    """Name a failed HTTP response.

    An unknown 5xx is transient and worth retrying; an unknown 4xx is our
    request being wrong, which a retry only reproduces at cost.
    """
    lowered = body.lower()
    if status == 429:
        return RateLimited(f"HTTP 429: {body[:200]}")
    for needle, exc in BODY_PATTERNS:
        if needle in lowered:
            return exc(f"HTTP {status}: {body[:200]}")
    if status in TRANSIENT_STATUS:
        return ProviderUnavailable(f"HTTP {status}: {body[:200]}")
    return AgentError(f"HTTP {status}: {body[:200]}")


def check_finish(reason: str, text: str) -> None:
    """Some failures arrive as a successful response with a telling reason."""
    if reason == "content_filter":
        return _raise(SafetyRefusal("provider stopped the response on content policy"))
    if reason == "length":
        return _raise(
            ContextOverflow(
                "response hit the output limit before finishing; raise max_tokens "
                "or narrow the prompt"
            )
        )
    for needle, exc in BODY_PATTERNS:
        if needle in text.lower()[:400]:
            return _raise(exc(f"provider error in body: {text[:200]}"))
    return None


def _raise(exc: AgentError) -> None:
    raise exc


def usage_of(payload: dict, *, model: str) -> Usage:
    raw = payload.get("usage") or {}
    cost = raw.get("cost")
    details = raw.get("completion_tokens_details") or {}
    return Usage(
        n_llm_calls=1,
        tokens_in=int(raw.get("prompt_tokens") or 0),
        tokens_out=int(raw.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        # Reported only when the provider says so. Deriving a price from a rate
        # card is a separate, labelled thing; see Usage.basis.
        cost_usd=float(cost) if cost is not None else None,
        basis="reported" if cost is not None else "unknown",
    )


def as_tool_spec(name: str, description: str, schema: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


def parse_completion(payload: dict, *, model: str) -> Completion:
    import json

    choices = payload.get("choices") or []
    if not choices:
        raise MalformedOutput(f"no choices in response: {str(payload)[:200]}")
    message = choices[0].get("message") or {}
    reason = str(choices[0].get("finish_reason") or "")
    text = str(message.get("content") or "")

    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        body = fn.get("arguments")
        try:
            args = json.loads(body) if isinstance(body, str) else dict(body or {})
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MalformedOutput(
                f"tool call {fn.get('name')!r} had unparseable arguments: {exc}"
            ) from exc
        calls.append(
            ToolCall(id=str(raw.get("id") or f"call_{len(calls)}"), name=str(fn.get("name") or ""),
                     arguments=args)
        )

    if not calls:
        check_finish(reason, text)
    return Completion(
        text=text,
        tool_calls=tuple(calls),
        usage=usage_of(payload, model=model),
        model=str(payload.get("model") or model),
        finish_reason=reason,
    )


@dataclass
class OpenRouter:
    """Chat completions over HTTP. Retries only what retrying can fix."""

    model: str = "anthropic/claude-sonnet-4"
    api_key: str | None = None
    temperature: float = 0.0
    timeout_s: float = 180.0
    max_retries: int = 2
    url: str = OPENROUTER_URL

    def complete(
        self,
        messages: list[dict],
        *,
        tools: tuple[dict, ...] = (),
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        import httpx

        if not self.api_key:
            raise AgentError(f"{ENV_OPENROUTER_KEY} is not set; cannot call {self.model}")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Ask for the charge so cost can be `reported` rather than guessed.
            "usage": {"include": True},
        }
        if tools:
            body["tools"] = list(tools)
        if force_tool:
            body["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        if max_tokens:
            body["max_tokens"] = max_tokens

        last: AgentError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(
                    self.url,
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout_s,
                )
            except httpx.TimeoutException as exc:
                from fintel.agents.base import AgentTimeout

                raise AgentTimeout(f"{self.model} timed out after {self.timeout_s}s") from exc
            except httpx.RequestError as exc:
                last = ProviderUnavailable(f"network error calling {self.model}: {exc}")
            else:
                if resp.status_code < 400:
                    return parse_completion(resp.json(), model=self.model)
                last = classify_status(resp.status_code, resp.text)
                if not isinstance(last, RateLimited | ProviderUnavailable):
                    raise last
            if attempt < self.max_retries:
                delay = 2.0 * (attempt + 1)
                logger.warning("%s: %s — retrying in %.0fs", self.model, last, delay)
                time.sleep(delay)
        raise last or ProviderUnavailable(f"{self.model} failed with no diagnosis")

    @staticmethod
    def from_env(model: str | None = None, **kw: Any) -> OpenRouter:
        import os

        return OpenRouter(
            model=model or OpenRouter.model,
            api_key=os.environ.get(ENV_OPENROUTER_KEY) or None,
            **kw,
        )
