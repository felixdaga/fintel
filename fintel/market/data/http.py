"""Massive.com HTTP access.

Retries 429 and 5xx with jittered backoff; raises on everything else. The
previous client returned `{}` for 401 and 403, so a bad key or an out-of-plan
window was indistinguishable from a symbol that genuinely had no data — a whole
backtest could come back empty and look merely uneventful.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from fintel.market.data.base import DataError, EntitlementError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.massive.com"
MAX_ATTEMPTS = 6


class MassiveClient:
    def __init__(self, api_key: str, *, base_url: str = BASE_URL, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.n_requests = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        p = dict(params or {})
        p["apiKey"] = self._api_key
        url = path if path.startswith("http") else f"{self._base_url}{path}"
        self.n_requests += 1

        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._client.get(url, params=p)
            except httpx.RequestError as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise DataError(f"network error for {path}: {exc}") from exc
                self._sleep(attempt, None, f"network error: {exc}")
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == MAX_ATTEMPTS - 1:
                    raise DataError(
                        f"HTTP {resp.status_code} for {path} after {MAX_ATTEMPTS} tries"
                    )
                self._sleep(attempt, resp.headers.get("Retry-After"), f"HTTP {resp.status_code}")
                continue

            if resp.status_code == 401:
                raise DataError("Massive API rejected the key (401); check MASSIVE_API_KEY")
            if resp.status_code == 403:
                msg = self._message(resp)
                if "timeframe" in msg.lower() or "entitlement" in msg.lower():
                    raise EntitlementError(
                        f"Massive plan does not cover the requested window for {path}: {msg[:200]}"
                    )
                raise DataError(f"Massive API forbade {path}: {msg[:200]}")
            if resp.status_code >= 400:
                raise DataError(f"HTTP {resp.status_code} for {path}: {resp.text[:200]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise DataError(f"non-JSON response for {path}: {exc}") from exc

        raise DataError(f"exhausted retries for {path}")

    def paginate(self, path: str, params: dict[str, Any]) -> list[dict]:
        out: list[dict] = []
        url: str | None = path
        p = dict(params)
        while url is not None:
            data = self.get(url, p)
            out.extend(data.get("results") or [])
            nxt = data.get("next_url")
            url, p = (nxt, {}) if nxt else (None, p)
        return out

    @staticmethod
    def _message(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text
        return str(body.get("error") or body.get("message") or resp.text)

    def _sleep(self, attempt: int, retry_after: str | None, why: str) -> None:
        try:
            wait = float(retry_after) if retry_after else min(2**attempt, 60.0)
        except ValueError:
            wait = min(2**attempt, 60.0)
        wait *= 0.8 + random.random() * 0.4
        logger.warning("%s (attempt %d/%d) — retry in %.1fs", why, attempt + 1, MAX_ATTEMPTS, wait)
        time.sleep(wait)

    def close(self) -> None:
        self._client.close()
