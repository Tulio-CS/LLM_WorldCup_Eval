"""OpenAI (and OpenAI-compatible, e.g. xAI Grok) provider.

Uses the Chat Completions API. GPT-5 reasoning models take ``reasoning_effort``
and only the default temperature, so temperature/top_p are sent only when the
model entry explicitly lists them. Request id is captured from response headers
and reasoning-token usage from ``completion_tokens_details``.
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseProvider, ProviderError, ProviderResult


def _to_dict(obj: Any) -> dict[str, Any]:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover
                pass
    try:
        return dict(obj)
    except Exception:  # pragma: no cover
        return {"repr": repr(obj)}


class OpenAICompatibleProvider(BaseProvider):
    """Works against the OpenAI API and any OpenAI-compatible endpoint."""

    name = "openai"

    def __init__(self, model_config: dict[str, Any], *, timeout: int = 120) -> None:
        super().__init__(model_config, timeout=timeout)
        self.api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
        self.base_url = model_config.get("base_url")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Environment variable {self.api_key_env} is not set."
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _build_kwargs(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        p = self.params
        if "temperature" in p:
            kwargs["temperature"] = p["temperature"]
        if "top_p" in p:
            kwargs["top_p"] = p["top_p"]
        if "seed" in p:
            kwargs["seed"] = p["seed"]
        if "reasoning_effort" in p:
            kwargs["reasoning_effort"] = p["reasoning_effort"]
        # Newer models use max_completion_tokens; accept either key in config.
        max_tokens = p.get("max_output_tokens", p.get("max_tokens"))
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        # Ask for a JSON object unless explicitly disabled.
        if p.get("response_format") != "none":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        client = self._get_client()
        kwargs = self._build_kwargs(system_prompt, user_prompt)

        request_id = None
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
            request_id = raw.headers.get("x-request-id")
            completion = raw.parse()
        except Exception as exc:  # noqa: BLE001 - normalize all SDK errors
            raise ProviderError(f"OpenAI API call failed: {exc}") from exc

        try:
            text = completion.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise ProviderError(f"Malformed OpenAI response: {exc}") from exc

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        reasoning_tokens = None
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

        response_payload = _to_dict(completion)
        response_payload["_request_id"] = request_id

        return ProviderResult(
            raw_response_text=text,
            request_payload=kwargs,
            response_payload=response_payload,
            response_id=getattr(completion, "id", None),
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model_reported=getattr(completion, "model", None),
            trace={
                "usage": _to_dict(usage) if usage is not None else None,
                "system_fingerprint": getattr(
                    completion, "system_fingerprint", None
                ),
                "request_id": request_id,
            },
        )
