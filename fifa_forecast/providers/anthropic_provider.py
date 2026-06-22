"""Anthropic Claude provider (Messages API).

Opus 4.8 / Sonnet 4.6 reject temperature/top_p, so those are sent only when a
model entry explicitly lists them (older models). Reasoning depth is controlled
by ``effort`` (-> output_config) and optional adaptive ``thinking``. The
request id is exposed by the SDK as ``message._request_id``.
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


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model_config: dict[str, Any], *, timeout: int = 120) -> None:
        super().__init__(model_config, timeout=timeout)
        self.api_key_env = model_config.get("api_key_env", "ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Environment variable {self.api_key_env} is not set."
            )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
        return self._client

    def _build_kwargs(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        p = self.params
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": p.get("max_tokens", 8000),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if "effort" in p:
            kwargs["output_config"] = {"effort": p["effort"]}
        if "thinking" in p:
            thinking = p["thinking"]
            kwargs["thinking"] = (
                {"type": thinking} if isinstance(thinking, str) else thinking
            )
        # Only for older models that still accept sampling params.
        if "temperature" in p:
            kwargs["temperature"] = p["temperature"]
        if "top_p" in p:
            kwargs["top_p"] = p["top_p"]
        return kwargs

    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        client = self._get_client()
        kwargs = self._build_kwargs(system_prompt, user_prompt)
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic API call failed: {exc}") from exc

        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        )

        usage = getattr(message, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        request_id = getattr(message, "_request_id", None)
        response_payload = _to_dict(message)
        response_payload["_request_id"] = request_id

        return ProviderResult(
            raw_response_text=text,
            request_payload=kwargs,
            response_payload=response_payload,
            response_id=getattr(message, "id", None),
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=None,  # thinking tokens are included in output_tokens
            total_tokens=total_tokens,
            model_reported=getattr(message, "model", None),
            trace={
                "usage": _to_dict(usage) if usage is not None else None,
                "stop_reason": getattr(message, "stop_reason", None),
                "request_id": request_id,
            },
        )
