"""Google Gemini provider (google-genai SDK).

Captures token usage (including ``thoughts_token_count`` as reasoning tokens),
candidate finish reasons and safety ratings into the trace. JSON output is
requested via ``response_mime_type``.
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
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                try:
                    return fn()
                except Exception:  # pragma: no cover
                    pass
            except Exception:  # pragma: no cover
                pass
    try:
        return dict(obj)
    except Exception:  # pragma: no cover
        return {"repr": repr(obj)}


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model_config: dict[str, Any], *, timeout: int = 120) -> None:
        super().__init__(model_config, timeout=timeout)
        self.api_key_env = model_config.get("api_key_env", "GEMINI_API_KEY")
        self._client = None
        self._types = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Environment variable {self.api_key_env} is not set."
            )
        self._types = types
        try:
            http_options = types.HttpOptions(timeout=self.timeout * 1000)
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        except Exception:  # pragma: no cover - older SDKs lack HttpOptions.timeout
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_config(self, system_prompt: str):
        types = self._types
        p = self.params
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
        }
        if "temperature" in p:
            cfg_kwargs["temperature"] = p["temperature"]
        if "top_p" in p:
            cfg_kwargs["top_p"] = p["top_p"]
        if "seed" in p:
            cfg_kwargs["seed"] = p["seed"]
        max_tokens = p.get("max_output_tokens", p.get("max_tokens"))
        if max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_tokens
        return types.GenerateContentConfig(**cfg_kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        client = self._get_client()
        config = self._build_config(system_prompt)
        try:
            response = client.models.generate_content(
                model=self.model_id, contents=user_prompt, config=config
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini API call failed: {exc}") from exc

        text = getattr(response, "text", None) or ""

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        completion_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        reasoning_tokens = getattr(usage, "thoughts_token_count", None)

        request_payload = {
            "provider": self.name,
            "model": self.model_id,
            "system_instruction": system_prompt,
            "contents": user_prompt,
            "config": _to_dict(config),
        }
        response_payload = _to_dict(response)

        # Safety + finish-reason metadata for the trace.
        candidates_meta: list[dict[str, Any]] = []
        for cand in getattr(response, "candidates", []) or []:
            candidates_meta.append(
                {
                    "finish_reason": str(getattr(cand, "finish_reason", None)),
                    "safety_ratings": [
                        _to_dict(sr)
                        for sr in (getattr(cand, "safety_ratings", None) or [])
                    ],
                }
            )

        return ProviderResult(
            raw_response_text=text,
            request_payload=request_payload,
            response_payload=response_payload,
            response_id=getattr(response, "response_id", None),
            request_id=getattr(response, "response_id", None),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model_reported=getattr(response, "model_version", None),
            trace={
                "usage": _to_dict(usage) if usage is not None else None,
                "candidates": candidates_meta,
                "prompt_feedback": _to_dict(getattr(response, "prompt_feedback", None))
                if getattr(response, "prompt_feedback", None) is not None
                else None,
            },
        )
