"""xAI Grok provider.

Grok exposes an OpenAI-compatible Chat Completions endpoint, so this is a thin
specialization of :class:`OpenAICompatibleProvider` with xAI defaults.
"""

from __future__ import annotations

from typing import Any

from .openai_provider import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    name = "grok"

    def __init__(self, model_config: dict[str, Any], *, timeout: int = 120) -> None:
        model_config = dict(model_config)
        model_config.setdefault("api_key_env", "XAI_API_KEY")
        model_config.setdefault("base_url", "https://api.x.ai/v1")
        super().__init__(model_config, timeout=timeout)
