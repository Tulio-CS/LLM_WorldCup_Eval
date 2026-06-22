"""Provider factory.

Maps a model entry's ``provider`` to the right adapter, or to the deterministic
mock when ``dry_run`` is set. SDK imports happen inside each adapter, so adding
a model never forces an SDK install you don't use.
"""

from __future__ import annotations

from typing import Any

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider, ProviderError
from .gemini_provider import GeminiProvider
from .grok_provider import GrokProvider
from .mock import MockProvider
from .openai_provider import OpenAICompatibleProvider

_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
}


def build_provider(
    model_config: dict[str, Any], *, dry_run: bool = False, timeout: int = 120
) -> BaseProvider:
    if dry_run:
        return MockProvider(model_config, timeout=timeout)
    provider_name = model_config.get("provider")
    cls = _REGISTRY.get(provider_name)
    if cls is None:
        raise ProviderError(
            f"Unknown provider {provider_name!r} for model "
            f"{model_config.get('key')!r}. Known providers: {sorted(_REGISTRY)}"
        )
    return cls(model_config, timeout=timeout)
