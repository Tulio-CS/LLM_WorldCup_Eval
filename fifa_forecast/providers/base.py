"""Provider base class and the normalized result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a provider call fails (network, auth, API error, etc.)."""


@dataclass
class ProviderResult:
    """Everything one model call yields, normalized across vendors.

    Nothing is discarded: ``request_payload`` and ``response_payload`` hold the
    full serialized request/response, while the typed fields surface the bits
    the dataset indexes directly.
    """

    raw_response_text: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]

    response_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    model_reported: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """One instance per model entry; reused across all of that model's calls."""

    #: provider identifier, e.g. "openai"
    name: str = "base"

    def __init__(self, model_config: dict[str, Any], *, timeout: int = 120) -> None:
        self.model_config = model_config
        self.model_id: str = model_config["model_id"]
        self.params: dict[str, Any] = dict(model_config.get("params") or {})
        self.timeout = timeout

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        """Execute a single forecast call. Raise :class:`ProviderError` on failure."""

    # -- shared helpers -----------------------------------------------------
    def compute_cost(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> float | None:
        pricing = self.model_config.get("pricing")
        if not pricing or prompt_tokens is None or completion_tokens is None:
            return None
        try:
            cost = (
                prompt_tokens / 1_000_000 * float(pricing["input"])
                + completion_tokens / 1_000_000 * float(pricing["output"])
            )
            return round(cost, 8)
        except (KeyError, TypeError, ValueError):
            return None
