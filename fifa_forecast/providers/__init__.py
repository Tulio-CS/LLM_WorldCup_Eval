"""Provider adapters.

Each adapter normalises one vendor's SDK into a common :class:`ProviderResult`.
SDK imports are lazy so the package loads even when a given SDK is not installed.
"""

from .base import BaseProvider, ProviderError, ProviderResult
from .registry import build_provider

__all__ = ["BaseProvider", "ProviderError", "ProviderResult", "build_provider"]
