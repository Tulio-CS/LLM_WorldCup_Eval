"""Central configuration for the FIFA WC2026 forecast benchmark.

Everything tunable lives here: the model catalog, prompt strategies, repetition
count, retry policy, file locations and pricing. Values can be overridden at
runtime by a ``config.json`` placed in the project root (shallow-merged over the
defaults below) so the code never has to be edited to add a model or change a
setting.

API keys are *not* stored here — they are read from the environment (.env).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Load .env if present; harmless if python-dotenv is missing.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ``FIFA_DATA_DIR`` and ``FIFA_DB_PATH`` let a deployment redirect all generated
# artifacts (JSON archive + SQLite DB) onto a mounted persistent volume without
# touching the code — e.g. on Coolify: FIFA_DATA_DIR=/data.
ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR holds the SQLite DB, raw artifacts and exports. Set FIFA_DATA_DIR to a
# mounted persistent volume (e.g. /data on Coolify) to relocate all generated
# state without touching code; locally it defaults to <repo>/data.
DATA_DIR = Path(os.environ.get("FIFA_DATA_DIR") or (ROOT / "data"))
RAW_REQUESTS_DIR = DATA_DIR / "raw_requests"
RAW_RESPONSES_DIR = DATA_DIR / "raw_responses"
TRACES_DIR = DATA_DIR / "traces"
METADATA_DIR = DATA_DIR / "metadata"
EXPORTS_DIR = DATA_DIR / "exports"

ALL_DATA_DIRS = (
    DATA_DIR,
    RAW_REQUESTS_DIR,
    RAW_RESPONSES_DIR,
    TRACES_DIR,
    METADATA_DIR,
    EXPORTS_DIR,
)

EXCEL_FILENAME = "FIFA_World_Cup_2026_AI_Forecasts.xlsx"


# --------------------------------------------------------------------------- #
# Model catalog
# --------------------------------------------------------------------------- #
# Each entry is a self-contained description of one model under test.
#
#   key       : unique identifier stored in the dataset (free to rename)
#   provider  : which adapter handles it (openai | anthropic | gemini | grok)
#   model_id  : the exact API model string sent on the wire
#   enabled   : skip the model entirely when False
#   api_key_env: environment variable holding the credential
#   params    : provider-specific request parameters. Whatever is present here
#               is sent on the wire AND recorded in the dataset. Omit a param
#               (e.g. temperature) when the provider/model rejects it.
#   pricing   : USD per 1,000,000 tokens {input, output}, used for api_cost and
#               the `estimate` command. Anthropic prices are authoritative; the
#               OpenAI/Gemini/xAI values below are BEST-EFFORT ESTIMATES — verify
#               them against each provider's current pricing page. Set to null to
#               store api_cost as NULL for that model.
#
# Notes on parameters:
#   * Anthropic Opus 4.8 / Sonnet 4.6 REJECT temperature/top_p — do not add them.
#     Depth is controlled by output_config.effort (passed via params.effort).
#   * OpenAI GPT-5 reasoning models use reasoning_effort and only accept the
#     default temperature; we therefore omit temperature for them.
DEFAULT_MODELS: list[dict[str, Any]] = [
    # ---- OpenAI -----------------------------------------------------------
    {
        "key": "gpt-5",
        "provider": "openai",
        "model_id": "gpt-5",
        "enabled": True,
        "api_key_env": "OPENAI_API_KEY",
        "params": {"reasoning_effort": "medium", "max_output_tokens": 8000},
        "pricing": {"input": 1.25, "output": 10.0},  # ESTIMATE — verify
    },
    {
        "key": "gpt-5-mini",
        "provider": "openai",
        "model_id": "gpt-5-mini",
        "enabled": True,
        "api_key_env": "OPENAI_API_KEY",
        "params": {"reasoning_effort": "medium", "max_output_tokens": 8000},
        "pricing": {"input": 0.25, "output": 2.0},  # ESTIMATE — verify
    },
    {
        "key": "gpt-5-nano",
        "provider": "openai",
        "model_id": "gpt-5-nano",
        "enabled": True,
        "api_key_env": "OPENAI_API_KEY",
        "params": {"reasoning_effort": "medium", "max_output_tokens": 8000},
        "pricing": {"input": 0.05, "output": 0.40},  # ESTIMATE — verify
    },
    # ---- Anthropic --------------------------------------------------------
    {
        "key": "claude-opus",
        "provider": "anthropic",
        "model_id": "claude-opus-4-8",
        "enabled": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        # No temperature/top_p (rejected by Opus 4.8). effort controls depth.
        "params": {"effort": "medium", "max_tokens": 8000},
        "pricing": {"input": 5.0, "output": 25.0},
    },
    {
        "key": "claude-sonnet",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "enabled": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        "params": {"effort": "medium", "max_tokens": 8000},
        "pricing": {"input": 3.0, "output": 15.0},
    },
    # ---- Google Gemini ----------------------------------------------------
    {
        "key": "gemini-2.5-pro",
        "provider": "gemini",
        "model_id": "gemini-2.5-pro",
        "enabled": True,
        "api_key_env": "GEMINI_API_KEY",
        "params": {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 8000},
        "pricing": {"input": 1.25, "output": 10.0},  # ESTIMATE — verify
    },
    {
        "key": "gemini-2.5-flash",
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
        "enabled": True,
        "api_key_env": "GEMINI_API_KEY",
        "params": {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 8000},
        "pricing": {"input": 0.30, "output": 2.50},  # ESTIMATE — verify
    },
    # ---- xAI Grok (OpenAI-compatible API) ---------------------------------
    {
        "key": "grok",
        "provider": "grok",
        "model_id": "grok-4",
        "enabled": True,
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "params": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 8000},
        "pricing": {"input": 3.0, "output": 15.0},  # ESTIMATE — verify
    },
]

# Repetitions per model — set once here so you don't re-enter a count on every
# run. Both team orderings (original + reversed) run for each rep, so reps=5
# means 5 original + 5 reversed per (prompt, moment). Any model not listed
# falls back to runs_per_combination.
_REPS_PER_MODEL: dict[str, int] = {
    "gpt-5": 5, "gpt-5-mini": 5, "gpt-5-nano": 5,
    "claude-opus": 2, "claude-sonnet": 2,
    "gemini-2.5-pro": 2, "gemini-2.5-flash": 2,
    "grok": 5,
}
for _model in DEFAULT_MODELS:
    _model.setdefault("reps", _REPS_PER_MODEL.get(_model["key"], 10))


@dataclass
class Config:
    """Resolved experiment configuration."""

    runs_per_combination: int = 10
    team_order_types: list[str] = field(
        default_factory=lambda: ["original", "reversed"]
    )
    # Capture the same match at three points in time: before kickoff, at
    # half-time, and after the final whistle. The kickoff time + moment are
    # injected into every prompt (see prompts.render).
    match_moments: list[str] = field(
        default_factory=lambda: ["pre_match"]
    )
    prompt_ids: list[str] = field(
        default_factory=lambda: [
            "simple-prediction",
            "probability-prediction",
            "six-hats-prediction",
        ]
    )
    max_retries: int = 3
    retry_base_delay: float = 2.0  # seconds; exponential backoff
    request_timeout: int = 120  # seconds

    matches_csv: str = "fifa_world_cup_2026_future_matches.csv"
    database_path: str = field(
        default_factory=lambda: os.environ.get("FIFA_DB_PATH", "fifa_forecasts.db")
    )

    models: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_MODELS))

    software_version: str = "1.0.0"

    # --- derived helpers ---------------------------------------------------
    def enabled_models(self) -> list[dict[str, Any]]:
        return [m for m in self.models if m.get("enabled", True)]

    def model_by_key(self, key: str) -> dict[str, Any] | None:
        for m in self.models:
            if m.get("key") == key:
                return m
        return None

    def reps_for(self, key: str) -> int:
        m = self.model_by_key(key)
        if m and m.get("reps") is not None:
            return int(m["reps"])
        return self.runs_per_combination


def load_config(overrides_path: str | os.PathLike | None = None) -> Config:
    """Build a :class:`Config`, applying ``config.json`` overrides if present.

    Override file shape (all keys optional)::

        {
          "runs_per_combination": 3,
          "prompt_ids": ["simple-prediction"],
          "models": [ {... full model entries replacing the defaults ...} ]
        }
    """
    cfg = Config()
    path = Path(overrides_path) if overrides_path else (ROOT / "config.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Environment overrides (used by container deployments; take precedence).
    # Absolute paths work directly because `ROOT / "/data/x"` resolves to "/data/x".
    env_db = os.environ.get("FIFA_DB_PATH")
    if env_db:
        cfg.database_path = env_db
    env_matches = os.environ.get("FIFA_MATCHES_CSV")
    if env_matches:
        cfg.matches_csv = env_matches
    return cfg
