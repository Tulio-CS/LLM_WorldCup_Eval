"""Experiment manifest for reproducibility.

Captures execution date, git commit, model/prompt/software versions, installed
provider-SDK versions and environment info into ``experiment_manifest.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from . import config as cfg
from .prompts import PROMPT_VERSION, prompt_catalog


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cfg.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover
        pass
    return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cfg.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        pass
    return None


def _sdk_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in ("openai", "anthropic", "google-genai", "pandas", "openpyxl"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(
    config: cfg.Config, *, dry_run: bool, matches_count: int
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "execution_date_utc": now.isoformat().replace("+00:00", "Z"),
        "execution_date_local": datetime.now().astimezone().isoformat(),
        "software_version": __version__,
        "config_software_version": config.software_version,
        "prompt_version": PROMPT_VERSION,
        "dry_run": dry_run,
        "git": {
            "commit": _git_commit(),
            "dirty_working_tree": _git_dirty(),
        },
        "experiment": {
            "runs_per_combination": config.runs_per_combination,
            "team_order_types": config.team_order_types,
            "prompt_ids": config.prompt_ids,
            "max_retries": config.max_retries,
            "request_timeout": config.request_timeout,
            "matches_csv": config.matches_csv,
            "matches_count": matches_count,
            "database_path": config.database_path,
        },
        "models": [
            {
                "key": m.get("key"),
                "provider": m.get("provider"),
                "model_id": m.get("model_id"),
                "enabled": m.get("enabled", True),
                "params": m.get("params"),
                "pricing": m.get("pricing"),
            }
            for m in config.models
        ],
        "prompts": [
            {k: v for k, v in p.items() if k != "template"} for p in prompt_catalog()
        ],
        "provider_sdk_versions": _sdk_versions(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }


def write_manifest(manifest: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path) if path else (cfg.ROOT / "experiment_manifest.json")
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target
