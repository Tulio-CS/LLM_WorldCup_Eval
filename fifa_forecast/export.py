"""Excel export.

Builds ``FIFA_World_Cup_2026_AI_Forecasts.xlsx`` with the worksheets:
Forecasts, Matches, Models, Prompts, Raw Responses, API Metadata, Errors and
Summary. The Summary sheet contains counts only — no evaluation metrics.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from . import config as cfg
from . import prompts as prompt_lib
from .matches import load_matches

# Excel hard limit on characters per cell.
_CELL_LIMIT = 32_000


def _clean_cell(value: Any) -> Any:
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        if len(value) > _CELL_LIMIT:
            value = value[:_CELL_LIMIT] + " …[truncated]"
    return value


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.map(_clean_cell)


def _load_forecasts(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM forecast_runs", conn)
    finally:
        conn.close()


def _models_df(config: cfg.Config) -> pd.DataFrame:
    rows = []
    for m in config.models:
        pricing = m.get("pricing") or {}
        rows.append(
            {
                "key": m.get("key"),
                "provider": m.get("provider"),
                "model_id": m.get("model_id"),
                "enabled": m.get("enabled", True),
                "params": json.dumps(m.get("params") or {}, ensure_ascii=False),
                "price_input_per_1m": pricing.get("input"),
                "price_output_per_1m": pricing.get("output"),
                "api_key_env": m.get("api_key_env"),
            }
        )
    return pd.DataFrame(rows)


def _matches_df(config: cfg.Config) -> pd.DataFrame:
    try:
        matches = load_matches(cfg.ROOT / config.matches_csv)
    except Exception:
        return pd.DataFrame(columns=["match_id", "phase", "team_1", "team_2", "kickoff_datetime"])
    return pd.DataFrame([m.as_dict() for m in matches])


def _prompts_df() -> pd.DataFrame:
    return pd.DataFrame(prompt_lib.prompt_catalog())


def _summary_df(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Counts only — deliberately no accuracy/metric computation."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any) -> None:
        rows.append({"metric": metric, "value": value})

    add("total_executions", len(forecasts))
    if not forecasts.empty:
        status = forecasts["execution_status"].value_counts(dropna=False)
        add("successful_executions", int(status.get("success", 0)))
        add("errored_executions", int(status.get("error", 0)))
        add("valid_json_responses", int((forecasts["json_valid"] == 1).sum()))
        add("invalid_json_responses", int((forecasts["json_valid"] != 1).sum()))
        add("distinct_matches", int(forecasts["match_id"].nunique()))
        add("distinct_models", int(forecasts["model"].nunique()))
        add("distinct_prompts", int(forecasts["prompt_id"].nunique()))
        add("distinct_team_orders", int(forecasts["team_order_type"].nunique()))
    return pd.DataFrame(rows)


def _breakdown(forecasts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame(columns=by + ["executions"])
    grouped = (
        forecasts.groupby(by, dropna=False).size().reset_index(name="executions")
    )
    return grouped


def export_workbook(config: cfg.Config, output_path: str | Path | None = None) -> Path:
    db_path = cfg.ROOT / config.database_path
    forecasts = _load_forecasts(db_path)

    raw_cols = [
        c
        for c in ("run_id", "match_id", "model", "prompt_id", "team_order_type",
                  "response_id", "raw_response")
        if c in forecasts.columns
    ]
    raw_df = forecasts[raw_cols] if not forecasts.empty else pd.DataFrame(columns=raw_cols)

    meta_cols = [
        c
        for c in (
            "run_id", "provider", "model", "model_id", "prompt_id",
            "request_timestamp_utc", "response_timestamp_utc", "latency_ms",
            "temperature", "top_p", "seed", "max_tokens", "reasoning_effort",
            "response_format", "response_id", "request_id", "trace_id",
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "total_tokens", "api_cost", "json_valid", "execution_status",
        )
        if c in forecasts.columns
    ]
    meta_df = forecasts[meta_cols] if not forecasts.empty else pd.DataFrame(columns=meta_cols)

    if not forecasts.empty:
        errors_df = forecasts[forecasts["execution_status"] == "error"]
        err_cols = [
            c
            for c in ("run_id", "match_id", "model", "model_id", "prompt_id",
                      "team_order_type", "repetition_number", "attempt_count",
                      "error_message", "request_timestamp_utc")
            if c in errors_df.columns
        ]
        errors_df = errors_df[err_cols]
    else:
        errors_df = pd.DataFrame(
            columns=["run_id", "match_id", "model", "model_id", "prompt_id",
                     "team_order_type", "repetition_number", "attempt_count",
                     "error_message", "request_timestamp_utc"]
        )

    target = Path(output_path) if output_path else (cfg.EXPORTS_DIR / cfg.EXCEL_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pd.DataFrame] = {
        "Forecasts": forecasts,
        "Matches": _matches_df(config),
        "Models": _models_df(config),
        "Prompts": _prompts_df(),
        "Raw Responses": raw_df,
        "API Metadata": meta_df,
        "Errors": errors_df,
        "Summary": _summary_df(forecasts),
        "Counts by Model": _breakdown(forecasts, ["provider", "model", "execution_status"]),
        "Counts by Prompt": _breakdown(forecasts, ["prompt_id", "execution_status"]),
        "Counts by Order": _breakdown(forecasts, ["team_order_type", "execution_status"]),
    }

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, df in sheets.items():
            _clean_df(df).to_excel(writer, sheet_name=name[:31], index=False)

    return target
