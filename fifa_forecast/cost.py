"""Cost projection for a planned data-collection run.

Grounds the estimate in the **observed** average token usage already recorded in
``fifa_forecasts.db`` (per model x prompt) and multiplies by the configured
per-token prices and the planned grid size:

    calls(model, prompt) = orders x reps x occasions x matches
    cost  = sum over (model, prompt) of
            (avg_input/1e6 * price_in + avg_output/1e6 * price_out) * calls

"occasions" is how many times each match is forecast (e.g. 1 pre-match + 2
in-play = 3). Output tokens are taken as ``total_tokens - prompt_tokens`` so
provider-specific reasoning/thinking tokens are counted as billed output
regardless of how each SDK reports them.

Prices come from each model's ``pricing`` in the config (Anthropic authoritative;
others are estimates — see config.py). Token cells with no observed data fall
back to the cross-model average for that prompt, and are flagged.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as cfg


@dataclass
class CostEstimate:
    matches: int
    occasions: int
    reps: int
    orders: int
    total_usd: float
    per_model: pd.DataFrame
    per_prompt: pd.DataFrame
    detail: pd.DataFrame
    total_calls: int
    assumptions: list[str] = field(default_factory=list)


def _observed_tokens(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["model", "prompt_id", "avg_in", "avg_out", "n"])
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT model, prompt_id, prompt_tokens, total_tokens FROM forecast_runs "
            "WHERE execution_status='success' AND prompt_tokens IS NOT NULL "
            "AND total_tokens IS NOT NULL",
            con,
        )
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["model", "prompt_id", "avg_in", "avg_out", "n"])
    df["out"] = (df["total_tokens"] - df["prompt_tokens"]).clip(lower=0)
    return (
        df.groupby(["model", "prompt_id"])
        .agg(avg_in=("prompt_tokens", "mean"), avg_out=("out", "mean"), n=("out", "size"))
        .reset_index()
    )


def estimate_cost(
    config: cfg.Config,
    *,
    matches: int,
    occasions: int = 1,
    reps: int | None = None,
    orders: int | None = None,
    prompt_ids: list[str] | None = None,
) -> CostEstimate:
    reps = reps if reps is not None else config.runs_per_combination
    orders = orders if orders is not None else len(config.team_order_types)
    prompt_ids = prompt_ids or config.prompt_ids
    models = config.enabled_models()

    tok = _observed_tokens(cfg.ROOT / config.database_path)
    lookup = {
        (r["model"], r["prompt_id"]): (r["avg_in"], r["avg_out"])
        for _, r in tok.iterrows()
    }
    # cross-model average per prompt, for cells with no observed data
    prompt_fallback: dict[str, tuple[float, float]] = {}
    if not tok.empty:
        pm = tok.groupby("prompt_id").agg(avg_in=("avg_in", "mean"), avg_out=("avg_out", "mean"))
        prompt_fallback = {pid: (row.avg_in, row.avg_out) for pid, row in pm.iterrows()}

    calls_per_cell = orders * reps * occasions * matches
    assumptions: list[str] = []
    missing_price: list[str] = []
    fell_back: list[str] = []

    detail_rows: list[dict[str, Any]] = []
    for m in models:
        key = m["key"]
        pricing = m.get("pricing")
        if not pricing or pricing.get("input") is None or pricing.get("output") is None:
            missing_price.append(key)
            continue
        pin, pout = float(pricing["input"]), float(pricing["output"])
        for pid in prompt_ids:
            if (key, pid) in lookup:
                avg_in, avg_out = lookup[(key, pid)]
                src = "observed"
            elif pid in prompt_fallback:
                avg_in, avg_out = prompt_fallback[pid]
                src = "fallback(prompt avg)"
                fell_back.append(f"{key}/{pid}")
            else:
                avg_in, avg_out = 0.0, 0.0
                src = "no data"
            cost_per_call = avg_in / 1e6 * pin + avg_out / 1e6 * pout
            detail_rows.append(
                {
                    "model": key,
                    "provider": m.get("provider"),
                    "prompt_id": pid,
                    "avg_in_tokens": round(avg_in, 1),
                    "avg_out_tokens": round(avg_out, 1),
                    "price_in_per_1m": pin,
                    "price_out_per_1m": pout,
                    "usd_per_call": round(cost_per_call, 6),
                    "calls": calls_per_cell,
                    "usd_total": round(cost_per_call * calls_per_cell, 2),
                    "token_source": src,
                }
            )

    detail = pd.DataFrame(detail_rows)
    total_usd = round(float(detail["usd_total"].sum()) if not detail.empty else 0.0, 2)
    total_calls = int(detail["calls"].sum()) if not detail.empty else 0

    per_model = (
        detail.groupby(["provider", "model"], as_index=False)
        .agg(calls=("calls", "sum"), usd_total=("usd_total", "sum"))
        .sort_values("usd_total", ascending=False)
        .reset_index(drop=True)
        if not detail.empty
        else pd.DataFrame()
    )
    per_prompt = (
        detail.groupby("prompt_id", as_index=False)
        .agg(calls=("calls", "sum"), usd_total=("usd_total", "sum"))
        .sort_values("usd_total", ascending=False)
        .reset_index(drop=True)
        if not detail.empty
        else pd.DataFrame()
    )

    assumptions.append(
        f"Grid per occasion per match: {orders} orderings x {len(models)} models "
        f"x {len(prompt_ids)} prompts x {reps} reps."
    )
    assumptions.append(
        "Token usage is the observed average per model x prompt from "
        "fifa_forecasts.db; output = total_tokens - prompt_tokens (includes "
        "reasoning/thinking)."
    )
    assumptions.append(
        "Prices are per config.py: Anthropic authoritative; OpenAI/Gemini/xAI are "
        "ESTIMATES - verify before quoting."
    )
    if fell_back:
        assumptions.append(
            "No observed tokens (used cross-model prompt average) for: "
            + ", ".join(sorted(set(fell_back)))
        )
    if missing_price:
        assumptions.append(
            "EXCLUDED (no price set): " + ", ".join(sorted(set(missing_price)))
        )

    return CostEstimate(
        matches=matches,
        occasions=occasions,
        reps=reps,
        orders=orders,
        total_usd=total_usd,
        per_model=per_model,
        per_prompt=per_prompt,
        detail=detail,
        total_calls=total_calls,
        assumptions=assumptions,
    )


def print_estimate(est: CostEstimate) -> None:
    def fmt(df: pd.DataFrame) -> str:
        if df.empty:
            return "  (none)"
        with pd.option_context("display.width", 200, "display.max_columns", None):
            return df.to_string(index=False)

    print("=" * 72)
    print(
        f"COST ESTIMATE - {est.matches} matches x {est.occasions} occasions "
        f"x {est.orders} orderings x {est.reps} reps"
    )
    print("=" * 72)
    print(f"Total API calls:   {est.total_calls:,}")
    print(f"TOTAL (USD):       ${est.total_usd:,.2f}")
    if est.occasions:
        print(f"  per occasion:    ${est.total_usd / est.occasions:,.2f}")
    if est.matches:
        print(f"  per match (all occasions): ${est.total_usd / est.matches:,.2f}")
    print("\nBy model:")
    print(fmt(est.per_model))
    print("\nBy prompt:")
    print(fmt(est.per_prompt))
    print("\nAssumptions:")
    for a in est.assumptions:
        print(f"- {a}")
