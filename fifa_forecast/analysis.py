"""Reporting & variability analysis over the collected dataset.

This is a *read-only* companion to the collector: it never calls an API. It
loads whatever is already in ``fifa_forecasts.db`` (including a partially
completed run) and answers three questions:

1. **Counts / errors** — how much data exists, success vs error, per model/prompt.
2. **Prompt quality** — how reliably each (model, prompt) yields a valid,
   schema-correct, complete response (JSON validity, probabilities summing to
   100, all six hats present and >= 30 words).
3. **Variability & convergence** — how noisy each model's predictions are across
   repetitions, and how many repetitions per model are needed before the
   running estimate stops moving ("remove the variability").

A *combination* is one (model, match, prompt, team-ordering); its repetitions
are the independent samples whose spread we measure. Two numeric signals are
tracked per combination:

* ``gd``  — goal difference ``score_team_1 - score_team_2`` (every prompt).
* ``p1``  — ``team1_win_probability`` (probability / six-hats prompts only).

Convergence is shown empirically: for each combination we shuffle its
repetitions many times and, for each prefix length ``k``, measure how far the
mean of the first ``k`` samples lands from the combination's full mean. Averaged
over combinations, the curve ``deviation(k)`` flattens as ``k`` grows — the
recommended repetition count is the smallest ``k`` whose deviation drops below a
target margin. A closed-form cross-check (``n = (1.96 * sigma / margin)^2``) is
also reported.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .prompts import HAT_FIELDS, PROMPTS_WITH_HATS, PROMPTS_WITH_PROBABILITIES

COMBO_KEYS = ["model", "match_id", "prompt_id", "team_order_type"]


@dataclass
class ReportOptions:
    prob_margin: float = 5.0  # acceptable +/- on win probability (percentage points)
    gd_margin: float = 0.3  # acceptable +/- on goal difference (goals)
    shuffles: int = 40  # permutations per combination for the convergence curve
    min_reps_for_variability: int = 2
    seed: int = 12345


@dataclass
class Report:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading & feature derivation
# --------------------------------------------------------------------------- #
def load_dataframe(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM forecast_runs", con)
    finally:
        con.close()
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["success"] = df["execution_status"] == "success"
    df["valid"] = df["json_valid"] == 1
    for col in (
        "parsed_score_team_1",
        "parsed_score_team_2",
        "parsed_team1_win_probability",
        "parsed_draw_probability",
        "parsed_team2_win_probability",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gd"] = df["parsed_score_team_1"] - df["parsed_score_team_2"]
    df["p1"] = df["parsed_team1_win_probability"]
    df["prob_sum"] = (
        df["parsed_team1_win_probability"]
        + df["parsed_draw_probability"]
        + df["parsed_team2_win_probability"]
    )
    # winner from the goal difference: 1=team1, 0=draw, -1=team2
    df["winner"] = np.sign(df["gd"])
    return df


def _min_hat_words(row: pd.Series) -> float:
    counts = []
    for hat in HAT_FIELDS:
        text = row.get(hat)
        if isinstance(text, str) and text.strip():
            counts.append(len(text.split()))
        else:
            counts.append(0)
    return float(min(counts)) if counts else 0.0


# --------------------------------------------------------------------------- #
# Count / error / quality tables
# --------------------------------------------------------------------------- #
def _overview(df: pd.DataFrame, opts: ReportOptions) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any) -> None:
        rows.append({"metric": metric, "value": value})

    add("total_executions", len(df))
    add("successful", int(df["success"].sum()) if not df.empty else 0)
    add("errored", int((~df["success"]).sum()) if not df.empty else 0)
    add("valid_json", int(df["valid"].sum()) if not df.empty else 0)
    if not df.empty and df["success"].any():
        add(
            "json_valid_rate_of_success",
            round(df.loc[df["success"], "valid"].mean(), 4),
        )
    add("distinct_models", int(df["model"].nunique()) if not df.empty else 0)
    add("distinct_matches", int(df["match_id"].nunique()) if not df.empty else 0)
    add("distinct_prompts", int(df["prompt_id"].nunique()) if not df.empty else 0)
    add("prob_margin_points", opts.prob_margin)
    add("gd_margin_goals", opts.gd_margin)
    return pd.DataFrame(rows)


def _counts_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + ["n", "success", "error", "valid"])
    g = df.groupby(keys, dropna=False)
    out = g.agg(
        n=("run_id", "size"),
        success=("success", "sum"),
        error=("success", lambda s: int((~s).sum())),
        valid=("valid", "sum"),
    ).reset_index()
    out["error_rate"] = (out["error"] / out["n"]).round(4)
    out["json_valid_rate"] = (out["valid"] / out["n"]).round(4)
    out["avg_latency_ms"] = (
        g["latency_ms"].mean().round(1).reset_index(drop=True)
    )
    out["avg_total_tokens"] = (
        g["total_tokens"].mean().round(1).reset_index(drop=True)
    )
    return out


def _prompt_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, prompt): how well-formed and complete the responses are."""
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model, prompt_id), grp in df.groupby(["model", "prompt_id"], dropna=False):
        n = len(grp)
        succ = grp[grp["success"]]
        valid = grp[grp["valid"]]
        rec: dict[str, Any] = {
            "model": model,
            "prompt_id": prompt_id,
            "n": n,
            "error_rate": round((~grp["success"]).mean(), 4),
            "json_valid_rate": round(grp["valid"].mean(), 4),
        }
        if prompt_id in PROMPTS_WITH_PROBABILITIES and not valid.empty:
            ps = valid["prob_sum"].dropna()
            rec["prob_sum_100_rate"] = (
                round((ps == 100).mean(), 4) if len(ps) else None
            )
            inrange = valid[
                ["parsed_team1_win_probability", "parsed_draw_probability",
                 "parsed_team2_win_probability"]
            ]
            ok = inrange.apply(
                lambda r: r.notna().all() and ((r >= 0) & (r <= 100)).all(), axis=1
            )
            rec["prob_in_range_rate"] = round(ok.mean(), 4) if len(inrange) else None
        if prompt_id in PROMPTS_WITH_HATS and not valid.empty:
            min_words = valid.apply(_min_hat_words, axis=1)
            rec["hats_complete_rate"] = round((min_words > 0).mean(), 4)
            rec["hats_min30_words_rate"] = round((min_words >= 30).mean(), 4)
            rec["avg_min_hat_words"] = round(min_words.mean(), 1)
        rec["avg_completion_tokens"] = (
            round(succ["completion_tokens"].mean(), 1) if not succ.empty else None
        )
        rec["avg_latency_ms"] = (
            round(succ["latency_ms"].mean(), 1) if not succ.empty else None
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "prompt_id"]).reset_index(drop=True)


def _error_signature(message: Any) -> str:
    """Collapse a verbose provider error into a stable, groupable signature.

    Drops the JSON/body and any long tail so that, e.g., dozens of distinct
    429 quota messages collapse to "Gemini API call failed: 429 ...".
    """
    if not isinstance(message, str) or not message.strip():
        return "(no message)"
    head = message.split("{", 1)[0].splitlines()[0].strip()
    return head[:140] if head else message[:140]


def _errors(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["error_signature", "count", "models", "prompts", "sample_message"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    errs = df[~df["success"]].copy()
    if errs.empty:
        return pd.DataFrame(columns=cols)
    errs["error_signature"] = errs["error_message"].map(_error_signature)
    out = (
        errs.groupby("error_signature", dropna=False)
        .agg(
            count=("run_id", "size"),
            models=("model", lambda s: ", ".join(sorted(set(map(str, s))))),
            prompts=("prompt_id", lambda s: ", ".join(sorted(set(map(str, s))))),
            sample_message=("error_message", lambda s: str(s.iloc[0])[:300]),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return out


# --------------------------------------------------------------------------- #
# Variability & convergence
# --------------------------------------------------------------------------- #
def _combination_stats(df: pd.DataFrame, opts: ReportOptions) -> pd.DataFrame:
    """One row per combination, with within-combination spread of gd and p1."""
    valid = df[df["success"] & df["valid"]]
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, grp in valid.groupby(COMBO_KEYS, dropna=False):
        model, match_id, prompt_id, order = keys
        gd = grp["gd"].dropna().to_numpy()
        p1 = grp["p1"].dropna().to_numpy()
        winners = grp["winner"].dropna().to_numpy()
        n = len(grp)
        rec: dict[str, Any] = {
            "model": model,
            "match_id": match_id,
            "prompt_id": prompt_id,
            "team_order_type": order,
            "reps": n,
            "gd_mean": float(np.mean(gd)) if len(gd) else None,
            "gd_std": float(np.std(gd, ddof=1)) if len(gd) > 1 else None,
            "p1_mean": float(np.mean(p1)) if len(p1) else None,
            "p1_std": float(np.std(p1, ddof=1)) if len(p1) > 1 else None,
        }
        if len(winners):
            vals, counts = np.unique(winners, return_counts=True)
            rec["winner_agreement"] = float(counts.max() / counts.sum())
            rec["modal_winner"] = {1.0: "team1", 0.0: "draw", -1.0: "team2"}.get(
                float(vals[counts.argmax()]), "?"
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _variability_by_model(combo: pd.DataFrame, opts: ReportOptions) -> pd.DataFrame:
    if combo.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model, grp in combo.groupby("model", dropna=False):
        usable = grp[grp["reps"] >= opts.min_reps_for_variability]
        rows.append(
            {
                "model": model,
                "combinations": int(len(grp)),
                "combinations_multi_rep": int(len(usable)),
                "max_reps_observed": int(grp["reps"].max()),
                "mean_gd_std": round(grp["gd_std"].mean(), 3)
                if grp["gd_std"].notna().any()
                else None,
                "median_gd_std": round(grp["gd_std"].median(), 3)
                if grp["gd_std"].notna().any()
                else None,
                "mean_p1_std": round(grp["p1_std"].mean(), 2)
                if grp["p1_std"].notna().any()
                else None,
                "median_p1_std": round(grp["p1_std"].median(), 2)
                if grp["p1_std"].notna().any()
                else None,
                "mean_winner_agreement": round(grp["winner_agreement"].mean(), 3)
                if grp["winner_agreement"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _series_per_combo(df: pd.DataFrame, metric: str, opts: ReportOptions) -> dict:
    """For each model, the list of per-combination value arrays for ``metric``."""
    valid = df[df["success"] & df["valid"]]
    out: dict[str, list[np.ndarray]] = {}
    if valid.empty:
        return out
    for keys, grp in valid.groupby(COMBO_KEYS, dropna=False):
        model = keys[0]
        vals = grp.sort_values("repetition_number")[metric].dropna().to_numpy()
        if len(vals) >= opts.min_reps_for_variability:
            out.setdefault(model, []).append(vals)
    return out


def _convergence_curve(
    series_by_model: dict[str, list[np.ndarray]], opts: ReportOptions
) -> pd.DataFrame:
    """Mean |running-mean - full-mean| as a function of k, per model (long format).

    For each combination the repetitions are shuffled ``opts.shuffles`` times so
    the curve does not depend on execution order; the deviation at ``k`` is the
    gap between the mean of the first ``k`` samples and the combination's full
    mean, averaged over all combinations and shuffles.
    """
    rng = np.random.default_rng(opts.seed)
    records: list[dict[str, Any]] = []

    for model, arrays in series_by_model.items():
        max_k = max(len(a) for a in arrays)
        dev_acc: dict[int, list[float]] = {k: [] for k in range(1, max_k + 1)}
        for vals in arrays:
            n = len(vals)
            full_mean = float(vals.mean())
            for _ in range(opts.shuffles):
                perm = rng.permutation(vals)
                csum = np.cumsum(perm)
                for k in range(1, n + 1):
                    dev_acc[k].append(abs(csum[k - 1] / k - full_mean))
        for k in range(1, max_k + 1):
            vlist = dev_acc[k]
            records.append(
                {
                    "model": model,
                    "k": k,
                    "mean_deviation": round(float(np.mean(vlist)), 4)
                    if vlist
                    else math.nan,
                }
            )
    return pd.DataFrame(records)


def _recommend_reps(
    df: pd.DataFrame, combo: pd.DataFrame, opts: ReportOptions
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (recommendation table, gd convergence curve, p1 convergence curve)."""
    gd_series = _series_per_combo(df, "gd", opts)
    p1_series = _series_per_combo(df, "p1", opts)

    gd_curve = _convergence_curve(gd_series, opts)
    p1_curve = _convergence_curve(p1_series, opts)

    def threshold(curve_df: pd.DataFrame, margin: float) -> dict[str, int | None]:
        rec: dict[str, int | None] = {}
        if curve_df.empty:
            return rec
        for model, grp in curve_df.groupby("model"):
            grp = grp.sort_values("k")
            hit = grp[grp["mean_deviation"] <= margin]
            rec[model] = int(hit["k"].min()) if not hit.empty else None
        return rec

    gd_k = threshold(gd_curve, opts.gd_margin)
    p1_k = threshold(p1_curve, opts.prob_margin)

    # closed-form cross-check: n = (1.96 * sigma / margin)^2 using typical sigma
    rows: list[dict[str, Any]] = []
    models = sorted(set(combo["model"]) if not combo.empty else [])
    for model in models:
        grp = combo[combo["model"] == model]
        gd_sigma = grp["gd_std"].median()
        p1_sigma = grp["p1_std"].median()
        gd_formula = (
            int(math.ceil((1.96 * gd_sigma / opts.gd_margin) ** 2))
            if pd.notna(gd_sigma) and gd_sigma > 0
            else (1 if pd.notna(gd_sigma) else None)
        )
        p1_formula = (
            int(math.ceil((1.96 * p1_sigma / opts.prob_margin) ** 2))
            if pd.notna(p1_sigma) and p1_sigma > 0
            else (1 if pd.notna(p1_sigma) else None)
        )
        rec_gd = gd_k.get(model)
        rec_p1 = p1_k.get(model)
        # Headline recommendation uses the CI formula (n = (1.96*sigma/margin)^2),
        # the statistically principled answer; the empirical convergence columns
        # corroborate it but understate, since they compare against a mean drawn
        # from the same small sample.
        candidates = [c for c in (gd_formula, p1_formula) if c is not None]
        overall = max(candidates) if candidates else None
        rows.append(
            {
                "model": model,
                "max_reps_observed": int(grp["reps"].max()) if not grp.empty else 0,
                "median_gd_std": round(gd_sigma, 3) if pd.notna(gd_sigma) else None,
                "median_p1_std": round(p1_sigma, 2) if pd.notna(p1_sigma) else None,
                "reps_for_gd_convergence": rec_gd,
                "reps_for_prob_convergence": rec_p1,
                "reps_gd_formula": gd_formula,
                "reps_prob_formula": p1_formula,
                "recommended_reps": overall,
            }
        )
    rec_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    return rec_df, gd_curve, p1_curve


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_report(config: cfg.Config, opts: ReportOptions | None = None) -> Report:
    opts = opts or ReportOptions()
    df_raw = load_dataframe(cfg.ROOT / config.database_path)
    df = _derive(df_raw)
    report = Report()

    report.tables["Overview"] = _overview(df, opts)
    report.tables["By Model"] = _counts_by(df, ["provider", "model"])
    report.tables["By Prompt"] = _counts_by(df, ["prompt_id"])
    report.tables["By Model x Order"] = _counts_by(df, ["model", "team_order_type"])
    report.tables["Prompt Quality"] = _prompt_quality(df)
    report.tables["Errors"] = _errors(df)

    combo = _combination_stats(df, opts)
    report.tables["Variability by Model"] = _variability_by_model(combo, opts)
    report.tables["Combination Stats"] = combo

    rec_df, gd_curve, p1_curve = _recommend_reps(df, combo, opts)
    report.tables["Recommended Reps"] = rec_df
    report.tables["Convergence (gd)"] = gd_curve
    report.tables["Convergence (prob)"] = p1_curve

    # notes / caveats
    if df.empty:
        report.notes.append("Database is empty — no executions recorded yet.")
    else:
        usable = combo[combo["reps"] >= opts.min_reps_for_variability] if not combo.empty else combo
        report.notes.append(
            f"{0 if combo.empty else len(usable)} combinations have "
            f">= {opts.min_reps_for_variability} repetitions (usable for variability)."
        )
        if not combo.empty and combo["reps"].max() < 10:
            report.notes.append(
                f"Run appears partial: max repetitions observed is "
                f"{int(combo['reps'].max())} (< configured 10). Convergence "
                "estimates beyond that k are extrapolation — re-run the report "
                "once more repetitions land."
            )
        report.notes.append(
            "recommended_reps = max of the CI-formula estimates "
            "n=(1.96*sigma/margin)^2 for goal-difference and win-probability "
            f"(target margins: prob +/-{opts.prob_margin} pts, "
            f"gd +/-{opts.gd_margin} goals). The reps_for_*_convergence columns "
            "show where the empirical running-mean curve first stays within the "
            "same margin; they understate because each running mean is compared "
            "to a full mean drawn from the same small (<=10) sample."
        )
        report.notes.append(
            "Variability is measured per combination (model x match x prompt x "
            "team-order) across repetitions; gd_std/p1_std of 0 means the model "
            "returned an identical scoreline/probability every time."
        )
    return report


def write_report_excel(report: Report, output_path: str | Path | None = None) -> Path:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    target = (
        Path(output_path)
        if output_path
        else (cfg.EXPORTS_DIR / "FIFA_Quality_Variability_Report.xlsx")
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.map(
            lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
        )

    notes_df = pd.DataFrame({"notes": report.notes or ["(none)"]})
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        clean(notes_df).to_excel(writer, sheet_name="Notes", index=False)
        for name, df in report.tables.items():
            clean(df).to_excel(writer, sheet_name=name[:31], index=False)
    return target


def _fmt(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "  (empty)"
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_rows", max_rows
    ):
        return df.to_string(index=False)


def print_report(report: Report) -> None:
    def section(title: str) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    section("OVERVIEW")
    print(_fmt(report.tables.get("Overview", pd.DataFrame())))

    section("COUNTS BY MODEL")
    print(_fmt(report.tables.get("By Model", pd.DataFrame())))

    section("PROMPT QUALITY (per model x prompt)")
    print(_fmt(report.tables.get("Prompt Quality", pd.DataFrame()), max_rows=60))

    section("ERRORS")
    errors = report.tables.get("Errors", pd.DataFrame())
    if not errors.empty:
        errors = errors.drop(columns=["sample_message"], errors="ignore")
    print(_fmt(errors))

    section("VARIABILITY BY MODEL")
    print(_fmt(report.tables.get("Variability by Model", pd.DataFrame())))

    section("RECOMMENDED REPETITIONS PER MODEL")
    print(_fmt(report.tables.get("Recommended Reps", pd.DataFrame())))

    if report.notes:
        section("NOTES")
        for note in report.notes:
            print(f"- {note}")
