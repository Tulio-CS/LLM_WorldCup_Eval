"""Web dashboard for the FIFA forecast collector (FastAPI single-page app).

An operations UI intended to run as the Coolify web service:

* **Overview** — dataset KPIs, the accuracy leaderboard, and the live job log.
* **Run** — launch a collection with full control: pick a single match (or all),
  the moment(s), the models, and the repetition count; dry-run toggle. Plus
  one-click fetch-results / fetch-fixtures / rebuild-report.
* **Results** — the ingested actual scores.
* **Compare** — for any match, what each model predicted vs the real result.

Serve with::

    uvicorn fifa_forecast.web:app --host 0.0.0.0 --port 8000

Security: set ``DASHBOARD_PASSWORD`` (and optionally ``DASHBOARD_USER``, default
``admin``) for HTTP Basic auth. Without a password the UI is read-only — the
money-spending action endpoints are disabled. ``/health`` is always open.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import __version__
from . import config as cfg
from . import prompts as prompt_lib
from .evaluation import build_evaluation, per_match_comparison, write_evaluation_excel
from .export import export_workbook
from .matches import load_matches
from .storage import Database

app = FastAPI(title="FIFA WC2026 Forecast Dashboard", version=__version__)

_DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")
_security = HTTPBasic(auto_error=False)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_view(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    if not _DASHBOARD_PASSWORD:
        return
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username, _DASHBOARD_USER)
        and secrets.compare_digest(creds.password, _DASHBOARD_PASSWORD)
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", {"WWW-Authenticate": "Basic"})


def require_action(creds: HTTPBasicCredentials | None = Depends(_security)) -> None:
    if not _DASHBOARD_PASSWORD:
        raise HTTPException(403, "Actions disabled. Set DASHBOARD_PASSWORD to enable.")
    require_view(creds)


# --------------------------------------------------------------------------- #
# Background job runner (one at a time)
# --------------------------------------------------------------------------- #
_job_lock = threading.Lock()
_JOB: dict[str, Any] = {
    "command": None, "status": "idle", "started_at": None,
    "finished_at": None, "returncode": None, "log_path": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start_job(args: list[str]) -> tuple[bool, str]:
    with _job_lock:
        if _JOB["status"] == "running":
            return False, "A job is already running."
        log_dir = cfg.DATA_DIR / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{stamp}_{args[0]}.log"
        cmd = [sys.executable, "-m", "fifa_forecast", *args]
        _JOB.update(command=" ".join(args), status="running", started_at=_now(),
                    finished_at=None, returncode=None, log_path=str(log_path))

    def _run() -> None:
        rc = -1
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"$ python -m fifa_forecast {' '.join(args)}\n\n")
                fh.flush()
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        cwd=str(cfg.ROOT), env=os.environ.copy())
                rc = proc.wait()
        except Exception as exc:  # noqa: BLE001
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n[runner error] {exc!r}\n")
            except Exception:
                pass
        with _job_lock:
            _JOB.update(status="finished" if rc == 0 else "failed",
                        finished_at=_now(), returncode=rc)

    threading.Thread(target=_run, daemon=True).start()
    return True, "started"


def _job_payload() -> dict[str, Any]:
    job = dict(_JOB)
    log = ""
    path = job.get("log_path")
    if path and Path(path).exists():
        try:
            log = "\n".join(
                Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            )
        except Exception:
            log = ""
    job["log"] = log
    progress = None
    found = re.findall(r"\[(\d+)/(\d+)\]", log)
    if found:
        done, total = found[-1]
        progress = {"done": int(done), "total": int(total)}
    job["progress"] = progress
    return job


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _db_path() -> Path:
    return cfg.ROOT / cfg.load_config().database_path


def _summary() -> dict[str, Any]:
    path = _db_path()
    out: dict[str, Any] = {"db_exists": path.exists()}
    if not path.exists():
        return out
    con = sqlite3.connect(path)
    try:
        def n(sql: str) -> int:
            r = con.execute(sql).fetchone()
            return int(r[0]) if r and r[0] is not None else 0
        out["total"] = n("SELECT COUNT(*) FROM forecast_runs")
        out["success"] = n("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='success'")
        out["error"] = n("SELECT COUNT(*) FROM forecast_runs WHERE execution_status='error'")
        out["valid"] = n("SELECT COUNT(*) FROM forecast_runs WHERE json_valid=1")
        out["matches"] = n("SELECT COUNT(DISTINCT match_id) FROM forecast_runs")
        out["models"] = n("SELECT COUNT(DISTINCT model) FROM forecast_runs")
        try:
            out["results"] = n("SELECT COUNT(*) FROM match_results")
            out["finished"] = n("SELECT COUNT(*) FROM match_results WHERE status='finished'")
        except sqlite3.OperationalError:
            out["results"] = out["finished"] = 0
    finally:
        con.close()
    return out


def _matches_overview(config: cfg.Config) -> list[dict[str, Any]]:
    matches = load_matches(cfg.ROOT / config.matches_csv)
    path = _db_path()
    runs: dict[str, tuple[int, int]] = {}   # match_id -> (n_runs, n_models)
    finished_ids: set[str] = set()
    if path.exists():
        con = sqlite3.connect(path)
        try:
            for mid, n, nm in con.execute(
                "SELECT match_id, COUNT(*), COUNT(DISTINCT model) "
                "FROM forecast_runs GROUP BY match_id"
            ):
                runs[mid] = (int(n), int(nm))
            try:
                finished_ids = {
                    r[0] for r in con.execute(
                        "SELECT match_id FROM match_results WHERE status='finished'"
                    )
                }
            except sqlite3.OperationalError:
                pass
        finally:
            con.close()
    out = []
    for m in matches:
        n_runs, n_models = runs.get(m.match_id, (0, 0))
        out.append({
            "match_id": m.match_id, "team_1": m.team_1, "team_2": m.team_2,
            "local_date": m.local_date, "phase": m.phase,
            "has_predictions": n_runs > 0,
            "n_runs": n_runs, "n_models": n_models,
            "finished": m.match_id in finished_ids,
            "label": f"{m.match_id}. {m.team_1} vs {m.team_2} ({m.local_date})",
        })
    return out


def _results_rows() -> list[dict[str, Any]]:
    path = _db_path()
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        try:
            rows = con.execute(
                "SELECT match_id, team_1, team_2, status, actual_score_team_1, "
                "actual_score_team_2, actual_winner, source FROM match_results "
                "ORDER BY CAST(match_id AS INTEGER)"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# JSON APIs
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@app.get("/api/bootstrap")
def api_bootstrap(_: None = Depends(require_view)) -> JSONResponse:
    config = cfg.load_config()
    models = [
        {"key": m["key"], "provider": m.get("provider"), "model_id": m.get("model_id"),
         "reps": config.reps_for(m["key"])}
        for m in config.enabled_models()
    ]
    return JSONResponse({
        "version": __version__,
        "actions_enabled": bool(_DASHBOARD_PASSWORD),
        "summary": _summary(),
        "moments": list(prompt_lib.MATCH_MOMENTS),
        "default_moments": list(config.match_moments),
        "reps_default": config.runs_per_combination,
        "orders": len(config.team_order_types),
        "models": models,
        "prompts": list(config.prompt_ids),
        "matches": _matches_overview(config),
    })


@app.get("/api/job")
def api_job(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_job_payload())


@app.get("/api/summary")
def api_summary(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_summary())


@app.get("/api/results")
def api_results(_: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse({"results": _results_rows()})


def _jsonable(o: Any) -> Any:
    """Recursively make a value strict-JSON safe (NaN/Inf -> None, numpy -> py)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _jsonable(o.item())
        except Exception:
            return o
    return o


def _records(df: Any) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN/Inf -> None)."""
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in rec.items()
        })
    return out


@app.get("/api/evaluate")
def api_evaluate(_: None = Depends(require_view)) -> JSONResponse:
    config = cfg.load_config()
    try:
        ev = build_evaluation(config)
    except FileNotFoundError:
        return JSONResponse({"leaderboard": [], "by_prompt": [], "notes": ["No database yet."]})
    return JSONResponse({
        "leaderboard": _records(ev.tables.get("Leaderboard (by model)")),
        "by_prompt": _records(ev.tables.get("By Model & Prompt")),
        "notes": ev.notes,
    })


@app.get("/api/compare")
def api_compare(match_id: str, _: None = Depends(require_view)) -> JSONResponse:
    return JSONResponse(_jsonable(per_match_comparison(cfg.load_config(), match_id)))


def _eval_frame():
    """Canonical per-prediction DataFrame (finished matches only), reusing the
    evaluation module's join + canonicalization. Empty frame when nothing joins."""
    from .evaluation import load_joined, _canonicalize

    joined = load_joined(_db_path())
    if joined.empty:
        return joined
    for col, default in (("match_moment", None), ("team_order_type", "original")):
        if col not in joined.columns:
            joined[col] = default
    return _canonicalize(joined)


_BOX_METRICS = ["abs_goal_diff_error", "total_goals_error", "brier"]
_NO_EVAL = "No finished results joined to valid predictions yet — ingest results first."


@app.get("/api/metrics/box")
def api_metrics_box(
    moment: str = "all", prompt: str = "split", group: str = "",
    _: None = Depends(require_view),
) -> JSONResponse:
    """Five-number summaries (Tukey box + outliers) per group, per metric.

    ``prompt``: "split" = one box per model×prompt (default); "agg" = one box per
    model pooling all prompts; or a specific prompt_id to keep only that one.
    ``group="prompt"`` = one box per prompt pooling every model."""
    import numpy as np
    import pandas as pd

    df = _eval_frame()
    if df is None or df.empty:
        return JSONResponse({"groups": [], "metrics": _BOX_METRICS, "note": _NO_EVAL})
    if moment and moment != "all" and "match_moment" in df.columns:
        df = df[df["match_moment"] == moment]
    if prompt not in ("split", "agg", "all", ""):
        df = df[df["prompt_id"] == prompt]
    if df.empty:
        return JSONResponse(
            {"groups": [], "metrics": _BOX_METRICS, "note": "No predictions match this filter."}
        )

    def box(vals) -> dict | None:
        v = pd.to_numeric(vals, errors="coerce").dropna().to_numpy()
        if v.size == 0:
            return None
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inl = v[(v >= lo) & (v <= hi)]
        outl = sorted(float(x) for x in v[(v < lo) | (v > hi)])
        return {
            "n": int(v.size), "min": float(v.min()), "q1": q1, "med": med, "q3": q3,
            "max": float(v.max()), "mean": float(v.mean()),
            "whislo": float(inl.min()) if inl.size else float(v.min()),
            "whishi": float(inl.max()) if inl.size else float(v.max()),
            "outliers": outl[:60],
        }

    if group == "prompt":
        by = ["prompt_id"]
    else:
        by = ["model"] if prompt == "agg" else ["model", "prompt_id"]
    groups = []
    for keys, g in df.groupby(by, dropna=False):
        kv = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        entry = {
            "model": str(kv.get("model", "all models")),
            "prompt_id": str(kv.get("prompt_id", "all")),
            "n": int(len(g)),
        }
        for m in _BOX_METRICS:
            entry[m] = box(g[m]) if m in g.columns else None
        groups.append(entry)
    groups.sort(key=lambda e: (e["model"], e["prompt_id"]))
    return JSONResponse(
        {"groups": groups, "metrics": _BOX_METRICS, "moment": moment, "prompt": prompt, "note": ""}
    )


@app.get("/api/race")
def api_race(
    moment: str = "pre_match", prompt: str = "all", series: str = "model",
    _: None = Depends(require_view),
) -> JSONResponse:
    """Cumulative outcome accuracy across matches ordered by kickoff.

    For each match the group's modal predicted winner is scored against the actual
    outcome; the series is the running fraction correct (step-carried across matches
    the group didn't cover). ``prompt``: "all" pools every prompt into the modal
    vote; a specific prompt_id restricts the vote to that prompt. ``series``:
    "model" (one line per model) or "prompt" (one line per prompt, pooling models)."""
    from collections import Counter

    import pandas as pd

    df = _eval_frame()
    if df is None or df.empty:
        return JSONResponse({"labels": [], "series": {}, "note": _NO_EVAL})
    if moment and moment != "all" and "match_moment" in df.columns:
        sub = df[df["match_moment"] == moment]
        df = sub if not sub.empty else df
    if prompt not in ("all", "agg", ""):
        df = df[df["prompt_id"] == prompt]
    if df.empty:
        return JSONResponse(
            {"labels": [], "series": {}, "note": "No predictions match this filter."}
        )

    order = (
        df.groupby("match_id")["kickoff_datetime"].min().sort_values().index.tolist()
    )
    labels, meta = [], {}
    for i, mid in enumerate(order, start=1):
        g = df[df["match_id"] == mid]
        r = g.iloc[0]
        a1 = pd.to_numeric(g["actual_c1"], errors="coerce").dropna()
        a2 = pd.to_numeric(g["actual_c2"], errors="coerce").dropna()
        actual = f"{int(a1.iloc[0])}-{int(a2.iloc[0])}" if len(a1) and len(a2) else None
        meta[mid] = {
            "name": f"{r['team_1']} v {r['team_2']}",
            "date": str(r.get("kickoff_datetime") or "")[:10],
            "actual": actual,
        }
        labels.append({"i": i, "match_id": str(mid), **meta[mid]})

    key_col = "prompt_id" if series == "prompt" else "model"
    series_out: dict[str, list] = {}
    for key in sorted(x for x in df[key_col].dropna().unique().tolist()):
        md = df[df[key_col] == key]
        per_match: dict[str, int] = {}
        for mid, g in md.groupby("match_id"):
            outs = [o for o in g["pred_outcome"].tolist() if isinstance(o, str)]
            if not outs:
                continue
            modal = Counter(outs).most_common(1)[0][0]
            per_match[mid] = int(modal == g["actual_outcome"].iloc[0])
        pts, correct, total, last = [], 0, 0, None
        for mid in order:
            if mid in per_match:
                total += 1
                correct += per_match[mid]
                last = correct / total
                pts.append({"acc": round(last, 4), "c": per_match[mid]})
            else:
                pts.append({"acc": round(last, 4) if last is not None else None, "c": None})
        series_out[str(key)] = pts
    return JSONResponse(
        {"labels": labels, "series": series_out, "moment": moment, "prompt": prompt,
         "series_by": key_col, "note": ""}
    )


@app.get("/api/analytics")
def api_analytics(
    moment: str = "all", prompt: str = "all", _: None = Depends(require_view)
) -> JSONResponse:
    """One-shot dataset for the Analytics tab: per-model aggregates (accuracy,
    order sensitivity, per-moment accuracy, rep agreement, avg cost), calibration
    bins for probability prompts, and a model×match correctness grid."""
    from collections import Counter

    import numpy as np
    import pandas as pd

    df = _eval_frame()
    if df is None or df.empty:
        return JSONResponse({"models": [], "calibration": {}, "heatmap": None, "note": _NO_EVAL})
    if moment and moment != "all" and "match_moment" in df.columns:
        df = df[df["match_moment"] == moment]
    if prompt not in ("all", "agg", "split", ""):
        df = df[df["prompt_id"] == prompt]
    if df.empty:
        return JSONResponse({"models": [], "calibration": {}, "heatmap": None,
                             "note": "No predictions match this filter."})

    def pct(series) -> float | None:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return round(float(s.mean()) * 100, 1) if len(s) else None

    models_out: list[dict[str, Any]] = []
    for model, g in df.groupby("model"):
        cost = pd.to_numeric(g["api_cost"], errors="coerce").dropna()
        brier = g["brier"].dropna()
        gd = pd.to_numeric(g["abs_goal_diff_error"], errors="coerce").dropna()
        entry: dict[str, Any] = {
            "model": str(model),
            "n": int(len(g)),
            "outcome_acc": pct(g["outcome_correct"]),
            "exact_acc": pct(g["exact_score_correct"]),
            "gd_err": round(float(gd.mean()), 3) if len(gd) else None,
            "brier": round(float(brier.mean()), 4) if len(brier) else None,
            "avg_cost": round(float(cost.mean()), 6) if len(cost) else None,
        }
        for order in ("original", "reversed"):
            sub = g[g["team_order_type"] == order]
            entry[f"acc_{order}"] = pct(sub["outcome_correct"])
        entry["moments"] = {
            str(mm): {"acc": pct(gm["outcome_correct"]), "n": int(len(gm))}
            for mm, gm in g.groupby("match_moment")
        }
        # Rep agreement: within each (match, moment), do all reps pick one winner?
        agree = [
            1 if gg["pred_outcome"].nunique() == 1 else 0
            for _k, gg in g.groupby(["match_id", "match_moment"], dropna=False)
            if len(gg) >= 2
        ]
        entry["agreement"] = round(100 * sum(agree) / len(agree), 1) if agree else None
        entry["agreement_n"] = len(agree)
        models_out.append(entry)
    models_out.sort(key=lambda e: -(e["outcome_acc"] or 0))

    # Calibration: each prediction contributes its 3 normalized outcome
    # probabilities as (predicted p, outcome happened) pairs, binned per decile.
    calibration: dict[str, list] = {}
    p1 = pd.to_numeric(df["pc_team1"], errors="coerce")
    pdr = pd.to_numeric(df["pc_draw"], errors="coerce")
    p2 = pd.to_numeric(df["pc_team2"], errors="coerce")
    psum = p1 + pdr + p2
    cmask = psum.notna() & (psum > 0)
    if cmask.any():
        cdf = df[cmask]
        ps = psum[cmask]
        parts = []
        for col, out in (("pc_team1", "team_1"), ("pc_draw", "draw"), ("pc_team2", "team_2")):
            parts.append(pd.DataFrame({
                "model": cdf["model"],
                "p": pd.to_numeric(cdf[col], errors="coerce") / ps,
                "hit": (cdf["actual_outcome"] == out).astype(float),
            }))
        allp = pd.concat(parts).dropna(subset=["p"])
        if not allp.empty:
            allp["bin"] = np.clip((allp["p"] * 10).astype(int), 0, 9)

            def bins(dd: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"p": round(float(gg["p"].mean()), 3),
                     "obs": round(float(gg["hit"].mean()), 3),
                     "n": int(len(gg))}
                    for _b, gg in dd.groupby("bin")
                ]

            calibration["pooled"] = bins(allp)
            for m, gg in allp.groupby("model"):
                calibration[str(m)] = bins(gg)

    # Heatmap: modal-winner correctness per model × match, kickoff order.
    order_ids = df.groupby("match_id")["kickoff_datetime"].min().sort_values().index.tolist()
    hm_matches = []
    for mid in order_ids:
        g = df[df["match_id"] == mid]
        r = g.iloc[0]
        a1 = pd.to_numeric(g["actual_c1"], errors="coerce").dropna()
        a2 = pd.to_numeric(g["actual_c2"], errors="coerce").dropna()
        hm_matches.append({
            "match_id": str(mid),
            "name": f"{r['team_1']} v {r['team_2']}",
            "date": str(r.get("kickoff_datetime") or "")[:10],
            "actual": f"{int(a1.iloc[0])}-{int(a2.iloc[0])}" if len(a1) and len(a2) else None,
        })
    hm_rows: dict[str, list] = {}
    for model, md in df.groupby("model"):
        per: dict[Any, int] = {}
        for mid, gg in md.groupby("match_id"):
            outs = [o for o in gg["pred_outcome"].tolist() if isinstance(o, str)]
            if outs:
                modal = Counter(outs).most_common(1)[0][0]
                per[mid] = int(modal == gg["actual_outcome"].iloc[0])
        hm_rows[str(model)] = [per.get(mid) for mid in order_ids]

    # Majority-vote accuracy per match — the statistic the race chart uses.
    # It differs from per-prediction accuracy: pooling reps/orders into one
    # modal vote per match can rank models differently.
    for e in models_out:
        cells = [v for v in hm_rows.get(e["model"], []) if v is not None]
        e["modal_acc"] = round(100 * sum(cells) / len(cells), 1) if cells else None

    # Per-prompt aggregates pooling every model (prompt-vs-prompt comparison).
    prompts_out: list[dict[str, Any]] = []
    for pid, g in df.groupby("prompt_id"):
        brier = g["brier"].dropna()
        gd = pd.to_numeric(g["abs_goal_diff_error"], errors="coerce").dropna()
        out_toks = pd.to_numeric(g["completion_tokens"], errors="coerce").dropna()
        think = pd.to_numeric(g["reasoning_tokens"], errors="coerce").dropna()
        cost = pd.to_numeric(g["api_cost"], errors="coerce").dropna()
        modal_hits = []
        for mid, gg in g.groupby("match_id"):
            outs = [o for o in gg["pred_outcome"].tolist() if isinstance(o, str)]
            if outs:
                modal = Counter(outs).most_common(1)[0][0]
                modal_hits.append(int(modal == gg["actual_outcome"].iloc[0]))
        prompts_out.append({
            "prompt_id": str(pid),
            "n": int(len(g)),
            "outcome_acc": pct(g["outcome_correct"]),
            "exact_acc": pct(g["exact_score_correct"]),
            "modal_acc": round(100 * sum(modal_hits) / len(modal_hits), 1) if modal_hits else None,
            "gd_err": round(float(gd.mean()), 3) if len(gd) else None,
            "brier": round(float(brier.mean()), 4) if len(brier) else None,
            "avg_out_tokens": round(float(out_toks.mean()), 0) if len(out_toks) else None,
            "avg_think_tokens": round(float(think.mean()), 0) if len(think) else None,
            "avg_cost": round(float(cost.mean()), 6) if len(cost) else None,
        })
    prompts_out.sort(key=lambda e: -(e["outcome_acc"] or 0))

    # Usage stats over ALL successful calls (not just evaluated ones), same filters.
    usage: dict[str, Any] = {}
    try:
        con = sqlite3.connect(_db_path())
        where = "execution_status='success'"
        params: list[Any] = []
        if moment and moment != "all":
            where += " AND match_moment=?"
            params.append(moment)
        if prompt not in ("all", "agg", "split", ""):
            where += " AND prompt_id=?"
            params.append(prompt)
        row = con.execute(
            f"SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
            f"SUM(reasoning_tokens), SUM(api_cost), AVG(latency_ms) "
            f"FROM forecast_runs WHERE {where}", params,
        ).fetchone()
        usage = {
            "calls": int(row[0] or 0),
            "tokens_in": int(row[1] or 0),
            "tokens_out": int(row[2] or 0),
            "tokens_think": int(row[3] or 0),
            "cost": round(float(row[4] or 0), 4),
            "avg_latency_ms": round(float(row[5]), 0) if row[5] is not None else None,
        }
        usage["by_model"] = [
            {"model": r[0], "calls": int(r[1] or 0), "tokens_in": int(r[2] or 0),
             "tokens_out": int(r[3] or 0), "tokens_think": int(r[4] or 0),
             "cost": round(float(r[5] or 0), 4)}
            for r in con.execute(
                f"SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
                f"SUM(reasoning_tokens), SUM(api_cost) FROM forecast_runs "
                f"WHERE {where} GROUP BY model", params,
            ).fetchall()
        ]
        con.close()
    except Exception:  # pragma: no cover - usage cards are best-effort
        usage = {}

    return JSONResponse(_jsonable({
        "models": models_out,
        "prompts": prompts_out,
        "usage": usage,
        "calibration": calibration,
        "heatmap": {"matches": hm_matches, "rows": hm_rows},
        "moment": moment, "prompt": prompt, "note": "",
    }))


_FORECAST_COLS = [
    "run_id", "match_id", "team_1", "team_2", "provider", "model", "prompt_id",
    "match_moment", "team_order_type", "repetition_number",
    "parsed_score_team_1", "parsed_score_team_2",
    "parsed_team1_win_probability", "parsed_draw_probability", "parsed_team2_win_probability",
    "json_valid", "execution_status", "latency_ms", "total_tokens", "api_cost",
]


@app.get("/api/forecasts")
def api_forecasts(
    match_id: str | None = None, model: str | None = None, prompt_id: str | None = None,
    moment: str | None = None, order: str | None = None, status: str | None = None,
    valid: str | None = None, q: str | None = None, limit: int = 50, offset: int = 0,
    _: None = Depends(require_view),
) -> JSONResponse:
    """Filtered, paginated view of the forecast_runs table."""
    path = _db_path()
    if not path.exists():
        return JSONResponse({"total": 0, "rows": [], "limit": limit, "offset": offset})
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(forecast_runs)")}
        where: list[str] = []
        params: list[Any] = []

        def add(cond: str, val: Any) -> None:
            where.append(cond)
            params.append(val)

        if match_id:
            add("match_id = ?", match_id)
        if model:
            add("model = ?", model)
        if prompt_id:
            add("prompt_id = ?", prompt_id)
        if moment and "match_moment" in cols:
            add("match_moment = ?", moment)
        if order:
            add("team_order_type = ?", order)
        if status:
            add("execution_status = ?", status)
        if valid in ("0", "1"):
            add("json_valid = ?", int(valid))
        if q:
            where.append("(team_1 LIKE ? OR team_2 LIKE ? OR model LIKE ? OR prompt_id LIKE ?)")
            like = f"%{q}%"
            params += [like, like, like, like]
        wsql = (" WHERE " + " AND ".join(where)) if where else ""

        total = con.execute(f"SELECT COUNT(*) FROM forecast_runs{wsql}", params).fetchone()[0]
        sel = [c for c in _FORECAST_COLS if c in cols]
        lim = max(1, min(int(limit), 500))
        off = max(0, int(offset))
        rows = con.execute(
            f"SELECT {', '.join(sel)} FROM forecast_runs{wsql} "
            "ORDER BY CAST(match_id AS INTEGER), run_id LIMIT ? OFFSET ?",
            params + [lim, off],
        ).fetchall()
        return JSONResponse(_jsonable({
            "total": int(total), "limit": lim, "offset": off,
            "rows": [dict(r) for r in rows],
        }))
    finally:
        con.close()


@app.get("/api/forecast/{run_id}")
def api_forecast_detail(run_id: str, _: None = Depends(require_view)) -> JSONResponse:
    """Full row (incl. prompt text, raw response, six hats) for one execution."""
    path = _db_path()
    if not path.exists():
        raise HTTPException(404, "no database")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM forecast_runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(404, "not found")
    return JSONResponse(_jsonable(dict(r)))


# --------------------------------------------------------------------------- #
# Actions (JSON body)
# --------------------------------------------------------------------------- #
class RunReq(BaseModel):
    match_id: str | None = None     # single, kept for backward compatibility
    match_ids: list[str] = []       # run several matches at once ([] = all)
    moments: list[str] = []
    models: list[str] = []
    reps: int | None = None         # override for every model (None = per-model)
    retry_errors: bool = False      # re-run errored + missing; keep successes
    dry_run: bool = False


class FetchReq(BaseModel):
    mode: str = "results"   # results | fixtures | both
    dry_run: bool = False


@app.post("/actions/run")
def action_run(req: RunReq, _: None = Depends(require_action)) -> JSONResponse:
    args = ["run", "--no-export"]
    ids = list(req.match_ids)
    if req.match_id and req.match_id not in ids:
        ids.append(req.match_id)
    for mid in ids:
        args += ["--match-id", mid]
    for m in req.moments:
        if m in prompt_lib.MATCH_MOMENTS:
            args += ["--moment", m]
    for k in req.models:
        args += ["--model", k]
    if req.reps:
        args += ["--reps", str(req.reps)]
    if req.retry_errors:
        args.append("--retry-errors")
    if req.dry_run:
        args.append("--dry-run")
    ok, msg = start_job(args)
    return JSONResponse({"ok": ok, "message": msg, "command": " ".join(args)},
                        status_code=200 if ok else 409)


@app.post("/actions/fetch")
def action_fetch(req: FetchReq, _: None = Depends(require_action)) -> JSONResponse:
    args = ["fetch"]
    if req.mode in ("results", "both"):
        args.append("--results")
    if req.mode in ("fixtures", "both"):
        args.append("--fixtures")
    if req.dry_run:
        args.append("--dry-run")
    ok, msg = start_job(args)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/actions/report")
def action_report(_: None = Depends(require_action)) -> JSONResponse:
    ok, msg = start_job(["report"])
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/actions/merge-upload")
async def action_merge_upload(
    file: UploadFile = File(...),
    replace: bool = Form(False),
    _: None = Depends(require_action),
) -> JSONResponse:
    """Merge a forecast .db uploaded from another machine into the server DB.

    This is how locally-run models (Ollama on your PC) reach the Coolify DB:
    run the collection locally into a scratch .db, then upload it here. Rows are
    added by deterministic run_id (INSERT OR IGNORE), so cloud rows are never
    clobbered; ``replace=true`` overwrites same-id rows instead.
    """
    up_dir = cfg.DATA_DIR / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tmp = up_dir / f"merge_{stamp}.db"
    data = await file.read()
    tmp.write_bytes(data)
    try:
        if data[:16] != b"SQLite format 3\x00":
            raise HTTPException(400, "Uploaded file is not a SQLite database.")
        db = Database(_db_path())
        try:
            stats = db.merge_from(tmp, replace=replace)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise HTTPException(400, f"Merge failed: {exc}")
    finally:
        try:
            tmp.unlink()
        except OSError:  # pragma: no cover
            pass
    return JSONResponse({"ok": True, "filename": file.filename, **stats})


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
@app.get("/download/forecasts")
def download_forecasts(_: None = Depends(require_view)) -> FileResponse:
    path = export_workbook(cfg.load_config())
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/evaluation")
def download_evaluation(_: None = Depends(require_view)) -> FileResponse:
    ev = build_evaluation(cfg.load_config())
    path = write_evaluation_excel(ev)
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/report")
def download_report(_: None = Depends(require_view)) -> FileResponse:
    from .analysis import ReportOptions, build_report, write_report_excel

    rep = build_report(cfg.load_config(), ReportOptions())
    path = write_report_excel(rep)
    return FileResponse(path, filename=Path(path).name)


@app.get("/download/archive")
def download_archive(_: None = Depends(require_view)) -> FileResponse:
    """Zip the SQLite DB (consistent snapshot) plus every raw JSON artifact.

    The DB is copied via SQLite's online backup API so it's safe to download
    even while a collection job is writing. The zip is streamed and deleted
    afterwards. JSON lives under data/{raw_requests,raw_responses,traces,
    metadata}; the manifest sits at the repo root.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip", dir=str(cfg.DATA_DIR))
    os.close(fd)
    zip_path = Path(tmp_zip)

    db_path = _db_path()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # DB: online-backup snapshot into a temp file, then add to the zip.
        if db_path.exists():
            fd2, tmp_db = tempfile.mkstemp(suffix=".db", dir=str(cfg.DATA_DIR))
            os.close(fd2)
            try:
                src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                dst = sqlite3.connect(tmp_db)
                with dst:
                    src.backup(dst)
                dst.close()
                src.close()
                zf.write(tmp_db, arcname=db_path.name)
            finally:
                try:
                    os.unlink(tmp_db)
                except OSError:  # pragma: no cover
                    pass
        # All raw JSON artifacts, preserving their folder names in the zip.
        for d in (
            cfg.RAW_REQUESTS_DIR,
            cfg.RAW_RESPONSES_DIR,
            cfg.TRACES_DIR,
            cfg.METADATA_DIR,
        ):
            if d.exists():
                for f in sorted(d.glob("*.json")):
                    zf.write(f, arcname=f"{d.name}/{f.name}")
        # Experiment manifest (JSON at the repo root).
        manifest = cfg.ROOT / "experiment_manifest.json"
        if manifest.exists():
            zf.write(manifest, arcname=manifest.name)

    return FileResponse(
        zip_path,
        filename=f"fifa_forecast_backup_{stamp}.zip",
        media_type="application/zip",
        background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
    )


# --------------------------------------------------------------------------- #
# Single-page app
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def home(_: None = Depends(require_view)) -> HTMLResponse:
    return HTMLResponse(_SPA_HTML)


_SPA_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FIFA WC2026 — Forecast Console</title>
<style>
:root{
  --bg:#0b0f17; --panel:#121a28; --panel2:#0e1521; --border:#22304a; --border2:#2c3d5c;
  --txt:#e8ecf4; --muted:#8a99b5; --accent:#3b82f6; --accent2:#1e4ed8;
  --ok:#22c55e; --warn:#f59e0b; --err:#ef4444;
}
*{box-sizing:border-box} html,body{margin:0}
body{background:radial-gradient(1200px 600px at 80% -10%,#16233b 0%,var(--bg) 55%);
  color:var(--txt);font:14px/1.45 system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;min-height:100vh}
a{color:#7eb0ff;text-decoration:none} a:hover{text-decoration:underline}
.top{position:sticky;top:0;z-index:10;backdrop-filter:blur(8px);
  background:#0b0f17cc;border-bottom:1px solid var(--border);padding:12px 20px;
  display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{font-weight:700;font-size:16px;letter-spacing:.2px}
.brand small{color:var(--muted);font-weight:400;margin-left:8px}
.kpis{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
.kpi{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1}
.kpi b{font-size:16px} .kpi span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.card{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:104px}
.card b{font-size:16px;display:block;white-space:nowrap}
.card span{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.4px}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--border2)}
.b-idle{color:var(--muted)} .b-running{color:#0b0f17;background:var(--warn);border-color:var(--warn)}
.b-finished{color:#0b0f17;background:var(--ok);border-color:var(--ok)} .b-failed{color:#fff;background:var(--err);border-color:var(--err)}
.tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid var(--border);background:transparent}
.tab{padding:12px 16px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;font-weight:600}
.tab:hover{color:var(--txt)} .tab.active{color:#fff;border-bottom-color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
.panel{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
  border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:18px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin:0 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.card{background:#0e1521;border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.card b{font-size:24px;display:block} .card span{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:760px){.grid2{grid-template-columns:1fr}}
label{display:block;font-size:12px;color:var(--muted);margin:0 0 6px;text-transform:uppercase;letter-spacing:.4px}
select,input[type=number],input[type=text]{width:100%;background:#0b1220;border:1px solid var(--border2);
  color:var(--txt);border-radius:8px;padding:9px 10px;font-size:14px}
.checkset{display:flex;flex-wrap:wrap;gap:8px}
.chk{display:flex;align-items:center;gap:7px;background:#0b1220;border:1px solid var(--border2);
  border-radius:8px;padding:7px 11px;cursor:pointer;user-select:none;font-size:13px}
.chk input{accent-color:var(--accent)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:14px}
.row>div{flex:1;min-width:160px}
.btn{background:var(--accent);border:0;color:#fff;border-radius:9px;padding:10px 16px;font-weight:700;
  cursor:pointer;font-size:14px} .btn:hover{background:var(--accent2)} .btn:disabled{opacity:.45;cursor:not-allowed}
.btn.warn{background:var(--err)} .btn.ghost{background:#1c2840} .btn.ok{background:var(--ok);color:#06210f}
.btnrow{display:flex;gap:10px;flex-wrap:wrap}
.muted{color:var(--muted)} .small{font-size:12px}
.progress{height:8px;background:#0b1220;border-radius:999px;overflow:hidden;border:1px solid var(--border)}
.progress>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#22d3ee);width:0%}
pre.log{background:#06090f;border:1px solid var(--border);border-radius:10px;padding:12px;
  font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;max-height:300px;overflow:auto;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;background:var(--panel)}
tr:hover td{background:#0e1726}
.tablewrap{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:10px}
.pill{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid var(--border2)}
.pill.ok{color:#06210f;background:var(--ok);border-color:var(--ok)}
.pill.bad{color:#fff;background:var(--err);border-color:var(--err)}
.pill.na{color:var(--muted)}
.banner{background:#3a2a0a;border:1px solid #6b4e16;color:#f7d99a;padding:10px 14px;border-radius:10px;margin-bottom:16px}
.hide{display:none!important}
.toast{position:fixed;right:18px;bottom:18px;background:#0e1726;border:1px solid var(--border2);
  padding:12px 16px;border-radius:10px;max-width:360px;box-shadow:0 8px 30px #0008}
.section-actions{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
.picklist{max-height:220px;overflow:auto;border:1px solid var(--border2);border-radius:8px;padding:8px;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:4px}
.clickrow td{cursor:pointer} .clickrow:hover td{background:#16233b}
.modal{position:fixed;inset:0;background:#000a;display:flex;align-items:flex-start;justify-content:center;
  padding:40px 16px;z-index:50;overflow:auto}
.modalbox{background:var(--panel);border:1px solid var(--border2);border-radius:14px;max-width:860px;width:100%;padding:20px}
.kv{display:grid;grid-template-columns:170px 1fr;gap:5px 14px;font-size:13px;margin:8px 0}
.kv .k{color:var(--muted)} .x{float:right;cursor:pointer;color:var(--muted);font-size:18px}
</style></head>
<body>
<div class="top">
  <div class="brand">⚽ FIFA WC2026 <small>Forecast Console</small></div>
  <div class="kpis" id="kpis"></div>
  <span id="jobBadge" class="badge b-idle">idle</span>
</div>
<div class="tabs" id="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="run">Run</div>
  <div class="tab" data-tab="results">Results</div>
  <div class="tab" data-tab="compare">Compare</div>
  <div class="tab" data-tab="forecasts">Forecasts</div>
  <div class="tab" data-tab="analytics">Analytics</div>
  <div class="tab" data-tab="downloads">Downloads</div>
</div>
<div class="wrap">
  <div id="roBanner" class="banner hide">Read-only mode — set <b>DASHBOARD_PASSWORD</b> to enable the Run / Fetch buttons.</div>

  <!-- OVERVIEW -->
  <section id="t-overview">
    <div class="panel"><h2>Dataset</h2><div class="cards" id="ovCards"></div></div>
    <div class="panel"><h2>Accuracy leaderboard (finished matches)</h2><div id="ovLeader" class="tablewrap"></div>
      <p class="muted small" id="ovNotes"></p></div>
    <div class="panel"><h2>Current job</h2>
      <div id="ovJobMeta" class="small muted" style="margin-bottom:8px"></div>
      <div class="progress"><i id="ovBar"></i></div>
      <pre class="log" id="ovLog">(no job yet)</pre>
    </div>
  </section>

  <!-- RUN -->
  <section id="t-run" class="hide">
    <div class="panel">
      <h2>Run a collection</h2>
      <div style="margin-bottom:14px">
        <label>Matches — check one or many (none = all)</label>
        <div class="small" id="matchCoverage" style="margin-bottom:8px"></div>
        <input type="text" id="runMatchSearch" placeholder="filter by team, date or phase…" style="margin-bottom:8px">
        <label class="chk" style="display:inline-flex;margin-bottom:8px"><input type="checkbox" id="hideRun"> show only not-run yet</label>
        <div id="runMatchList" class="picklist"></div>
        <div class="small muted" style="margin-top:6px">
          <a href="#" id="matchAll">all</a> · <a href="#" id="matchNone">none</a> ·
          <a href="#" id="matchNotRun">select not-run</a> · <a href="#" id="matchShown">select shown</a> · <b id="matchCount"></b></div>
      </div>
      <div style="margin-bottom:14px"><label>Moments</label><div class="checkset" id="runMoments"></div></div>
      <div style="margin-bottom:14px"><label>Models</label><div class="checkset" id="runModels"></div>
        <div class="small muted" style="margin-top:6px"><a href="#" id="modelsAll">select all</a> · <a href="#" id="modelsNone">none</a></div></div>
      <div class="row"><div style="max-width:220px"><label>Reps override (blank = per-model)</label><input type="number" id="runReps" min="1" max="50" placeholder="per model"></div></div>
      <label class="chk" style="display:inline-flex;margin-bottom:10px"><input type="checkbox" id="runRetry"> only re-run failed &amp; missing (keep what already succeeded)</label>
      <div class="section-actions">
        <label class="chk"><input type="checkbox" id="runDry"> dry-run (free, mock)</label>
        <button class="btn warn" id="btnRun">▶ Run</button>
        <span class="small muted" id="runEstimate"></span>
      </div>
    </div>
    <div class="panel">
      <h2>Maintenance</h2>
      <div class="btnrow">
        <button class="btn ghost" data-fetch="results">⬇ Fetch results</button>
        <button class="btn ghost" data-fetch="fixtures">⬇ Fetch fixtures</button>
        <button class="btn ghost" data-fetch="both">⬇ Fetch both</button>
        <button class="btn ghost" id="btnReport">📊 Rebuild report</button>
      </div>
    </div>
    <div class="panel"><h2>Live job</h2>
      <div id="runJobMeta" class="small muted" style="margin-bottom:8px"></div>
      <div class="progress"><i id="runBar"></i></div>
      <pre class="log" id="runLog">(no job yet)</pre>
    </div>
  </section>

  <!-- RESULTS -->
  <section id="t-results" class="hide">
    <div class="panel"><h2>Actual results</h2><div id="resTable" class="tablewrap"></div></div>
  </section>

  <!-- COMPARE -->
  <section id="t-compare" class="hide">
    <div class="panel">
      <h2>Model forecasts vs actual result</h2>
      <div class="row"><div><label>Match</label><select id="cmpMatch"></select></div></div>
      <div id="cmpInfo" class="small" style="margin-bottom:12px"></div>
      <div id="cmpTable" class="tablewrap"></div>
    </div>
  </section>

  <!-- FORECASTS -->
  <section id="t-forecasts" class="hide">
    <div class="panel">
      <h2>Forecast runs — filterable table</h2>
      <div class="row">
        <div><label>Match</label><select id="fxMatch"></select></div>
        <div><label>Model</label><select id="fxModel"></select></div>
        <div><label>Prompt</label><select id="fxPrompt"></select></div>
        <div><label>Moment</label><select id="fxMoment"></select></div>
      </div>
      <div class="row">
        <div><label>Order</label><select id="fxOrder"><option value="">any</option><option>original</option><option>reversed</option></select></div>
        <div><label>Status</label><select id="fxStatus"><option value="">any</option><option>success</option><option>error</option></select></div>
        <div><label>JSON</label><select id="fxValid"><option value="">any</option><option value="1">valid</option><option value="0">invalid</option></select></div>
        <div><label>Search</label><input type="text" id="fxQ" placeholder="team / model…"></div>
      </div>
      <div class="section-actions" style="margin-bottom:10px">
        <button class="btn ghost" id="fxApply">Apply</button>
        <button class="btn ghost" id="fxReset">Reset</button>
        <span class="small muted" id="fxCount"></span>
        <span class="small muted" style="margin-left:auto">click a row → prompt + raw response</span>
      </div>
      <div id="fxTable" class="tablewrap"></div>
      <div class="section-actions">
        <button class="btn ghost" id="fxPrev">‹ Prev</button>
        <span class="small muted" id="fxPage"></span>
        <button class="btn ghost" id="fxNext">Next ›</button>
      </div>
    </div>
  </section>

  <!-- ANALYTICS -->
  <section id="t-analytics" class="hide">
    <div class="panel">
      <h2>Analytics — model performance on finished matches</h2>
      <div class="row">
        <div><label>Prompt</label><select id="anPrompt"></select></div>
        <div><label>Moment</label><select id="anMomentG">
          <option value="all">all</option><option value="pre_match">pre_match</option>
          <option value="halftime">halftime</option><option value="post_match">post_match</option>
        </select></div>
        <div style="align-self:end"><button class="btn ghost" id="anReload">Reload</button></div>
      </div>
      <p class="small muted">Filters apply to every chart below. Predictions are mapped to canonical
        team order; only finished matches with ingested results count. <span id="anNote"></span></p>
      <div id="anKpis" class="cards"></div>
    </div>

    <div class="panel">
      <h2>Leaderboard — outcome &amp; exact-score accuracy</h2>
      <p class="small muted">Solid bar = <b>per-prediction</b> accuracy (every rep/order counts);
        ◇ = <b>majority-vote</b> accuracy per match — the statistic the race uses, so the two can
        rank differently; thin bar = exact scoreline. Sorted by per-prediction accuracy.</p>
      <div id="chLeader"></div>
    </div>

    <div class="panel">
      <h2>Prompt comparison — all models aggregated</h2>
      <p class="small muted">One row per prompt strategy, pooling every model: does the prompt style
        itself change forecast quality (and at what token cost)?</p>
      <div id="chPrompts"></div>
    </div>

    <div class="panel">
      <h2>Accuracy race — cumulative outcome accuracy over time</h2>
      <div class="row"><div><label>Series</label><select id="raceSeries">
        <option value="model">Models</option><option value="prompt">Prompts (all models pooled)</option>
      </select></div></div>
      <p class="small muted">Matches ordered by kickoff; each match scores the group's modal predicted winner. Hit play to watch the ranking evolve.</p>
      <div id="raceWrap"></div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px">
      <div class="panel"><h2>Cost vs accuracy</h2>
        <p class="small muted">Avg API cost per call (√ scale) vs outcome accuracy — up-left is the value corner. Free local models sit on the axis.</p>
        <div id="chCost"></div></div>
      <div class="panel"><h2>Rep consistency</h2>
        <p class="small muted">How often all repetitions of a match agree on the winner — higher = more deterministic model.</p>
        <div id="chAgree"></div></div>
      <div class="panel"><h2>Team-order sensitivity</h2>
        <p class="small muted">Accuracy with teams presented in original vs reversed order. A wide gap = the model is biased by presentation order.</p>
        <div id="chOrder"></div></div>
      <div class="panel"><h2>Accuracy by moment</h2>
        <p class="small muted">Does in-game information (halftime / post-match) actually improve the forecast?</p>
        <div id="chMoment"></div></div>
    </div>

    <div class="panel">
      <h2>Calibration — predicted probability vs reality</h2>
      <div class="row"><div><label>Model</label><select id="calibModel"></select></div></div>
      <p class="small muted">Probability prompts only. Each prediction contributes its three outcome
        probabilities; a well-calibrated model hugs the diagonal. Dot size = sample count.</p>
      <div id="chCalib"></div>
    </div>

    <div class="panel">
      <h2>Usage — tokens &amp; cost by model</h2>
      <p class="small muted">All successful calls under the current filter (not just evaluated ones).
        Stacked: <span style="color:#94a3b8">input</span> ·
        <span style="color:var(--accent)">output</span> ·
        <span style="color:#a855f7">thinking</span> tokens. Cost is the recorded api_cost
        (thinking billed separately on Gemini is not included).</p>
      <div id="chUsage"></div>
    </div>

    <div class="panel">
      <h2>Metric distributions — boxplots</h2>
      <div class="row">
        <div><label>Metric</label><select id="boxMetric">
          <option value="abs_goal_diff_error">Goal-difference error</option>
          <option value="total_goals_error">Total-goals error</option>
          <option value="brier">Brier (probability prompts)</option>
        </select></div>
        <div><label>Group by</label><select id="boxGroup">
          <option value="model_prompt">model × prompt</option>
          <option value="model">model (prompts pooled)</option>
          <option value="prompt">prompt (models pooled)</option>
        </select></div>
      </div>
      <p class="small muted">Lower is better; red dots are outliers.</p>
      <div id="boxWrap" class="tablewrap"></div>
    </div>

    <div class="panel">
      <h2>Match grid — who got which game right</h2>
      <p class="small muted">One column per finished match (kickoff order). Green = modal winner correct,
        red = wrong, gray = not run. Hover a cell for details. Rows sorted by hit rate.</p>
      <div id="chHeat" style="overflow-x:auto"></div>
    </div>
  </section>

  <!-- DOWNLOADS -->
  <section id="t-downloads" class="hide">
    <div class="panel"><h2>Excel workbooks</h2>
      <p><a href="./download/evaluation">⬇ Evaluation report (forecast vs results)</a></p>
      <p><a href="./download/forecasts">⬇ Full forecasts workbook</a></p>
      <p><a href="./download/report">⬇ Quality / variability report</a></p>
    </div>
    <div class="panel"><h2>Full backup (.zip)</h2>
      <p class="muted small">The complete raw dataset in one archive: the SQLite
        database plus every JSON artifact (requests, responses, traces, metadata)
        and the manifest. The DB is a consistent snapshot — safe to grab mid-run.</p>
      <p><a href="./download/archive">⬇ Download database + all JSON (zip)</a></p>
    </div>
    <div class="panel"><h2>Import local runs</h2>
      <p class="muted small">Ran open models (Qwen / Mistral / Gemma) on your own machine?
        Upload the resulting <code>.db</code> file to merge those forecasts into this
        server's database. Existing rows are kept (matched by run_id); tick replace to
        overwrite same-id rows.</p>
      <p><input type="file" id="mergeFile" accept=".db,.sqlite,application/octet-stream"></p>
      <label class="small"><input type="checkbox" id="mergeReplace"> replace rows with the same run_id</label>
      <p><button class="btn" id="btnMerge">⬆ Upload &amp; merge</button></p>
      <div id="mergeMsg" class="muted small"></div>
    </div>
  </section>
</div>
<div id="fxModal" class="modal hide"><div class="modalbox" id="fxModalBox"></div></div>
<div id="toast" class="toast hide"></div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let BOOT=null, POLL=null, lastStatus=null;
let ANALYTICS={box:null,race:null}, RACE=null;
const PALETTE=['#3b82f6','#22c55e','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6','#eab308','#8b5cf6','#64748b','#f43f5e'];
const pcolor=p=>({'simple-prediction':'#06b6d4','probability-prediction':'#22c55e','six-hats-prediction':'#f59e0b'}[p]||'#3b82f6');
const fmtPct=v=>v==null?'–':(v+'%');
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function toast(msg,ms=3500){const t=$('#toast');t.textContent=msg;t.classList.remove('hide');
  clearTimeout(t._t);t._t=setTimeout(()=>t.classList.add('hide'),ms);}

function tab(name){
  $$('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  ['overview','run','results','compare','forecasts','analytics','downloads'].forEach(n=>$('#t-'+n).classList.toggle('hide',n!==name));
  if(name==='results')loadResults();
  if(name==='compare')loadCompare();
  if(name==='forecasts')loadForecasts();
  if(name==='overview')loadEvaluate();
  if(name==='analytics')loadAnalytics();
}
$$('.tab').forEach(t=>t.onclick=()=>tab(t.dataset.tab));

// ---- Analytics ----------------------------------------------------------
let AN={sum:null};
let MODEL_COLOR={};
// Fixed-pixel SVGs (no viewBox stretching) keep text crisp and consistently sized.
const svgOpen=(W,H)=>`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="font:12px system-ui;max-width:100%;height:auto;display:block">`;
const MOMENT_COLORS={pre_match:'#3b82f6',halftime:'#f59e0b',post_match:'#22c55e'};
function modelColor(m){return MODEL_COLOR[m]||'#64748b';}
function chartW(el,max){
  let w=(el&&el.clientWidth)||0;
  if(w<100)w=(el&&el.parentElement&&el.parentElement.clientWidth)||0;  // transient 0-width guard
  if(w<100)w=720;
  return Math.max(340,Math.min(max||980,w-2));
}
function labelPad(names){const L=Math.max(6,...names.map(s=>String(s).length));return Math.min(190,Math.max(90,14+L*7.3));}
function shortName(s,n){n=n||22;s=String(s);return s.length>n?s.slice(0,n-1)+'…':s;}
function noData(sel,msg){$(sel).innerHTML='<div class="small muted">'+esc(msg||(AN.sum&&AN.sum.note)||'No data yet — ingest finished results first.')+'</div>';}
function fmtCost(c){return c==null?'free':('$'+(c<0.01?c.toFixed(4):c.toFixed(3)));}
function fmtK(v){return v==null?'–':(v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'k':String(Math.round(v)));}
function pct1(v){return v==null?'–':((+v).toFixed(1).replace(/\.0$/,'')+'%');}
function anQ(){return `prompt=${encodeURIComponent($('#anPrompt').value||'all')}&moment=${encodeURIComponent($('#anMomentG').value||'all')}`;}
function boxQuery(){
  const p=$('#anPrompt').value||'all', g=$('#boxGroup').value||'model_prompt';
  const m=`moment=${encodeURIComponent($('#anMomentG').value||'all')}`;
  if(g==='prompt')return `group=prompt&prompt=${encodeURIComponent(p)}&`+m;
  if(p!=='all')return `prompt=${encodeURIComponent(p)}&`+m;
  return `prompt=${g==='model'?'agg':'split'}&`+m;
}
function raceQ(){return anQ()+'&series='+encodeURIComponent($('#raceSeries').value||'model');}
function fillAnFilters(){const sel=$('#anPrompt');if(sel.dataset.filled)return;
  const ps=(BOOT&&BOOT.prompts)||[];
  sel.innerHTML='<option value="all">All prompts</option>'+ps.map(p=>`<option value="${esc(p)}">${esc(p.replace('-prediction',''))}</option>`).join('');
  sel.dataset.filled='1';}
async function loadBoxOnly(){
  try{ANALYTICS.box=await api('./api/metrics/box?'+boxQuery());renderBox();}
  catch(e){$('#boxWrap').innerHTML='<div class="small muted">Error: '+esc(e.message)+'</div>';}}
async function loadRaceOnly(){
  try{ANALYTICS.race=await api('./api/race?'+raceQ());renderRace();}
  catch(e){$('#raceWrap').innerHTML='<div class="small muted">Error: '+esc(e.message)+'</div>';}}
async function loadAnalytics(){
  fillAnFilters();
  loadBoxOnly();
  try{
    const [sum,race]=await Promise.all([api('./api/analytics?'+anQ()),api('./api/race?'+raceQ())]);
    AN.sum=sum;ANALYTICS.race=race;
    MODEL_COLOR={};(sum.models||[]).map(x=>x.model).sort().forEach((m,i)=>MODEL_COLOR[m]=PALETTE[i%PALETTE.length]);
    $('#anNote').textContent=sum.note||'';
    renderKpiCards();renderLeader();renderPrompts();renderCost();renderAgree();renderOrder();renderMoment();
    renderCalibSel();renderCalib();renderUsage();renderHeat();renderRace();
  }catch(e){toast('Analytics error: '+e.message);}
}
function renderKpiCards(){
  const s=AN.sum||{},u=s.usage||{},el=$('#anKpis');
  const best=(s.models&&s.models[0])||null, bp=(s.prompts&&s.prompts[0])||null;
  const cards=[
    ['evaluated preds', s.models?s.models.reduce((a,m)=>a+m.n,0):0],
    ['finished matches', (s.heatmap&&s.heatmap.matches)?s.heatmap.matches.length:0],
    ['successful calls', fmtK(u.calls)],
    ['input tokens', fmtK(u.tokens_in)],
    ['output tokens', fmtK(u.tokens_out)],
    ['thinking tokens', fmtK(u.tokens_think)],
    ['recorded cost', u.cost!=null?('$'+u.cost.toFixed(2)):'–'],
    ['avg latency', u.avg_latency_ms!=null?((u.avg_latency_ms/1000).toFixed(1)+'s'):'–'],
    ['best model', best?`${best.model} · ${best.outcome_acc}%`:'–'],
    ['best prompt', bp?`${bp.prompt_id.replace('-prediction','')} · ${bp.outcome_acc}%`:'–'],
  ];
  el.innerHTML=cards.map(([lab,val])=>`<div class="card"><b>${esc(String(val))}</b><span>${esc(lab)}</span></div>`).join('');
}
function renderPrompts(){
  const ps=(AN.sum&&AN.sum.prompts)||[],el=$('#chPrompts');
  if(!ps.length)return noData('#chPrompts');
  const padL=labelPad(ps.map(p=>p.prompt_id.replace('-prediction','')));
  const W=chartW(el,980),padR=235,rowH=46,H=16+ps.length*rowH+26;
  const x=v=>padL+((v||0)/100)*(W-padL-padR);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const xx=x(t*25);
    s+=`<line x1="${xx}" y1="10" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${t*25}%</text>`;}
  ps.forEach((p,i)=>{const cy=18+i*rowH,col=pcolor(p.prompt_id);
    s+=`<text x="${padL-9}" y="${cy+13}" fill="${col}" text-anchor="end" font-weight="600">${esc(p.prompt_id.replace('-prediction',''))}</text>`;
    s+=`<rect x="${padL}" y="${cy}" width="${Math.max(1,x(p.outcome_acc)-padL)}" height="13" rx="2" fill="${col}"><title>per-prediction winner accuracy: ${p.outcome_acc??'–'}%</title></rect>`;
    s+=`<rect x="${padL}" y="${cy+15}" width="${Math.max(1,x(p.exact_acc)-padL)}" height="5" rx="2" fill="${col}" opacity="0.45"><title>exact score: ${p.exact_acc??'–'}%</title></rect>`;
    if(p.modal_acc!=null){const mx=x(p.modal_acc);
      s+=`<path d="M ${mx} ${cy} l 6 9 l -6 9 l -6 -9 Z" fill="none" stroke="var(--txt)" stroke-width="1.6"><title>majority vote per match: ${p.modal_acc}%</title></path>`;}
    s+=`<text x="${x(100)+12}" y="${cy+6}" fill="var(--txt)" font-size="11">${pct1(p.outcome_acc)} win · ◇${pct1(p.modal_acc)} · ${pct1(p.exact_acc)} exact</text>`;
    s+=`<text x="${x(100)+12}" y="${cy+20}" fill="var(--muted)" font-size="11">brier ${p.brier??'–'} · ${fmtK(p.avg_out_tokens)} out · ${fmtK(p.avg_think_tokens)} think · n=${p.n}</text>`;});
  s+='</svg>';el.innerHTML=s;
}
function renderUsage(){
  const u=(AN.sum&&AN.sum.usage)||{},el=$('#chUsage');
  const rows=(u.by_model||[]).slice().sort((a,b)=>(b.tokens_in+b.tokens_out+b.tokens_think)-(a.tokens_in+a.tokens_out+a.tokens_think));
  if(!rows.length)return noData('#chUsage','No successful calls under this filter.');
  const padL=labelPad(rows.map(r=>r.model)),W=chartW(el,980),padR=215,rowH=26,H=12+rows.length*rowH+16;
  const maxT=Math.max(1,...rows.map(r=>r.tokens_in+r.tokens_out+r.tokens_think));
  const x=v=>(v/maxT)*(W-padL-padR);
  let s=svgOpen(W,H);
  rows.forEach((r,i)=>{const cy=10+i*rowH;let cx=padL;
    s+=`<text x="${padL-9}" y="${cy+11}" fill="var(--txt)" text-anchor="end">${esc(shortName(r.model))}<title>${esc(r.model)}</title></text>`;
    [[r.tokens_in,'#94a3b8','input'],[r.tokens_out,'var(--accent)','output'],[r.tokens_think,'#a855f7','thinking']].forEach(([v,col,lab])=>{
      const w=x(v);if(w>0.5)s+=`<rect x="${cx}" y="${cy}" width="${w}" height="14" fill="${col}" opacity="0.85"><title>${esc(r.model)} · ${lab}: ${fmtK(v)} tokens</title></rect>`;cx+=w;});
    s+=`<text x="${W-padR+8}" y="${cy+11}" fill="var(--muted)" font-size="11">${fmtK(r.tokens_in+r.tokens_out+r.tokens_think)} tok · ${r.cost?('$'+r.cost.toFixed(2)):'free'} · ${fmtK(r.calls)} calls</text>`;});
  s+='</svg>';el.innerHTML=s;
}

function renderLeader(){
  const ms=(AN.sum&&AN.sum.models)||[],el=$('#chLeader');
  if(!ms.length)return noData('#chLeader');
  const padL=labelPad(ms.map(m=>m.model)),W=chartW(el,980),padR=175,rowH=34,H=16+ms.length*rowH+26;
  const x=v=>padL+((v||0)/100)*(W-padL-padR);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const xx=x(t*25);
    s+=`<line x1="${xx}" y1="10" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${t*25}%</text>`;}
  ms.forEach((m,i)=>{const cy=14+i*rowH,col=modelColor(m.model);
    s+=`<text x="${padL-9}" y="${cy+12}" fill="var(--txt)" text-anchor="end">${esc(shortName(m.model))}<title>${esc(m.model)}</title></text>`;
    s+=`<rect x="${padL}" y="${cy}" width="${Math.max(1,x(m.outcome_acc)-padL)}" height="13" rx="2" fill="${col}"><title>per-prediction winner accuracy: ${m.outcome_acc??'–'}%</title></rect>`;
    s+=`<rect x="${padL}" y="${cy+15}" width="${Math.max(1,x(m.exact_acc)-padL)}" height="5" rx="2" fill="${col}" opacity="0.45"><title>exact score: ${m.exact_acc??'–'}%</title></rect>`;
    if(m.modal_acc!=null){const mx=x(m.modal_acc);
      s+=`<path d="M ${mx} ${cy} l 6 9 l -6 9 l -6 -9 Z" fill="none" stroke="var(--txt)" stroke-width="1.6"><title>majority vote per match (race statistic): ${m.modal_acc}%</title></path>`;}
    s+=`<text x="${x(100)+12}" y="${cy+13}" fill="var(--muted)" font-size="11">${pct1(m.outcome_acc)}  ◇${pct1(m.modal_acc)}  ·n=${m.n}</text>`;});
  s+='</svg>';el.innerHTML=s;
}
function renderCost(){
  const ms=((AN.sum&&AN.sum.models)||[]).filter(m=>m.outcome_acc!=null),el=$('#chCost');
  if(!ms.length)return noData('#chCost');
  const W=chartW(el,640),H=300,padL=46,padR=16,padT=14,padB=32,labelRoom=118;
  const maxC=Math.max(1e-9,...ms.map(m=>m.avg_cost||0));
  const x=c=>padL+Math.sqrt((c||0)/maxC)*(W-padL-padR-labelRoom);
  const y=a=>padT+(1-(a||0)/100)*(H-padT-padB);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const yy=y(t*25);
    s+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="var(--border)"/>`;
    s+=`<text x="${padL-6}" y="${yy+4}" fill="var(--muted)" text-anchor="end" font-size="11">${t*25}%</text>`;}
  [[0,'free'],[maxC/4,fmtCost(maxC/4)],[maxC,fmtCost(maxC)]].forEach(([c,lab])=>{
    s+=`<text x="${x(c)}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${lab}</text>`;});
  // Greedy label lanes: sort by y, keep ≥13px between labels, connector when displaced.
  const pts=ms.map(m=>({m,cx:x(m.avg_cost),cy:y(m.outcome_acc)})).sort((a,b)=>a.cy-b.cy);
  let lastY=-99;
  pts.forEach(p=>{p.ly=Math.max(p.cy+4,lastY+13);lastY=p.ly;});
  const over=lastY-(H-padB-2);
  if(over>0)pts.forEach(p=>{p.ly-=over;});
  pts.forEach(p=>{const col=modelColor(p.m.model);
    s+=`<circle cx="${p.cx}" cy="${p.cy}" r="5.5" fill="${col}"><title>${esc(p.m.model)} · acc ${p.m.outcome_acc}% · ${fmtCost(p.m.avg_cost)}/call</title></circle>`;
    if(Math.abs(p.ly-(p.cy+4))>8)s+=`<line x1="${p.cx+6}" y1="${p.cy}" x2="${p.cx+15}" y2="${p.ly-4}" stroke="${col}" opacity="0.45"/>`;
    s+=`<text x="${p.cx+17}" y="${p.ly}" fill="var(--txt)" font-size="11">${esc(shortName(p.m.model,17))}</text>`;});
  s+='</svg>';el.innerHTML=s;
}
function renderAgree(){
  const ms=((AN.sum&&AN.sum.models)||[]).filter(m=>m.agreement!=null)
    .slice().sort((a,b)=>b.agreement-a.agreement);
  if(!ms.length)return noData('#chAgree','Needs ≥2 repetitions per match to measure agreement.');
  const el=$('#chAgree'),padL=labelPad(ms.map(m=>m.model)),W=chartW(el,640),padR=58,rowH=26,H=12+ms.length*rowH+26;
  const x=v=>padL+((v||0)/100)*(W-padL-padR);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const xx=x(t*25);
    s+=`<line x1="${xx}" y1="8" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${t*25}%</text>`;}
  ms.forEach((m,i)=>{const cy=10+i*rowH,col=modelColor(m.model);
    s+=`<text x="${padL-9}" y="${cy+11}" fill="var(--txt)" text-anchor="end">${esc(shortName(m.model))}<title>${esc(m.model)}</title></text>`;
    s+=`<rect x="${padL}" y="${cy}" width="${Math.max(1,x(m.agreement)-padL)}" height="13" rx="2" fill="${col}"><title>${m.agreement}% of ${m.agreement_n} match-groups unanimous</title></rect>`;
    s+=`<text x="${x(m.agreement)+6}" y="${cy+11}" fill="var(--muted)" font-size="11">${pct1(m.agreement)}</text>`;});
  s+='</svg>';el.innerHTML=s;
}
function renderOrder(){
  const ms=((AN.sum&&AN.sum.models)||[]).filter(m=>m.acc_original!=null&&m.acc_reversed!=null)
    .slice().sort((a,b)=>Math.abs(b.acc_original-b.acc_reversed)-Math.abs(a.acc_original-a.acc_reversed));
  if(!ms.length)return noData('#chOrder','Needs runs in both team orders.');
  const el=$('#chOrder'),padL=labelPad(ms.map(m=>m.model)),W=chartW(el,640),padR=58,rowH=26,H=30+ms.length*rowH+26;
  const x=v=>padL+((v||0)/100)*(W-padL-padR);
  let s=svgOpen(W,H);
  s+=`<circle cx="${padL}" cy="11" r="4.5" fill="var(--accent)"/><text x="${padL+9}" y="15" fill="var(--muted)" font-size="11">original</text>`;
  s+=`<circle cx="${padL+86}" cy="11" r="4.5" fill="var(--warn)"/><text x="${padL+95}" y="15" fill="var(--muted)" font-size="11">reversed</text>`;
  for(let t=0;t<=4;t++){const xx=x(t*25);
    s+=`<line x1="${xx}" y1="24" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${t*25}%</text>`;}
  ms.forEach((m,i)=>{const cy=38+i*rowH,xo=x(m.acc_original),xr=x(m.acc_reversed);
    const d=m.acc_reversed-m.acc_original;
    s+=`<text x="${padL-9}" y="${cy+4}" fill="var(--txt)" text-anchor="end">${esc(shortName(m.model))}<title>${esc(m.model)}</title></text>`;
    s+=`<line x1="${xo}" y1="${cy}" x2="${xr}" y2="${cy}" stroke="var(--muted)" stroke-width="2" opacity="0.6"/>`;
    s+=`<circle cx="${xo}" cy="${cy}" r="5.5" fill="var(--accent)"><title>original: ${m.acc_original}%</title></circle>`;
    s+=`<circle cx="${xr}" cy="${cy}" r="5.5" fill="var(--warn)"><title>reversed: ${m.acc_reversed}%</title></circle>`;
    s+=`<text x="${W-padR+6}" y="${cy+4}" fill="${Math.abs(d)>=10?'var(--err)':'var(--muted)'}" font-size="11">${d>0?'+':''}${d.toFixed(0)}pp</text>`;});
  s+='</svg>';el.innerHTML=s;
}
function renderMoment(){
  const ms=(AN.sum&&AN.sum.models)||[],el=$('#chMoment');
  const order=['pre_match','halftime','post_match'];
  const present=order.filter(mm=>ms.some(m=>m.moments&&m.moments[mm]&&m.moments[mm].acc!=null));
  if(!present.length)return noData('#chMoment');
  const rows=ms.filter(m=>present.some(mm=>m.moments[mm]&&m.moments[mm].acc!=null));
  const padL=labelPad(rows.map(m=>m.model)),W=chartW(el,640),padR=52,barH=10,gap=3;
  const rowH=present.length*(barH+gap)+9,H=28+rows.length*rowH+26;
  const x=v=>padL+((v||0)/100)*(W-padL-padR);
  let s=svgOpen(W,H);
  present.forEach((mm,j)=>{const lx=padL+j*95;
    s+=`<rect x="${lx}" y="5" width="9" height="9" rx="2" fill="${MOMENT_COLORS[mm]}"/><text x="${lx+13}" y="13" fill="var(--muted)" font-size="11">${mm.replace('_match','').replace('_','')}</text>`;});
  for(let t=0;t<=4;t++){const xx=x(t*25);
    s+=`<line x1="${xx}" y1="22" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${t*25}%</text>`;}
  rows.forEach((m,i)=>{const top=30+i*rowH;
    s+=`<text x="${padL-9}" y="${top+rowH/2-2}" fill="var(--txt)" text-anchor="end">${esc(shortName(m.model))}<title>${esc(m.model)}</title></text>`;
    present.forEach((mm,j)=>{const info=m.moments[mm];if(!info||info.acc==null)return;
      const cy=top+j*(barH+gap);
      s+=`<rect x="${padL}" y="${cy}" width="${Math.max(1,x(info.acc)-padL)}" height="${barH}" rx="2" fill="${MOMENT_COLORS[mm]}"><title>${mm}: ${info.acc}% (n=${info.n})</title></rect>`;
      s+=`<text x="${x(info.acc)+5}" y="${cy+9}" fill="var(--muted)" font-size="10">${pct1(info.acc)}</text>`;});});
  s+='</svg>';el.innerHTML=s;
}
function renderCalibSel(){
  const cal=(AN.sum&&AN.sum.calibration)||{},sel=$('#calibModel');
  const keys=Object.keys(cal),prev=sel.value;
  sel.innerHTML=keys.length
    ?('<option value="pooled">All models (pooled)</option>'+keys.filter(k=>k!=='pooled').sort().map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join(''))
    :'<option value="">–</option>';
  if(keys.includes(prev))sel.value=prev;
}
function renderCalib(){
  const cal=(AN.sum&&AN.sum.calibration)||{},el=$('#chCalib');
  const key=$('#calibModel').value||'pooled';
  const pts=(cal[key]||[]).slice().sort((a,b)=>a.p-b.p);
  if(!pts.length)return noData('#chCalib','No probability predictions under this filter (needs probability / six-hats prompts).');
  const W=chartW(el,560),H=320,padL=44,padR=14,padT=12,padB=32;
  const x=p=>padL+p*(W-padL-padR),y=p=>padT+(1-p)*(H-padT-padB);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const p=t/4;
    s+=`<line x1="${x(0)}" y1="${y(p)}" x2="${x(1)}" y2="${y(p)}" stroke="var(--border)"/>`;
    s+=`<text x="${padL-5}" y="${y(p)+4}" fill="var(--muted)" text-anchor="end" font-size="11">${(p*100)|0}%</text>`;
    s+=`<text x="${x(p)}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${(p*100)|0}%</text>`;}
  s+=`<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  const col=key==='pooled'?'var(--accent)':modelColor(key);
  let d='';pts.forEach((b,i)=>{d+=(i?'L':'M')+x(b.p).toFixed(1)+' '+y(b.obs).toFixed(1)+' ';});
  s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;
  pts.forEach(b=>{const r=3+Math.min(6,Math.sqrt(b.n)/3);
    s+=`<circle cx="${x(b.p)}" cy="${y(b.obs)}" r="${r}" fill="${col}" opacity="0.85"><title>predicted ${(b.p*100).toFixed(0)}% → happened ${(b.obs*100).toFixed(0)}% (n=${b.n})</title></circle>`;});
  s+=`<text x="${W-padR}" y="${H-6}" fill="var(--muted)" text-anchor="end">predicted probability →</text>`;
  s+='</svg>';el.innerHTML=s;
}
function renderHeat(){
  const hm=AN.sum&&AN.sum.heatmap,el=$('#chHeat');
  if(!hm||!hm.matches||!hm.matches.length)return noData('#chHeat');
  const rows=Object.entries(hm.rows).map(([m,cells])=>{
    const done=cells.filter(v=>v!=null);
    return {m,cells,acc:done.length?done.reduce((a,b)=>a+b,0)/done.length:0,n:done.length};
  }).sort((a,b)=>b.acc-a.acc);
  const cell=16,padL=labelPad(rows.map(r=>r.m)),padT=20,W=padL+hm.matches.length*cell+70,H=padT+rows.length*cell+8;
  let s=`<svg width="${W}" height="${H}" style="font:10px system-ui">`;
  hm.matches.forEach((mt,j)=>{if(j===0||(j+1)%5===0)
    s+=`<text x="${padL+j*cell+cell/2}" y="${padT-6}" fill="var(--muted)" text-anchor="middle">${j+1}</text>`;});
  rows.forEach((r,i)=>{const cy=padT+i*cell;
    s+=`<text x="${padL-8}" y="${cy+12}" fill="var(--txt)" text-anchor="end" style="font-size:11px">${esc(r.m)}</text>`;
    r.cells.forEach((v,j)=>{const mt=hm.matches[j];
      const fill=v==null?'var(--border)':(v?'var(--ok)':'var(--err)');
      s+=`<rect x="${padL+j*cell}" y="${cy}" width="${cell-2}" height="${cell-2}" rx="2" fill="${fill}" opacity="${v==null?0.35:0.85}"><title>${esc(r.m)} · ${esc(mt.name)} (${mt.date})${mt.actual?' · actual '+mt.actual:''} → ${v==null?'not run':(v?'correct':'wrong')}</title></rect>`;});
    s+=`<text x="${padL+hm.matches.length*cell+6}" y="${cy+12}" fill="var(--muted)" style="font-size:11px">${(r.acc*100).toFixed(0)}%</text>`;});
  s+='</svg>';el.innerHTML=s;
}

function renderBox(){
  const data=ANALYTICS.box, wrap=$('#boxWrap'); if(!data){return;}
  const metric=$('#boxMetric').value;
  const groups=data.groups.filter(g=>g[metric]).map(g=>({model:g.model,prompt:g.prompt_id,b:g[metric]}));
  if(!groups.length){wrap.innerHTML='<div class="small muted">'+esc(data.note||'No data for this metric yet (needs finished results; Brier needs probability prompts).')+'</div>';return;}
  let lo=Infinity,hi=-Infinity;
  groups.forEach(g=>{const b=g.b;lo=Math.min(lo,b.whislo,...(b.outliers||[]));hi=Math.max(hi,b.whishi,...(b.outliers||[]));});
  if(!(hi>lo))hi=lo+1;
  groups.sort((a,b)=>a.b.med-b.b.med);
  // Label depends on grouping: model×prompt, model-only or prompt-only.
  groups.forEach(g=>{const pShort=g.prompt.replace('-prediction','');
    if(g.model==='all models')g.lab=pShort;
    else if(g.prompt==='all')g.lab=shortName(g.model);
    else g.lab=shortName(g.model,16)+' · '+pShort;});
  const padL=labelPad(groups.map(g=>g.lab))+42,W=chartW(wrap,980),padR=28,padT=26,rowH=26,H=padT+groups.length*rowH+26;
  const x=v=>padL+(v-lo)/(hi-lo)*(W-padL-padR);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const v=lo+(hi-lo)*t/4,xx=x(v);
    s+=`<line x1="${xx}" y1="${padT-8}" x2="${xx}" y2="${H-22}" stroke="var(--border)"/>`;
    s+=`<text x="${xx}" y="${H-8}" fill="var(--muted)" text-anchor="middle" font-size="11">${v.toFixed(2)}</text>`;}
  groups.forEach((g,i)=>{const cy=padT+i*rowH+rowH/2-2,b=g.b;
    const col=(g.prompt!=='all')?pcolor(g.prompt):modelColor(g.model);
    const tip=`median ${b.med.toFixed(2)} · IQR ${b.q1.toFixed(2)}–${b.q3.toFixed(2)} · n=${b.n}`;
    s+=`<line x1="${x(b.whislo)}" y1="${cy}" x2="${x(b.whishi)}" y2="${cy}" stroke="var(--muted)"/>`;
    s+=`<line x1="${x(b.whislo)}" y1="${cy-5}" x2="${x(b.whislo)}" y2="${cy+5}" stroke="var(--muted)"/>`;
    s+=`<line x1="${x(b.whishi)}" y1="${cy-5}" x2="${x(b.whishi)}" y2="${cy+5}" stroke="var(--muted)"/>`;
    s+=`<rect x="${x(b.q1)}" y="${cy-8}" width="${Math.max(1.5,x(b.q3)-x(b.q1))}" height="16" fill="${col}22" stroke="${col}" stroke-width="1.5"><title>${tip}</title></rect>`;
    s+=`<line x1="${x(b.med)}" y1="${cy-8}" x2="${x(b.med)}" y2="${cy+8}" stroke="${col}" stroke-width="2.5"/>`;
    (b.outliers||[]).forEach(o=>{s+=`<circle cx="${x(o)}" cy="${cy}" r="2.2" fill="var(--err)" opacity="0.55"><title>outlier: ${o}</title></circle>`;});
    s+=`<text x="${padL-9}" y="${cy+4}" fill="var(--txt)" text-anchor="end">${esc(g.lab)} <tspan fill="var(--muted)" font-size="10.5">n=${b.n}</tspan><title>${esc(g.model)} · ${esc(g.prompt)}</title></text>`;});
  s+='</svg>';wrap.innerHTML=s;
}

function renderRace(){
  const data=ANALYTICS.race, wrap=$('#raceWrap');
  if(!data||!data.labels.length){wrap.innerHTML='<div class="small muted">'+esc((data&&data.note)||'No finished results yet.')+'</div>';return;}
  const models=Object.keys(data.series), N=data.labels.length;
  wrap.innerHTML=`<div class="section-actions" style="margin-bottom:8px">
      <button class="btn ghost" id="racePlay">▶ Play</button>
      <input type="range" id="raceScrub" min="1" max="${N}" value="${N}" style="flex:1;min-width:160px">
      <span class="small muted" id="raceAt"></span></div>
    <div style="display:flex;gap:14px;flex-wrap:wrap">
      <div style="flex:1;min-width:320px" id="raceSvg"></div>
      <div id="raceLegend" class="small" style="min-width:190px"></div></div>`;
  RACE={data,models,N,k:N,timer:null};
  $('#raceScrub').oninput=e=>{RACE.k=+e.target.value;drawRace();};
  $('#racePlay').onclick=toggleRacePlay;
  drawRace();
}
function drawRace(){
  const {data,models,N,k}=RACE;
  const box=$('#raceSvg');
  const W=Math.max(420,Math.min(920,(box&&box.clientWidth)||700)),H=360,padL=46,padR=14,padT=14,padB=28;
  const x=i=>padL+(N<=1?0:(i-1)/(N-1))*(W-padL-padR), y=a=>padT+(1-a)*(H-padT-padB);
  let s=svgOpen(W,H);
  for(let t=0;t<=4;t++){const a=t/4,yy=y(a);
    s+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="var(--border)"/>`;
    s+=`<text x="${padL-6}" y="${yy+4}" fill="var(--muted)" text-anchor="end" font-size="11">${(a*100)|0}%</text>`;}
  s+=`<line x1="${x(k)}" y1="${padT}" x2="${x(k)}" y2="${H-padB}" stroke="var(--accent)" stroke-dasharray="3 3" opacity="0.6"/>`;
  const rank=[];
  models.forEach((m,mi)=>{const pts=data.series[m],col=MODEL_COLOR[m]||(data.series_by==='prompt_id'?pcolor(m):PALETTE[mi%PALETTE.length]);
    let d='',on=false,ly=null,la=null;
    for(let i=1;i<=k;i++){const p=pts[i-1];if(!p||p.acc==null)continue;
      const px=x(i),py=y(p.acc);d+=(on?'L':'M')+px.toFixed(1)+' '+py.toFixed(1)+' ';on=true;ly=py;la=p.acc;}
    if(d){s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2" opacity="0.9"/>`;
      s+=`<circle cx="${x(k)}" cy="${ly}" r="3" fill="${col}"/>`;rank.push({m,acc:la,col});}});
  s+='</svg>';$('#raceSvg').innerHTML=s;
  const lab=data.labels[k-1];
  $('#raceAt').textContent=`#${k}/${N} · ${lab.date} · ${lab.name}`+(lab.actual?` (${lab.actual})`:'');
  rank.sort((a,b)=>b.acc-a.acc);
  $('#raceLegend').innerHTML='<b>Ranking @ '+k+'</b>'+rank.map((r,ix)=>
    `<div style="display:flex;align-items:center;gap:6px;margin-top:3px">
      <span style="width:14px;color:var(--muted)">${ix+1}</span>
      <span style="width:10px;height:10px;border-radius:2px;background:${r.col};display:inline-block"></span>
      <span style="flex:1">${esc(r.m.replace('-prediction',''))}</span><b>${(r.acc*100).toFixed(0)}%</b></div>`).join('');
}
function toggleRacePlay(){const btn=$('#racePlay');
  if(RACE.timer){clearInterval(RACE.timer);RACE.timer=null;btn.textContent='▶ Play';return;}
  if(RACE.k>=RACE.N)RACE.k=1;
  btn.textContent='⏸ Pause';
  RACE.timer=setInterval(()=>{RACE.k++;$('#raceScrub').value=RACE.k;drawRace();
    if(RACE.k>=RACE.N){clearInterval(RACE.timer);RACE.timer=null;btn.textContent='▶ Play';}},260);
}
$('#anPrompt').onchange=loadAnalytics;
$('#anMomentG').onchange=loadAnalytics;
$('#anReload').onclick=loadAnalytics;
$('#boxMetric').onchange=renderBox;
$('#boxGroup').onchange=loadBoxOnly;
$('#raceSeries').onchange=loadRaceOnly;
$('#calibModel').onchange=renderCalib;
let _anRz=null;
window.addEventListener('resize',()=>{clearTimeout(_anRz);_anRz=setTimeout(()=>{
  if($('#t-analytics').classList.contains('hide'))return;
  if(AN.sum){renderLeader();renderPrompts();renderCost();renderAgree();renderOrder();renderMoment();renderCalib();renderUsage();renderHeat();}
  if(ANALYTICS.box)renderBox();
  if(RACE)drawRace();
},200);});

function renderKpis(s){
  const k=$('#kpis');
  if(!s||!s.db_exists){k.innerHTML='<div class="kpi"><b>0</b><span>no data</span></div>';return;}
  const tot=(BOOT&&BOOT.matches)?BOOT.matches.length:0;
  let html=[['total','runs'],['success','ok'],['error','errors']]
    .map(([key,lab])=>`<div class="kpi"><b>${s[key]??0}</b><span>${lab}</span></div>`).join('');
  if(tot)html+=`<div class="kpi"><b style="color:var(--ok)">${s.matches??0}</b><span>matches run</span></div>`
            +`<div class="kpi"><b style="color:var(--warn)">${tot-(s.matches||0)}</b><span>to run</span></div>`;
  k.innerHTML=html;
}
function renderOvCards(s){
  const c=$('#ovCards');
  if(!s||!s.db_exists){c.innerHTML='<div class="card"><b>0</b><span>no database yet — run a collection</span></div>';return;}
  const tot=(BOOT&&BOOT.matches)?BOOT.matches.length:0;
  const items=[['total','executions'],['success','success'],['error','errors'],['valid','valid JSON'],
    ['models','models'],['results','results'],['finished','finished']];
  let html=items.map(([k,l])=>`<div class="card"><b>${s[k]??0}</b><span>${l}</span></div>`).join('');
  if(tot)html=`<div class="card"><b>${s.matches??0} / ${tot}</b><span>matches run</span></div>`
              +`<div class="card"><b style="color:var(--warn)">${tot-(s.matches||0)}</b><span>matches to run</span></div>`+html;
  c.innerHTML=html;
}

function table(rows,cols){
  if(!rows||!rows.length)return '<p class="muted small" style="padding:10px">No data.</p>';
  const head='<tr>'+cols.map(c=>`<th>${esc(c.label)}</th>`).join('')+'</tr>';
  const body=rows.map(r=>'<tr>'+cols.map(c=>`<td>${c.render?c.render(r[c.key],r):esc(r[c.key])}</td>`).join('')+'</tr>').join('');
  return `<table>${head}${body}</table>`;
}

async function api(path,opts){const r=await fetch(path,opts);if(!r.ok){throw new Error((await r.text())||r.status);}return r.json();}

async function boot(){
  BOOT=await api('./api/bootstrap');
  $('#roBanner').classList.toggle('hide',BOOT.actions_enabled);
  renderKpis(BOOT.summary); renderOvCards(BOOT.summary);
  buildRunForm();
  buildMatchPicker();
  buildForecastFilters();
  const opt=m=>`<option value="${m.match_id}">${esc(m.label)}${m.finished?' ✓':''}</option>`;
  const withPred=BOOT.matches.filter(m=>m.has_predictions);
  $('#cmpMatch').innerHTML=(withPred.length?withPred:BOOT.matches).map(opt).join('');
  loadEvaluate();
}

function buildRunForm(){
  $('#runReps').value='';
  $('#runMoments').innerHTML=BOOT.moments.map(m=>{
    const on=BOOT.default_moments.includes(m)?'checked':'';
    return `<label class="chk"><input type="checkbox" value="${m}" ${on}> ${m}</label>`;}).join('');
  $('#runModels').innerHTML=BOOT.models.map(m=>
    `<label class="chk"><input type="checkbox" value="${m.key}" checked> ${esc(m.key)} <span class="muted small">(${esc(m.provider)}) · ${m.reps}×</span></label>`).join('');
  const act=BOOT.actions_enabled;
  $('#btnRun').disabled=!act;
  $$('[data-fetch]').forEach(b=>b.disabled=!act); $('#btnReport').disabled=!act;
  updateEstimate();
  $('#runReps').oninput=updateEstimate;
  $('#runModels').onchange=$('#runMoments').onchange=updateEstimate;
}
function selModels(){return $$('#runModels input:checked').map(i=>i.value);}
function selMoments(){return $$('#runMoments input:checked').map(i=>i.value);}
function selMatches(){return $$('#runMatchList input:checked').map(i=>i.value);}
function updateEstimate(){
  const c=selMatches().length, matches=c||BOOT.matches.length;
  const override=+$('#runReps').value||null, orders=BOOT.orders||2;
  const repsSum=selModels().reduce((a,k)=>{const m=BOOT.models.find(x=>x.key===k);return a+(override||(m?m.reps:1));},0);
  const calls=matches*Math.max(selMoments().length,1)*BOOT.prompts.length*orders*repsSum;
  $('#runEstimate').textContent=`≈ ${calls.toLocaleString()} API calls (${orders} orderings)`;
  const mc=$('#matchCount'); if(mc)mc.textContent=c?`${c} selected`:'none → all matches';
}
$('#modelsAll').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=true);updateEstimate();};
$('#modelsNone').onclick=e=>{e.preventDefault();$$('#runModels input').forEach(i=>i.checked=false);updateEstimate();};

$('#btnRun').onclick=async()=>{
  const body={match_ids:selMatches(),moments:selMoments(),models:selModels(),
    reps:+$('#runReps').value||null,retry_errors:$('#runRetry').checked,dry_run:$('#runDry').checked};
  if(!body.models.length)return toast('Select at least one model.');
  if(!body.moments.length)return toast('Select at least one moment.');
  if(!body.dry_run && !confirm('This calls the paid LLM APIs and may cost money. Continue?'))return;
  try{const r=await api('./actions/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    toast(r.ok?('Started: '+r.command):('Busy: '+r.message));startPolling();tab('run');}
  catch(e){toast('Error: '+e.message);}
};
$$('[data-fetch]').forEach(b=>b.onclick=async()=>{
  try{const r=await api('./actions/fetch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:b.dataset.fetch,dry_run:false})});
    toast(r.ok?'Fetch started':('Busy: '+r.message));startPolling();}catch(e){toast('Error: '+e.message);}});
$('#btnReport').onclick=async()=>{try{const r=await api('./actions/report',{method:'POST'});
  toast(r.ok?'Report rebuild started':('Busy: '+r.message));startPolling();}catch(e){toast('Error: '+e.message);}};
$('#btnMerge').onclick=async()=>{
  const f=$('#mergeFile').files[0];
  if(!f)return toast('Choose a .db file first.');
  const fd=new FormData();fd.append('file',f);fd.append('replace',$('#mergeReplace').checked?'true':'false');
  const btn=$('#btnMerge');btn.disabled=true;$('#mergeMsg').textContent='Uploading & merging…';
  try{const r=await api('./actions/merge-upload',{method:'POST',body:fd});
    $('#mergeMsg').innerHTML=`Merged <b>${esc(r.filename||'')}</b>: +${r.runs_added} forecast row(s), +${r.results_added} result row(s). `+
      `Totals: ${r.runs_total} forecasts, ${r.results_total} results.`;
    toast(`Merged +${r.runs_added} rows`);refreshSummary();}
  catch(e){$('#mergeMsg').textContent='Error: '+e.message;toast('Merge failed');}
  finally{btn.disabled=false;}
};

function renderJob(j){
  const badge=$('#jobBadge');badge.className='badge b-'+(j.status||'idle');badge.textContent=j.status||'idle';
  const meta=`${j.command?('<b>'+esc(j.command)+'</b> · '):''}${j.status||'idle'}${j.returncode!=null?(' · rc='+j.returncode):''}${j.started_at?(' · '+j.started_at):''}`;
  $('#ovJobMeta').innerHTML=meta; $('#runJobMeta').innerHTML=meta;
  const pct=j.progress?Math.round(100*j.progress.done/Math.max(j.progress.total,1)):(j.status==='finished'?100:0);
  $('#ovBar').style.width=pct+'%'; $('#runBar').style.width=pct+'%';
  const log=j.log||'(no job yet)'; $('#ovLog').textContent=log; $('#runLog').textContent=log;
  $('#ovLog').scrollTop=$('#ovLog').scrollHeight; $('#runLog').scrollTop=$('#runLog').scrollHeight;
}
async function pollJob(){
  try{const j=await api('./api/job');renderJob(j);
    if(j.status==='running'){/*keep polling*/}
    else{ if(lastStatus==='running'){ // just finished
        refreshSummary();loadEvaluate(); if(!$('#t-results').classList.contains('hide'))loadResults();}
      stopPolling();}
    lastStatus=j.status;
  }catch(e){/* ignore transient */}
}
function startPolling(){lastStatus='running';renderJob({status:'running',command:'(starting)'});if(POLL)clearInterval(POLL);POLL=setInterval(pollJob,2000);pollJob();}
function stopPolling(){if(POLL){clearInterval(POLL);POLL=null;}}

async function refreshSummary(){try{const s=await api('./api/summary');BOOT.summary=s;renderKpis(s);renderOvCards(s);}catch(e){}}

async function loadEvaluate(){
  try{const d=await api('./api/evaluate');
    $('#ovNotes').textContent=(d.notes||[]).join('  ·  ');
    $('#ovLeader').innerHTML=table(d.leaderboard,[
      {key:'model',label:'Model'},{key:'provider',label:'Provider'},{key:'n',label:'Preds'},
      {key:'outcome_acc',label:'Winner %',render:v=>fmtPct(v)},
      {key:'exact_acc',label:'Exact %',render:v=>fmtPct(v)},
      {key:'mean_abs_gd_err',label:'GD err'},{key:'mean_brier',label:'Brier'}]);
  }catch(e){$('#ovLeader').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function loadResults(){
  try{const d=await api('./api/results');
    $('#resTable').innerHTML=table(d.results,[
      {key:'match_id',label:'#'},{key:'team_1',label:'Home'},{key:'team_2',label:'Away'},
      {key:'actual_score_team_1',label:'H'},{key:'actual_score_team_2',label:'A'},
      {key:'actual_winner',label:'Winner'},{key:'status',label:'Status'},{key:'source',label:'Source'}]);
  }catch(e){$('#resTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function loadCompare(){
  const sel=$('#cmpMatch'); if(!sel.value && sel.options.length)sel.value=sel.options[0].value;
  if(!sel.value){$('#cmpTable').innerHTML='<p class="muted small" style="padding:10px">No matches with predictions yet.</p>';return;}
  sel.onchange=loadCompare;
  try{const d=await api('./api/compare?match_id='+encodeURIComponent(sel.value));
    const i=d.info||{};
    $('#cmpInfo').innerHTML=`<b>${esc(i.team_1)} vs ${esc(i.team_2)}</b> — `+
      (i.actual_score?`actual <b>${esc(i.actual_score)}</b> (winner: ${esc(i.actual_winner)}) <span class="pill ok">finished</span>`
        :'<span class="pill na">no result yet</span>');
    const win=v=>v==null?'<span class="pill na">–</span>':(v?'<span class="pill ok">✓</span>':'<span class="pill bad">✗</span>');
    $('#cmpTable').innerHTML=table(d.rows,[
      {key:'model',label:'Model'},{key:'prompt_id',label:'Prompt'},{key:'moment',label:'Moment'},
      {key:'reps',label:'n'},{key:'modal_score',label:'Predicted'},
      {key:'winner_agreement',label:'Agree',render:v=>Math.round(v*100)+'%'},
      {key:'modal_winner',label:'Pred winner'},
      {key:'outcome_correct',label:'Winner?',render:win},
      {key:'exact_correct',label:'Exact?',render:win}]);
  }catch(e){$('#cmpTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}

function applyMatchFilter(){
  const q=$('#runMatchSearch').value.toLowerCase(), onlyNot=$('#hideRun').checked;
  [...$('#runMatchList').children].forEach(el=>{
    el.style.display=(el.dataset.text.includes(q) && (!onlyNot || el.dataset.run==='0'))?'':'none';});
}
function buildMatchPicker(){
  const list=$('#runMatchList');
  list.innerHTML=BOOT.matches.map(m=>{
    const badge=m.has_predictions
      ? `<span class="pill ok" title="${m.n_runs} runs · ${m.n_models} models">run · ${m.n_runs}</span>`
      : `<span class="pill na">not run</span>`;
    const fin=m.finished?` <span class="pill" style="color:#7eb0ff">finished</span>`:'';
    return `<label class="chk" style="justify-content:flex-start" data-run="${m.has_predictions?1:0}" data-text="${esc((m.label+' '+(m.phase||'')).toLowerCase())}">
       <input type="checkbox" value="${esc(m.match_id)}"> ${esc(m.match_id)}. ${esc(m.team_1)} v ${esc(m.team_2)}
       <span class="muted small">${esc(m.local_date||'')}</span> ${badge}${fin}</label>`;}).join('');
  list.onchange=updateEstimate;
  $('#runMatchSearch').oninput=applyMatchFilter;
  $('#hideRun').onchange=applyMatchFilter;
  const setAll=v=>{[...list.querySelectorAll('input')].forEach(i=>i.checked=v);updateEstimate();};
  $('#matchAll').onclick=e=>{e.preventDefault();setAll(true);};
  $('#matchNone').onclick=e=>{e.preventDefault();setAll(false);};
  $('#matchNotRun').onclick=e=>{e.preventDefault();[...list.children].forEach(el=>el.querySelector('input').checked=(el.dataset.run==='0'));updateEstimate();};
  $('#matchShown').onclick=e=>{e.preventDefault();[...list.children].forEach(el=>{if(el.style.display!=='none')el.querySelector('input').checked=true;});updateEstimate();};
  const run=BOOT.matches.filter(m=>m.has_predictions).length, tot=BOOT.matches.length;
  $('#matchCoverage').innerHTML=`<b style="color:var(--ok)">${run}</b> already run · <b style="color:var(--warn)">${tot-run}</b> not run yet · ${tot} total`;
  updateEstimate();
}

let fxOffset=0; const fxLimit=50;
function buildForecastFilters(){
  const opts=(arr,fn)=>'<option value="">any</option>'+arr.map(fn).join('');
  $('#fxMatch').innerHTML=opts(BOOT.matches,m=>`<option value="${esc(m.match_id)}">${esc(m.match_id)}. ${esc(m.team_1)} v ${esc(m.team_2)}</option>`);
  $('#fxModel').innerHTML=opts(BOOT.models,m=>`<option value="${esc(m.key)}">${esc(m.key)}</option>`);
  $('#fxPrompt').innerHTML=opts(BOOT.prompts,p=>`<option value="${esc(p)}">${esc(p)}</option>`);
  $('#fxMoment').innerHTML=opts(BOOT.moments,m=>`<option value="${esc(m)}">${esc(m)}</option>`);
  $('#fxApply').onclick=()=>{fxOffset=0;loadForecasts();};
  $('#fxReset').onclick=()=>{['fxMatch','fxModel','fxPrompt','fxMoment','fxOrder','fxStatus','fxValid'].forEach(id=>$('#'+id).value='');$('#fxQ').value='';fxOffset=0;loadForecasts();};
  $('#fxQ').onkeydown=e=>{if(e.key==='Enter'){fxOffset=0;loadForecasts();}};
  $('#fxPrev').onclick=()=>{if(fxOffset>0){fxOffset=Math.max(0,fxOffset-fxLimit);loadForecasts();}};
  $('#fxNext').onclick=()=>{fxOffset+=fxLimit;loadForecasts();};
}
function fxQuery(){
  const p=new URLSearchParams(), g=id=>$('#'+id).value;
  if(g('fxMatch'))p.set('match_id',g('fxMatch'));
  if(g('fxModel'))p.set('model',g('fxModel'));
  if(g('fxPrompt'))p.set('prompt_id',g('fxPrompt'));
  if(g('fxMoment'))p.set('moment',g('fxMoment'));
  if(g('fxOrder'))p.set('order',g('fxOrder'));
  if(g('fxStatus'))p.set('status',g('fxStatus'));
  if(g('fxValid'))p.set('valid',g('fxValid'));
  if(g('fxQ'))p.set('q',g('fxQ'));
  p.set('limit',fxLimit);p.set('offset',fxOffset);
  return p.toString();
}
async function loadForecasts(){
  try{const d=await api('./api/forecasts?'+fxQuery());
    const score=r=>(r.parsed_score_team_1!=null&&r.parsed_score_team_2!=null)?`${r.parsed_score_team_1}-${r.parsed_score_team_2}`:'–';
    const vp=v=>v?'<span class="pill ok">ok</span>':'<span class="pill bad">no</span>';
    const body=d.rows.map(r=>`<tr class="clickrow" data-id="${esc(r.run_id)}">
      <td>${esc(r.match_id)}</td><td>${esc(r.team_1)} v ${esc(r.team_2)}</td><td>${esc(r.model)}</td>
      <td>${esc(r.prompt_id)}</td><td>${esc(r.match_moment||'')}</td><td>${esc(r.team_order_type||'')}</td>
      <td>${esc(r.repetition_number)}</td><td>${score(r)}</td><td>${vp(r.json_valid)}</td>
      <td>${esc(r.execution_status)}</td><td>${r.total_tokens==null?'':r.total_tokens}</td></tr>`).join('');
    $('#fxTable').innerHTML=d.rows.length
      ? `<table><tr><th>#</th><th>Match</th><th>Model</th><th>Prompt</th><th>Moment</th><th>Order</th><th>Rep</th><th>Score</th><th>JSON</th><th>Status</th><th>Tok</th></tr>${body}</table>`
      : '<p class="muted small" style="padding:10px">No rows match these filters.</p>';
    $('#fxCount').textContent=`${d.total.toLocaleString()} rows`;
    $('#fxPage').textContent=d.total?`${d.offset+1}–${Math.min(d.offset+d.limit,d.total)} of ${d.total}`:'0';
    $('#fxPrev').disabled=d.offset<=0; $('#fxNext').disabled=d.offset+d.limit>=d.total;
    $$('#fxTable .clickrow').forEach(tr=>tr.onclick=()=>openDetail(tr.dataset.id));
  }catch(e){$('#fxTable').innerHTML='<p class="muted small" style="padding:10px">'+esc(e.message)+'</p>';}
}
async function openDetail(runId){
  try{const r=await api('./api/forecast/'+encodeURIComponent(runId));
    const row=(k,v)=>(v==null||v==='')?'':`<div class="k">${esc(k)}</div><div>${esc(v)}</div>`;
    let hats='';['white_hat','red_hat','black_hat','yellow_hat','green_hat','blue_hat'].forEach(h=>{if(r[h])hats+=`<div class="k">${h}</div><div>${esc(r[h])}</div>`;});
    $('#fxModalBox').innerHTML=`<span class="x" id="fxClose">✕</span>
      <h2 style="margin-top:0">${esc(r.model)} · ${esc(r.prompt_id)} · ${esc(r.match_moment||'')}</h2>
      <div class="kv">
        ${row('match',r.match_id+'. '+r.team_1+' v '+r.team_2)}${row('order',r.team_order_type)}${row('rep',r.repetition_number)}
        ${row('predicted',(r.parsed_score_team_1!=null?r.parsed_score_team_1+'-'+r.parsed_score_team_2:'–'))}
        ${row('probs (1/X/2)',[r.parsed_team1_win_probability,r.parsed_draw_probability,r.parsed_team2_win_probability].join(' / '))}
        ${row('json valid',r.json_valid?'yes':'no')}${row('status',r.execution_status)}${row('latency ms',r.latency_ms)}
        ${row('tokens p/c/total',[r.prompt_tokens,r.completion_tokens,r.total_tokens].join(' / '))}${row('cost usd',r.api_cost)}
        ${row('error',r.error_message)}</div>
      ${hats?`<h2>Six hats</h2><div class="kv">${hats}</div>`:''}
      <h2>Raw response</h2><pre class="log" style="max-height:260px">${esc(r.raw_response||'(none)')}</pre>
      <h2>Prompt sent</h2><pre class="log" style="max-height:200px">${esc(r.prompt_text||'')}</pre>`;
    $('#fxModal').classList.remove('hide');
    $('#fxClose').onclick=()=>$('#fxModal').classList.add('hide');
  }catch(e){toast('Error: '+e.message);}
}
$('#fxModal').onclick=e=>{if(e.target.id==='fxModal')$('#fxModal').classList.add('hide');};

boot().then(()=>{ // resume polling if a job is already running
  fetch('./api/job').then(r=>r.json()).then(j=>{if(j.status==='running')startPolling();else renderJob(j);});
}).catch(e=>toast('Load error: '+e.message,8000));
</script>
</body></html>
"""
