"""Gemini Batch API collection — 50% of interactive pricing.

Builds the same execution plan as ``run`` (per-model reps, both orderings,
deterministic run_ids, resume semantics) but, instead of live calls, submits
one Google Batch job per Gemini model, waits for completion and writes the
results into the same SQLite + JSON archive the interactive runner uses.
``api_cost`` is recorded at the 50% batch rate.

Suitable for **pre-match** collection (run it the day before): batch jobs
complete asynchronously — usually within minutes to a few hours, guaranteed
under 24h. Time-critical moments (halftime / post_match on match day) must
keep using the interactive ``run``.

Crash safety: every submission writes ``data/batch_jobs/<stamp>__<model>.json``
holding the job name plus the ordered run specs. If the wait is interrupted,
resume later with ``python -m fifa_forecast gemini-batch --collect <file>``
(results are matched back to run specs by position, which Google guarantees
for inlined requests).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from . import prompts as prompt_lib
from .matches import filter_matches, load_matches, ordered_pair
from .parsing import parse_forecast
from .providers.gemini_provider import GeminiProvider, _to_dict
from .runner import make_run_id, _response_format_for
from .storage import Database, FileArchive, RunRecord

BATCH_DISCOUNT = 0.5  # Google bills batch at 50% of interactive pricing.

_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _now_pair() -> tuple[str, str]:
    utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    local = datetime.now().astimezone().isoformat()
    return utc, local


def _state_dir() -> Path:
    p = cfg.DATA_DIR / "batch_jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gemini_models(config: cfg.Config, model_keys: list[str] | None = None) -> list[dict]:
    models = [m for m in config.enabled_models() if m.get("provider") == "gemini"]
    if model_keys:
        wanted = set(model_keys)
        models = [m for m in models if m["key"] in wanted]
    return models


def build_plan(
    config: cfg.Config,
    db: Database,
    model_config: dict[str, Any],
    *,
    dates: list[str] | None = None,
    match_ids: list[str] | None = None,
    moments: list[str] | None = None,
    reps: int | None = None,
    retry_errors: bool = False,
) -> list[dict[str, Any]]:
    """Pending executions for one model — mirrors the runner's grid + skip rules."""
    matches = filter_matches(
        load_matches(cfg.ROOT / config.matches_csv), dates=dates, match_ids=match_ids
    )
    eff_moments = list(moments) if moments else list(config.match_moments)
    mreps = int(reps if reps is not None else model_config.get("reps", config.runs_per_combination))

    plan: list[dict[str, Any]] = []
    for match in matches:
        for moment in eff_moments:
            for order in config.team_order_types:
                for prompt_id in config.prompt_ids:
                    for rep in range(1, mreps + 1):
                        run_id = make_run_id(
                            match.match_id, model_config["key"], prompt_id, order, rep, moment
                        )
                        status = db.status_of(run_id)
                        if status == "success":
                            continue
                        if status == "error" and not retry_errors:
                            continue
                        first, second = ordered_pair(match, order)
                        kickoff = match.kickoff_local or match.kickoff_datetime
                        plan.append({
                            "run_id": run_id,
                            "match_id": match.match_id,
                            "phase": match.phase,
                            "team_1": match.team_1,
                            "team_2": match.team_2,
                            "kickoff_datetime": match.kickoff_datetime,
                            "order": order,
                            "moment": moment,
                            "prompt_id": prompt_id,
                            "rep": rep,
                            "first": first,
                            "second": second,
                            "prompt_text": prompt_lib.render(
                                prompt_id, first, second, kickoff=kickoff, moment=moment
                            ),
                        })
    return plan


def build_requests(model_config: dict[str, Any], plan: list[dict[str, Any]]):
    """Inline batch requests with the exact generation config the live provider uses."""
    from google.genai import types

    prov = GeminiProvider(model_config)
    prov._types = types
    gen_cfg = prov._build_config(prompt_lib.SYSTEM_PROMPT)
    return [
        types.InlinedRequest(
            contents=[types.Content(role="user", parts=[types.Part(text=spec["prompt_text"])])],
            config=gen_cfg,
        )
        for spec in plan
    ]


def _client(model_config: dict[str, Any]):
    from google import genai

    api_key = os.environ.get(model_config.get("api_key_env", "GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError(
            f"Environment variable {model_config.get('api_key_env', 'GEMINI_API_KEY')} is not set."
        )
    return genai.Client(api_key=api_key)


def submit(model_config: dict[str, Any], plan: list[dict[str, Any]]):
    """Create the batch job; returns (client, job, state_path)."""
    from google.genai import types

    client = _client(model_config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    display = f"fifa-{model_config['key']}-{stamp}"
    job = client.batches.create(
        model=model_config["model_id"],
        src=build_requests(model_config, plan),
        config=types.CreateBatchJobConfig(display_name=display),
    )
    state_path = _state_dir() / f"{stamp}__{model_config['key']}.json"
    state_path.write_text(
        json.dumps(
            {
                "job_name": job.name,
                "model_key": model_config["key"],
                "submitted_at_utc": _now_pair()[0],
                "collected": False,
                "specs": plan,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return client, job, state_path


def wait(client, job_name: str, *, poll_interval: int = 60, timeout: int = 86400,
         progress: Callable[[str], None] = print):
    """Poll until the job reaches a terminal state (or timeout)."""
    start = time.time()
    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job.state, "name", None) or str(job.state)
        if state in _TERMINAL_STATES:
            progress(f"  {job_name}: {state}")
            return job
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            raise TimeoutError(
                f"Batch {job_name} still {state} after {elapsed}s — resume later with "
                f"`gemini-batch --collect <state file>`."
            )
        progress(f"  {job_name}: {state} ({elapsed}s elapsed)")
        time.sleep(poll_interval)


def collect(
    config: cfg.Config,
    model_config: dict[str, Any],
    job,
    plan: list[dict[str, Any]],
    db: Database,
    archive: FileArchive,
    *,
    submitted_at_utc: str | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Write one RunRecord (+ JSON artifacts) per plan spec from the job output.

    Google returns inlined responses in submission order, so responses are
    matched to specs by position; any missing/errored entry becomes an error row.
    """
    responses = list((getattr(job, "dest", None) and job.dest.inlined_responses) or [])
    job_name = getattr(job, "name", None)
    state = getattr(getattr(job, "state", None), "name", None) or str(getattr(job, "state", ""))
    if len(responses) != len(plan):
        progress(
            f"  WARNING: {len(plan)} requests but {len(responses)} responses "
            f"(job {state}); unmatched requests become error rows."
        )

    params = dict(model_config.get("params") or {})
    pricing = model_config.get("pricing")
    system_prompt = prompt_lib.SYSTEM_PROMPT
    resp_utc, resp_local = _now_pair()
    stats = {"success": 0, "error": 0}

    for i, spec in enumerate(plan):
        run_id = spec["run_id"]
        record = RunRecord(
            run_id=run_id,
            match_id=spec["match_id"],
            phase=spec["phase"],
            team_1=spec["team_1"],
            team_2=spec["team_2"],
            kickoff_datetime=spec["kickoff_datetime"],
            team_order_type=spec["order"],
            match_moment=spec["moment"],
            prompt_team_1=spec["first"],
            prompt_team_2=spec["second"],
            provider="gemini",
            model=model_config["key"],
            model_id=model_config.get("model_id"),
            prompt_id=spec["prompt_id"],
            repetition_number=spec["rep"],
            request_timestamp_utc=submitted_at_utc,
            response_timestamp_utc=resp_utc,
            response_timestamp_local=resp_local,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            seed=params.get("seed"),
            max_tokens=params.get("max_tokens", params.get("max_output_tokens")),
            response_format=_response_format_for("gemini", params),
            api_params_json=json.dumps(
                {**params, "_batch_job": job_name, "_batch_discount": BATCH_DISCOUNT},
                ensure_ascii=False,
            ),
            system_prompt=system_prompt,
            prompt_text=spec["prompt_text"],
            attempt_count=1,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        request_payload = {
            "provider": "gemini",
            "mode": "batch",
            "batch_job": job_name,
            "model_id": model_config.get("model_id"),
            "system": system_prompt,
            "prompt": spec["prompt_text"],
            "params": params,
        }

        entry = responses[i] if i < len(responses) else None
        err = getattr(entry, "error", None) if entry is not None else None
        resp = getattr(entry, "response", None) if entry is not None else None
        if resp is None:
            record.execution_status = "error"
            record.error_message = (
                f"batch: {err}" if err else f"batch: no response returned (job {state})"
            )
            record.json_valid = 0
            archive.write_request(run_id, request_payload)
            archive.write_response(run_id, {"error": record.error_message})
            archive.write_trace(run_id, {"error": record.error_message, "batch_job": job_name})
            archive.write_metadata(run_id, {"run_id": run_id, "status": "error"})
            db.insert_run(record)
            stats["error"] += 1
            continue

        text = getattr(resp, "text", None) or ""
        usage = getattr(resp, "usage_metadata", None)
        record.prompt_tokens = getattr(usage, "prompt_token_count", None)
        record.completion_tokens = getattr(usage, "candidates_token_count", None)
        record.reasoning_tokens = getattr(usage, "thoughts_token_count", None)
        record.total_tokens = getattr(usage, "total_token_count", None)
        record.response_id = getattr(resp, "response_id", None)
        record.execution_status = "success"
        record.raw_response = text
        if pricing and record.prompt_tokens is not None and record.completion_tokens is not None:
            try:
                record.api_cost = round(
                    (
                        record.prompt_tokens / 1_000_000 * float(pricing["input"])
                        + record.completion_tokens / 1_000_000 * float(pricing["output"])
                    ) * BATCH_DISCOUNT,
                    8,
                )
            except (KeyError, TypeError, ValueError):
                record.api_cost = None

        parsed = parse_forecast(text, spec["prompt_id"])
        record.json_valid = 1 if parsed.json_valid else 0
        record.parsed_json = (
            json.dumps(parsed.parsed_json, ensure_ascii=False)
            if parsed.parsed_json is not None
            else None
        )
        record.parsed_score_team_1 = parsed.score_team_1
        record.parsed_score_team_2 = parsed.score_team_2
        record.parsed_team1_win_probability = parsed.team1_win_probability
        record.parsed_draw_probability = parsed.draw_probability
        record.parsed_team2_win_probability = parsed.team2_win_probability
        for hat, value in parsed.hats.items():
            setattr(record, hat, value)
        record.validation_notes = (
            "; ".join(parsed.validation_notes) if parsed.validation_notes else None
        )

        archive.write_request(run_id, request_payload)
        archive.write_response(run_id, {"raw_text": text, "response": _to_dict(resp)})
        archive.write_trace(
            run_id,
            {"usage": _to_dict(usage) if usage is not None else None, "batch_job": job_name},
        )
        archive.write_metadata(run_id, {"run_id": run_id, "status": "success", "batch_job": job_name})
        db.insert_run(record)
        stats["success"] += 1

    progress(f"  collected: ok={stats['success']} err={stats['error']}")
    return stats


def run_batch(
    config: cfg.Config,
    *,
    dates: list[str] | None = None,
    match_ids: list[str] | None = None,
    moments: list[str] | None = None,
    model_keys: list[str] | None = None,
    reps: int | None = None,
    retry_errors: bool = False,
    wait_for_results: bool = True,
    poll_interval: int = 60,
    timeout: int = 86400,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Submit one batch per Gemini model and (optionally) wait + collect."""
    totals = {"success": 0, "error": 0, "submitted": 0}
    db = Database(cfg.ROOT / config.database_path)
    archive = FileArchive()
    try:
        for mc in gemini_models(config, model_keys):
            plan = build_plan(
                config, db, mc, dates=dates, match_ids=match_ids, moments=moments,
                reps=reps, retry_errors=retry_errors,
            )
            if not plan:
                progress(f"{mc['key']}: nothing pending — skipped.")
                continue
            progress(f"{mc['key']}: submitting batch with {len(plan)} request(s)…")
            client, job, state_path = submit(mc, plan)
            totals["submitted"] += len(plan)
            progress(f"  job: {job.name}\n  state file: {state_path}")
            if not wait_for_results:
                progress("  --no-wait: collect later with "
                         f"`python -m fifa_forecast gemini-batch --collect \"{state_path}\"`")
                continue
            job = wait(client, job.name, poll_interval=poll_interval, timeout=timeout,
                       progress=progress)
            st = json.loads(state_path.read_text(encoding="utf-8"))
            stats = collect(config, mc, job, plan, db, archive,
                            submitted_at_utc=st.get("submitted_at_utc"), progress=progress)
            totals["success"] += stats["success"]
            totals["error"] += stats["error"]
            st["collected"] = True
            state_path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        db.close()
    return totals


def collect_from_state(
    config: cfg.Config,
    state_file: str | Path,
    *,
    poll_interval: int = 60,
    timeout: int = 86400,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Resume an interrupted batch: wait (if needed) and write its results."""
    state_path = Path(state_file)
    st = json.loads(state_path.read_text(encoding="utf-8"))
    mc = config.model_by_key(st["model_key"])
    if mc is None:
        raise RuntimeError(f"Model {st['model_key']!r} not found in the current config.")
    client = _client(mc)
    job = wait(client, st["job_name"], poll_interval=poll_interval, timeout=timeout,
               progress=progress)
    db = Database(cfg.ROOT / config.database_path)
    archive = FileArchive()
    try:
        stats = collect(config, mc, job, st["specs"], db, archive,
                        submitted_at_utc=st.get("submitted_at_utc"), progress=progress)
    finally:
        db.close()
    st["collected"] = True
    state_path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats
