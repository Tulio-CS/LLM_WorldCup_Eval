"""Experiment orchestration.

Iterates the full grid — match x team-ordering x model x prompt x repetition —
calling each provider with a bounded retry policy, parsing the response,
archiving the four JSON artifacts, and writing one row to ``forecast_runs`` per
execution. Failures are recorded as error rows, never silently dropped.

Run ids are deterministic in the combination, so a re-run resumes: already
completed combinations are skipped (unless ``overwrite=True``), and original
outputs are preserved.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from . import config as cfg
from . import prompts as prompt_lib
from .matches import Match, filter_matches, load_matches, ordered_pair
from .parsing import parse_forecast
from .providers import ProviderError, build_provider
from .providers.base import BaseProvider, ProviderResult
from .storage import Database, FileArchive, RunRecord

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(token: str) -> str:
    return _SAFE.sub("-", str(token))


def make_run_id(
    match_id: str,
    model_key: str,
    prompt_id: str,
    order: str,
    repetition: int,
    moment: str = "pre_match",
) -> str:
    return "__".join(
        [
            _safe(match_id),
            _safe(model_key),
            _safe(prompt_id),
            _safe(order),
            _safe(moment),
            f"rep{repetition:02d}",
        ]
    )


def _now() -> tuple[str, str]:
    """Return (utc_iso, local_iso)."""
    utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    local = datetime.now().astimezone().isoformat()
    return utc, local


def _response_format_for(provider: str, params: dict[str, Any]) -> str | None:
    if "response_format" in params:
        rf = params["response_format"]
        return None if rf == "none" else str(rf)
    if provider in ("openai", "grok"):
        return "json_object"
    if provider == "gemini":
        return "application/json"
    return None


class ExperimentRunner:
    def __init__(
        self,
        config: cfg.Config,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.progress = progress or (lambda msg: None)
        self.db = Database(cfg.ROOT / config.database_path)
        self.archive = FileArchive()
        self._providers: dict[str, BaseProvider] = {}
        self.stats = {"total": 0, "success": 0, "error": 0, "skipped": 0}

    # -- provider caching ---------------------------------------------------
    def _provider_for(self, model_config: dict[str, Any]) -> BaseProvider:
        key = model_config["key"]
        if key not in self._providers:
            self._providers[key] = build_provider(
                model_config, dry_run=self.dry_run, timeout=self.config.request_timeout
            )
        return self._providers[key]

    # -- planning -----------------------------------------------------------
    def plan_size(self, matches: list[Match]) -> int:
        return (
            len(matches)
            * len(self.config.team_order_types)
            * len(self.config.match_moments)
            * len(self.config.enabled_models())
            * len(self.config.prompt_ids)
            * self.config.runs_per_combination
        )

    # -- execution ----------------------------------------------------------
    def run(
        self,
        *,
        limit_matches: int | None = None,
        dates: list[str] | None = None,
        match_ids: list[str] | None = None,
    ) -> dict[str, int]:
        matches = load_matches(cfg.ROOT / self.config.matches_csv)
        matches = filter_matches(matches, dates=dates, match_ids=match_ids)
        if limit_matches is not None:
            matches = matches[:limit_matches]

        if not matches:
            self.progress(
                "No matches selected after filtering — nothing to do. "
                "Check --date (local UTC-3 date, e.g. 2026-06-22) / --match-id."
            )
            return dict(self.stats)

        models = self.config.enabled_models()
        total_planned = self.plan_size(matches)
        self.progress(
            f"Planned executions: {total_planned} "
            f"({len(matches)} matches x {len(self.config.team_order_types)} orders "
            f"x {len(self.config.match_moments)} moments "
            f"x {len(models)} models x {len(self.config.prompt_ids)} prompts "
            f"x {self.config.runs_per_combination} reps)"
        )

        done = 0
        for match in matches:
            for moment in self.config.match_moments:
                for order in self.config.team_order_types:
                    for model_config in models:
                        for prompt_id in self.config.prompt_ids:
                            for rep in range(1, self.config.runs_per_combination + 1):
                                done += 1
                                detail = self._execute_one(
                                    match, order, moment, model_config, prompt_id, rep
                                )
                                self.progress(f"[{done}/{total_planned}] {detail}")
        self.progress(
            f"Summary: ok={self.stats['success']}, err={self.stats['error']}, "
            f"skip={self.stats['skipped']}"
        )
        return dict(self.stats)

    def _execute_one(
        self,
        match: Match,
        order: str,
        moment: str,
        model_config: dict[str, Any],
        prompt_id: str,
        rep: int,
    ) -> str:
        self.stats["total"] += 1
        run_id = make_run_id(
            match.match_id, model_config["key"], prompt_id, order, rep, moment
        )

        if not self.overwrite and self.db.count("run_id = ?", (run_id,)) > 0:
            self.stats["skipped"] += 1
            return f"skip  {run_id}"

        first, second = ordered_pair(match, order)
        kickoff = match.kickoff_local or match.kickoff_datetime
        prompt_text = prompt_lib.render(
            prompt_id, first, second, kickoff=kickoff, moment=moment
        )
        system_prompt = prompt_lib.SYSTEM_PROMPT
        params = dict(model_config.get("params") or {})
        provider_name = model_config.get("provider")

        record = RunRecord(
            run_id=run_id,
            match_id=match.match_id,
            phase=match.phase,
            team_1=match.team_1,
            team_2=match.team_2,
            kickoff_datetime=match.kickoff_datetime,
            team_order_type=order,
            match_moment=moment,
            prompt_team_1=first,
            prompt_team_2=second,
            provider=provider_name,
            model=model_config["key"],
            model_id=model_config.get("model_id"),
            prompt_id=prompt_id,
            repetition_number=rep,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            seed=params.get("seed"),
            max_tokens=params.get("max_tokens", params.get("max_output_tokens")),
            reasoning_effort=params.get("reasoning_effort", params.get("effort")),
            response_format=_response_format_for(provider_name, params),
            api_params_json=json.dumps(params, ensure_ascii=False),
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        provider = self._provider_for(model_config)
        result, error_message, attempts = self._call_with_retries(
            provider, system_prompt, prompt_text, record
        )
        record.attempt_count = attempts

        if result is None:
            record.execution_status = "error"
            record.error_message = error_message
            record.json_valid = 0
            # Archive what we have so failures are inspectable too.
            self.archive.write_request(
                run_id,
                {
                    "provider": provider_name,
                    "model_id": model_config.get("model_id"),
                    "system": system_prompt,
                    "prompt": prompt_text,
                    "params": params,
                },
            )
            self.archive.write_response(run_id, {"error": error_message})
            self.archive.write_trace(run_id, {"error": error_message})
            self.archive.write_metadata(run_id, self._metadata(record))
            self.db.insert_run(record)
            self.stats["error"] += 1
            return f"ERR   {run_id} :: {(error_message or '')[:120]}"

        # Success path -----------------------------------------------------
        record.execution_status = "success"
        record.raw_response = result.raw_response_text
        record.response_id = result.response_id
        record.request_id = result.request_id
        record.trace_id = result.trace_id
        record.prompt_tokens = result.prompt_tokens
        record.completion_tokens = result.completion_tokens
        record.reasoning_tokens = result.reasoning_tokens
        record.total_tokens = result.total_tokens
        record.api_cost = provider.compute_cost(
            result.prompt_tokens, result.completion_tokens
        )

        parsed = parse_forecast(result.raw_response_text, prompt_id)
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

        self.archive.write_request(run_id, result.request_payload)
        self.archive.write_response(
            run_id,
            {"raw_text": result.raw_response_text, "response": result.response_payload},
        )
        self.archive.write_trace(run_id, result.trace)
        self.archive.write_metadata(run_id, self._metadata(record))
        self.db.insert_run(record)
        self.stats["success"] += 1

        if record.parsed_score_team_1 is not None and record.parsed_score_team_2 is not None:
            score = f"{record.parsed_score_team_1}-{record.parsed_score_team_2}"
        else:
            score = "no-score"
        valid = "ok" if record.json_valid else "invalid-json"
        return f"OK    {run_id} :: {score} ({valid}, {record.latency_ms:.0f}ms)"

    def _call_with_retries(
        self,
        provider: BaseProvider,
        system_prompt: str,
        prompt_text: str,
        record: RunRecord,
    ) -> tuple[ProviderResult | None, str | None, int]:
        last_error: str | None = None
        for attempt in range(1, self.config.max_retries + 1):
            req_utc, req_local = _now()
            start = time.perf_counter()
            try:
                result = provider.generate(system_prompt, prompt_text)
            except ProviderError as exc:
                last_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - defensive catch-all
                last_error = f"unexpected error: {exc!r}"
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                resp_utc, resp_local = _now()
                record.request_timestamp_utc = req_utc
                record.request_timestamp_local = req_local
                record.response_timestamp_utc = resp_utc
                record.response_timestamp_local = resp_local
                record.latency_ms = round(elapsed_ms, 3)
                return result, None, attempt

            # backoff before the next attempt
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_base_delay * (2 ** (attempt - 1)))
        # record timing of the failed attempt window
        record.request_timestamp_utc = record.request_timestamp_utc or req_utc
        record.request_timestamp_local = record.request_timestamp_local or req_local
        return None, last_error, self.config.max_retries

    @staticmethod
    def _metadata(record: RunRecord) -> dict[str, Any]:
        """Compact metadata view (ids, usage, timing, params, status)."""
        d = asdict(record)
        # Keep metadata focused: prompt/raw text live in request/response files.
        for big in ("prompt_text", "raw_response", "parsed_json", "system_prompt",
                    "white_hat", "red_hat", "black_hat", "yellow_hat",
                    "green_hat", "blue_hat"):
            d.pop(big, None)
        return d

    def close(self) -> None:
        self.db.close()
