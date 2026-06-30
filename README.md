# FIFA World Cup 2026 — AI Forecast Benchmark (Data Collection)

A reproducible framework for collecting LLM match forecasts for the FIFA World
Cup 2026. For every match it executes multiple models, multiple prompt
strategies, both team orderings and many repetitions, then preserves **every**
raw request, raw response, API parameter, provider trace and parsed prediction
into a SQLite dataset, an Excel workbook and an on-disk JSON archive.

This project is **data collection and storage only** — it computes no accuracy
or evaluation metrics. The resulting dataset is structured so that accuracy,
Brier score, calibration, consistency, variance, prompt sensitivity and
team-order sensitivity can all be computed later from the stored fields.

## What gets collected

For each execution (one model call) the framework stores:

- the rendered prompt and the shared system prompt
- the raw response text and the full serialized provider response
- the parsed prediction (scores, win/draw/win probabilities, and the six
  thinking-hat analyses where applicable)
- every API parameter actually used (model, temperature, top_p, seed,
  max_tokens, reasoning effort, response format, …)
- provider trace/metadata: response id, request id, token usage
  (prompt / completion / reasoning / total), and computed USD cost
- UTC + local timestamps and call latency
- validity flag, validation notes, execution status and error message

## The experiment grid

```
matches  ×  team orderings  ×  models  ×  prompt strategies  ×  repetitions
  (32)         (2)             (8)            (3)                 (10)        = 15,360 calls
```

- **Models** (configurable): GPT-5 / GPT-5-mini / GPT-5-nano (OpenAI),
  Claude Opus / Claude Sonnet (Anthropic), Gemini 2.5 Pro / Flash (Google),
  Grok (xAI).
- **Prompt strategies**: `simple-prediction`, `probability-prediction`,
  `six-hats-prediction` (de Bono's Six Thinking Hats).
- **Team-order sensitivity**: every match is run as `original`
  (team_1 vs team_2) and `reversed` (team_2 vs team_1). The model's
  `score_team_1` always refers to the team presented **first**, captured as
  `prompt_team_1` / `prompt_team_2`, while `team_1` / `team_2` stay canonical.
- **Repetitions**: `RUNS_PER_COMBINATION = 10` (configurable).

## Install

```bash
pip install -r requirements.txt
```

`pandas`, `openpyxl` and `python-dotenv` are always required. The provider SDKs
(`openai`, `anthropic`, `google-genai`) are needed only for real runs of the
corresponding provider — `--dry-run` needs none of them.

## Configure

### API keys

Put credentials in a `.env` file in the project root (auto-loaded):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...      # Grok
```

`.env` is git-ignored. Never commit real keys (including in `.env.example`).

### Experiment settings & model catalog

Defaults live in [`fifa_forecast/config.py`](fifa_forecast/config.py) — the
model catalog, prompt set, repetitions, retry policy and pricing. Override any
of them **without editing code** by creating a `config.json` in the project
root, e.g.:

```json
{
  "runs_per_combination": 3,
  "prompt_ids": ["simple-prediction", "probability-prediction"],
  "models": [
    {"key": "gpt-5", "provider": "openai", "model_id": "gpt-5",
     "enabled": true, "api_key_env": "OPENAI_API_KEY",
     "params": {"reasoning_effort": "medium", "max_output_tokens": 8000},
     "pricing": {"input": 1.25, "output": 10.0}}
  ]
}
```

Add a new model by adding an entry to `models` — no code changes required.

> **Adjust model IDs / pricing for your account.** `model_id` strings
> (`gpt-5`, `grok-4`, …) and the `pricing` values must match what your provider
> account actually exposes. Anthropic prices are pre-filled; others default to
> `null`, in which case `api_cost` is stored as `NULL`. Per-provider parameter
> rules are encoded in the adapters (e.g. Claude Opus 4.8 / Sonnet 4.6 reject
> `temperature`/`top_p`, so depth is set via `effort`).

## Run

```bash
# Offline smoke test — deterministic mock provider, no keys, no network:
python -m fifa_forecast run --dry-run --limit-matches 1

# See the planned execution count without running:
python -m fifa_forecast info

# Full collection (uses real APIs and incurs cost):
python -m fifa_forecast run

# Rebuild the Excel workbook from the existing database:
python -m fifa_forecast export

# Write the reproducibility manifest only:
python -m fifa_forecast manifest

# Analyze what's collected so far (read-only; works on a partial/interrupted run):
python -m fifa_forecast report

# Project USD cost from observed token usage (e.g. 62 matches, 3 forecasts each):
python -m fifa_forecast estimate --matches 62 --occasions 3
```

### Cost estimation

`estimate` projects spend by combining the **observed** average token usage per
model × prompt (from `fifa_forecasts.db`) with the per-token `pricing` in
[config.py](fifa_forecast/config.py) and the planned grid size:

```
calls = orderings × reps × occasions × matches
```

`--occasions` is how many times each match is forecast (e.g. 3 = 1 pre-match +
2 in-play). Anthropic prices are authoritative; **OpenAI / Gemini / xAI prices
in config are estimates — verify them** before quoting totals. Tune with
`--reps`, `--orders`, `--occasions`, `--matches`.

### Adding next-phase matches and actual results

**Next-phase fixtures:** append the new games as rows to the matches CSV (same
format — `team_1,team_2,stage[,datetime_utc_minus_03]`). `match_id` is assigned
by row order, so appending keeps existing ids stable. Then `run` (it resumes,
skipping completed combinations). Point `matches_csv` at a different file via
`config.json` if you prefer to keep phases in separate files.

**Actual results (ground truth):** stored in a separate `match_results` table
and joined back to predictions by `match_id` — predictions are never mixed with
outcomes. Workflow:

```bash
python -m fifa_forecast results-template --out results.csv   # blank row per match
# fill actual_score_team_1 / actual_score_team_2 (status auto-set to "finished")
python -m fifa_forecast add-results --csv results.csv        # ingest + refresh Excel
```

`add-results` upserts by `match_id` (safe to re-run as games finish). The Excel
workbook gains a **Results** sheet and an **Eval Base** sheet — every prediction
left-joined to its actual score and `actual_winner` (`team_1`/`team_2`/`draw`,
canonical order), ready for accuracy/Brier/calibration analysis later.

**Automatic fetching (football-data.org):** the `fetch` command pulls fixtures
and finished scores from a sports API and maps them to your `match_id`s
(accent/alias-normalized team names, unordered pairing). Get a free key at
<https://www.football-data.org/> (free tier includes the World Cup), add
`FOOTBALL_DATA_API_KEY=...` to `.env`, then:

```bash
python -m fifa_forecast fetch --results --dry-run   # preview the score mapping
python -m fifa_forecast fetch --results             # ingest finished scores
python -m fifa_forecast fetch --fixtures            # append newly-scheduled games to the CSV
python -m fifa_forecast fetch --results --fixtures  # both
```

Games the API reports but can't be matched to your CSV (name mismatch, or not
present) are listed, never guessed — add an alias in `fetch.py` `_ALIASES` or
the game to the matches CSV. `--dry-run` writes nothing.

### Comparing forecasts to actual results

Once results are ingested, score the AI against reality for **finished** matches:

```bash
python -m fifa_forecast evaluate              # console leaderboard + Excel report
python -m fifa_forecast evaluate --no-excel   # console only
```

It joins valid predictions to finished results, maps every prediction back to
canonical team order (so `original`/`reversed` runs are comparable), and reports
per model / prompt / moment: **outcome accuracy**, **exact-score accuracy**,
mean goal-difference & total-goals error, and the **3-way Brier score**
(probability/six-hats prompts). Output: console + `FIFA_Evaluation_Report.xlsx`,
and the dashboard's **Forecast vs results** page (`/evaluate`).

### Quality & variability report

`report` reads the existing `fifa_forecasts.db` (no API calls) and answers:

- **Counts & errors** — totals, per-model/prompt success vs error, error
  signatures (e.g. provider 429/529) grouped and counted.
- **Prompt quality** — per (model, prompt): JSON-validity rate, probabilities
  summing to 100, all six hats present and ≥30 words, avg tokens/latency.
- **Variability & convergence** — per model, the within-combination spread of
  goal difference and win probability across repetitions, winner-agreement, and
  **how many repetitions are needed before the estimate stops moving**.

Repetition recommendation: a *combination* is `model × match × prompt ×
team-ordering`; its repetitions are the samples. `recommended_reps` is the CI
formula `n = (1.96·σ/margin)²` (σ = median within-combination std), with an
empirical convergence curve as corroboration. Tune the targets:

```bash
python -m fifa_forecast report --prob-margin 5 --gd-margin 0.3 --shuffles 40
python -m fifa_forecast report --no-excel        # console only
```

Output: console digest + `data/exports/FIFA_Quality_Variability_Report.xlsx`
(sheets: Overview, By Model, By Prompt, Prompt Quality, Errors, Variability by
Model, Combination Stats, Recommended Reps, Convergence curves).

Run only selected matches by **date** (the local UTC-3 date shown in the CSV)
or by **match id** — both repeatable and available on `info` too:

```bash
python -m fifa_forecast info --date 2026-06-22          # preview the day's matches + call count
python -m fifa_forecast run  --date 2026-06-22           # only the 22 Jun fixtures
python -m fifa_forecast run  --date 2026-06-22 --date 2026-06-23   # two days
python -m fifa_forecast run  --match-id 1 --match-id 2   # specific matches
```

Useful flags for `run`: `--date YYYY-MM-DD`, `--match-id ID`,
`--limit-matches N`, `--overwrite`, `--no-export`, `--config path/to/config.json`.
(`python run.py ...` works the same as `python -m fifa_forecast ...`.)

**Resume is automatic.** Run ids are deterministic in the combination, so a
re-run skips combinations already in the database and never overwrites existing
outputs. Use `--overwrite` to force re-execution. Failed calls are retried up to
`max_retries` (default 3, exponential backoff) and, if still failing, recorded
as error rows — nothing is silently dropped.

## Outputs

| Deliverable | Location |
|---|---|
| SQLite dataset | `fifa_forecasts.db` (table `forecast_runs`) |
| Excel workbook | `data/exports/FIFA_World_Cup_2026_AI_Forecasts.xlsx` |
| Raw request archive | `data/raw_requests/<run_id>.json` |
| Raw response archive | `data/raw_responses/<run_id>.json` |
| Trace archive | `data/traces/<run_id>.json` |
| Per-execution metadata | `data/metadata/<run_id>.json` |
| Reproducibility manifest | `experiment_manifest.json` |

Each execution thus produces its request, response, trace and metadata JSON
files, keyed by a unique `run_id`.

**Excel worksheets:** `Forecasts` (one row per execution), `Matches`, `Models`,
`Prompts`, `Raw Responses`, `API Metadata`, `Errors`, `Summary` (counts only),
plus `Counts by Model / Prompt / Order` breakdowns.

The manifest records execution date, git commit, software/prompt versions,
installed SDK versions and environment info.

## Input matches

The loader reads `fifa_world_cup_2026_future_matches.csv`. It accepts either the
internal schema (`match_id, phase, team_1, team_2, kickoff_datetime`) or the
shipped fixtures schema (`datetime_utc_minus_03, team_1, team_2, stage`),
auto-generating `match_id`, mapping `stage`→`phase`, and converting the UTC-3
kickoff time to ISO-8601 UTC.

## Project layout

```
fifa_forecast/
  config.py        model catalog, settings, config.json overrides
  prompts.py       the three prompt strategies (versioned)
  matches.py       CSV ingestion + team-order helper
  parsing.py       JSON extraction + validation (never raises)
  storage.py       SQLite schema + JSON file archive
  manifest.py      reproducibility manifest
  runner.py        the experiment grid + retry/resume
  export.py        Excel workbook builder
  cli.py           command-line interface
  providers/       openai · anthropic · gemini · grok · mock + registry
```
