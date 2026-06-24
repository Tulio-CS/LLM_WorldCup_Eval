"""Prompt strategy catalog.

Three strategies, each a template rendered with the two team names in the order
they are presented to the model. The exact wording is fixed and versioned so the
dataset is reproducible; ``PROMPT_VERSION`` is recorded in the manifest.

``render(prompt_id, team1, team2)`` returns the user-message text. A shared
system prompt is also provided.
"""

from __future__ import annotations

PROMPT_VERSION = "1.1.0"

SYSTEM_PROMPT = (
    "You are an expert football (soccer) analyst forecasting FIFA World Cup "
    "2026 matches. Follow the output format exactly and return only what is "
    "requested."
)

# --------------------------------------------------------------------------- #
# Match moments — the same match is forecast at three points in time. The
# kickoff time and the moment are injected as a context header on top of every
# prompt so the model knows *when* the forecast is being captured.
# --------------------------------------------------------------------------- #
MATCH_MOMENTS = ("pre_match", "halftime", "post_match")

MOMENT_LABELS: dict[str, str] = {
    "pre_match": "before kickoff, with the match not yet started",
    "halftime": "during the half-time break, with the match in progress",
    "post_match": "right after the final whistle, with the match finished",
}


def context_header(kickoff: str | None, moment: str | None) -> str:
    """Build the context block prepended to every prompt.

    ``kickoff`` is the scheduled match time (local, UTC-3) and ``moment`` is one
    of :data:`MATCH_MOMENTS`. Returns "" when neither is provided so the legacy
    prompt wording is preserved.
    """
    lines: list[str] = []
    if kickoff:
        lines.append(f"Scheduled kickoff time (local, UTC-3): {kickoff}.")
    if moment:
        label = MOMENT_LABELS.get(moment, moment)
        lines.append(f"You are making this forecast {label}.")
    if not lines:
        return ""
    return "Context:\n" + "\n".join(lines) + "\n\n"

SIMPLE_PREDICTION = """You are forecasting a FIFA World Cup 2026 match.

Predict the result of the match {team1} vs {team2}.

Return ONLY a valid JSON object:

{{
  "score_team_1": <integer>,
  "score_team_2": <integer>
}}"""

PROBABILITY_PREDICTION = """You are forecasting a FIFA World Cup 2026 match.

Predict the result the match {team1} vs {team2}.

Return ONLY a valid JSON object:

{{
  "score_team_1": <integer>,
  "score_team_2": <integer>,
  "team1_win_probability": <integer>,
  "draw_probability": <integer>,
  "team2_win_probability": <integer>
}}

Requirements:
- Scores must be non-negative integers.
- Probabilities must be integers between 0 and 100.
- The three probabilities must sum exactly to 100.
- Output only the JSON object and nothing else."""

SIX_HATS_PREDICTION = """You are forecasting a FIFA World Cup 2026 match.

Predict the result of the match {team1} vs {team2}.

In addition to the prediction, provide a justification using Edward de Bono's Six Thinking Hats framework:

* White Hat: objective facts, statistics, rankings, form, and evidence.
* Red Hat: intuition, emotions, momentum, and subjective impressions.
* Black Hat: risks, weaknesses, uncertainties, and reasons the prediction could be wrong.
* Yellow Hat: strengths, opportunities, and reasons supporting the predicted outcome.
* Green Hat: creative, unconventional, or unexpected scenarios that could influence the match.
* Blue Hat: process control, synthesis, and final judgment. Integrate insights from all previous hats, resolve conflicts among them, and explain how they lead to the final score prediction and probabilities.

Return ONLY a valid JSON object in the following format:

{{
"score_team_1": <integer>,
"score_team_2": <integer>,
"team1_win_probability": <integer>,
"draw_probability": <integer>,
"team2_win_probability": <integer>,

"white_hat": "<analysis>",
"red_hat": "<analysis>",
"black_hat": "<analysis>",
"yellow_hat": "<analysis>",
"green_hat": "<analysis>",
"blue_hat": "<analysis>"
}}

Requirements:

* Scores must be non-negative integers.
* Probabilities must be integers between 0 and 100.
* The three probabilities must sum exactly to 100.
* Each hat must contain at least 30 words.
* The blue_hat must explicitly explain how the previous five hats were combined to produce the final forecast.
* Output only the JSON object and nothing else."""


PROMPT_TEMPLATES: dict[str, str] = {
    "simple-prediction": SIMPLE_PREDICTION,
    "probability-prediction": PROBABILITY_PREDICTION,
    "six-hats-prediction": SIX_HATS_PREDICTION,
}

# Which strategies request probability fields / six-hats fields. Used by the
# parser to decide which fields are expected.
PROMPTS_WITH_PROBABILITIES = {"probability-prediction", "six-hats-prediction"}
PROMPTS_WITH_HATS = {"six-hats-prediction"}

HAT_FIELDS = (
    "white_hat",
    "red_hat",
    "black_hat",
    "yellow_hat",
    "green_hat",
    "blue_hat",
)


def render(
    prompt_id: str,
    team1: str,
    team2: str,
    *,
    kickoff: str | None = None,
    moment: str | None = None,
) -> str:
    """Render a prompt template with the team names in presentation order.

    When ``kickoff`` and/or ``moment`` are supplied, a context header stating the
    match time and the forecasting moment is prepended (see
    :func:`context_header`).
    """
    try:
        template = PROMPT_TEMPLATES[prompt_id]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown prompt_id: {prompt_id!r}") from exc
    body = template.format(team1=team1, team2=team2)
    return context_header(kickoff, moment) + body


def prompt_catalog() -> list[dict[str, str]]:
    """Structured description of all prompts (for the Excel 'Prompts' sheet)."""
    return [
        {
            "prompt_id": pid,
            "version": PROMPT_VERSION,
            "requests_probabilities": pid in PROMPTS_WITH_PROBABILITIES,
            "requests_six_hats": pid in PROMPTS_WITH_HATS,
            "template": template,
        }
        for pid, template in PROMPT_TEMPLATES.items()
    ]
