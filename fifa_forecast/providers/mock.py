"""Deterministic mock provider for --dry-run.

Produces well-formed, schema-correct responses without any network call, so the
full pipeline (parsing, storage, archive, export) can be exercised offline.
Output is deterministic in the prompt text, which keeps dry-runs reproducible
while still varying across matches and orderings.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .base import BaseProvider, ProviderResult


def _seed_from(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class MockProvider(BaseProvider):
    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        wants_prob = "team1_win_probability" in user_prompt
        wants_hats = "white_hat" in user_prompt

        rng = random.Random(_seed_from(self.model_id, user_prompt))

        s1 = rng.randint(0, 4)
        s2 = rng.randint(0, 4)
        payload: dict[str, Any] = {"score_team_1": s1, "score_team_2": s2}

        if wants_prob:
            # Probabilities loosely informed by the scoreline, summing to 100.
            base = [40, 25, 35]
            if s1 > s2:
                base = [60, 20, 20]
            elif s2 > s1:
                base = [20, 20, 60]
            else:
                base = [30, 40, 30]
            jitter = [rng.randint(-8, 8) for _ in range(3)]
            vals = [max(1, b + j) for b, j in zip(base, jitter)]
            total = sum(vals)
            p1 = round(vals[0] * 100 / total)
            pd = round(vals[1] * 100 / total)
            p2 = 100 - p1 - pd
            payload["team1_win_probability"] = p1
            payload["draw_probability"] = pd
            payload["team2_win_probability"] = p2

        if wants_hats:
            filler = (
                "This is mock analysis generated offline for pipeline testing; "
                "it contains no real football insight but satisfies the minimum "
                "length requirement of at least thirty words for each thinking hat."
            )
            payload["white_hat"] = filler
            payload["red_hat"] = filler
            payload["black_hat"] = filler
            payload["yellow_hat"] = filler
            payload["green_hat"] = filler
            payload["blue_hat"] = (
                "Synthesising the five prior hats — facts, intuition, risks, "
                "strengths and creative scenarios — this mock blue hat reconciles "
                "them into the final scoreline and probabilities above for testing."
            )

        text = json.dumps(payload)
        prompt_tokens = len(user_prompt) // 4
        completion_tokens = len(text) // 4

        request_payload = {
            "provider": self.name,
            "model": self.model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "params": self.params,
        }
        response_payload = {"text": text, "mock": True}

        return ProviderResult(
            raw_response_text=text,
            request_payload=request_payload,
            response_payload=response_payload,
            response_id=f"mock-{_seed_from(self.model_id, user_prompt):x}",
            request_id=f"mockreq-{_seed_from(user_prompt, self.model_id):x}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=None,
            total_tokens=prompt_tokens + completion_tokens,
            model_reported=self.model_id,
            trace={"mock": True},
        )
