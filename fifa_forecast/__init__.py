"""FIFA World Cup 2026 AI Forecast Benchmark — data collection framework.

This package executes LLM match forecasts across multiple models, prompt
strategies, team orderings and repetitions, and preserves every raw response,
API parameter, trace and parsed prediction into a reproducible dataset
(SQLite + Excel + on-disk JSON archive).

The package is intentionally focused on *data collection and storage only* —
no statistical evaluation is performed here.
"""

__version__ = "1.0.0"
