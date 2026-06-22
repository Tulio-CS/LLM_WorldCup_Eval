#!/usr/bin/env python
"""Convenience entry point so you can run the framework without -m.

    python run.py run --dry-run
    python run.py info
    python run.py export

is equivalent to `python -m fifa_forecast ...`.
"""

from fifa_forecast.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
