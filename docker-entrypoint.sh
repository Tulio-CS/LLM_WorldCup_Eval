#!/bin/sh
# Seed the persistent volume on first boot, then run the given command.
set -e

DATA_DIR="${FIFA_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"

# Copy the bundled fixtures into the volume once, so `fetch --fixtures` appends
# persist across redeploys. The repo copy is the seed; the volume copy is canonical.
SEED="/app/fifa_world_cup_2026_future_matches.csv"
DEST="${FIFA_MATCHES_CSV:-$DATA_DIR/matches.csv}"
if [ ! -f "$DEST" ] && [ -f "$SEED" ]; then
  cp "$SEED" "$DEST"
  echo "[entrypoint] seeded matches CSV -> $DEST"
fi

exec "$@"
