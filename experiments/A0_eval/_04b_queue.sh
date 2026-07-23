#!/usr/bin/env bash
# Waits for the current night batch (PID 10520) to finish, then launches
# _run_04b_addon.sh. Spawned via setsid so it survives shell exit.
set -u
WATCH_PID="${1:-10520}"
ADDON="$(dirname "$0")/_run_04b_addon.sh"

echo "[queue $(date -u +%FT%TZ)] watching PID ${WATCH_PID}"
while kill -0 "${WATCH_PID}" 2>/dev/null; do
    sleep 60
done
echo "[queue $(date -u +%FT%TZ)] PID ${WATCH_PID} exited, launching ${ADDON}"
exec bash "${ADDON}"
