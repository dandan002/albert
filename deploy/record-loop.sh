#!/usr/bin/env bash
# Continuous L2 recording: one run per day, each re-selecting the market universe.
#   RECORD_CMD=record      (default) `albert record`: random sample (markets closing within 7 days)
#   RECORD_CMD=record-vol  `albert record-vol`: complete vol-study events, refreshed hourly in-run
# Stop with: touch data/$RECORD_CMD.stop   (or kill the loop's PID in data/$RECORD_CMD.pid).
# On macOS, wrap in `caffeinate -is` so the machine does not sleep.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
CMD="${RECORD_CMD:-record}"
rm -f "data/$CMD.stop"
echo $$ > "data/$CMD.pid"
while [ ! -f "data/$CMD.stop" ]; do
  "$PY" -m albert "$CMD" --minutes "${RECORD_MINUTES:-1440}" "$@" >> "data/$CMD.log" 2>&1
  status=$?
  if grep -q '"stopped_reason": "disk' "$(ls -t data/runs/*-"$CMD".json | head -1)"; then
    echo "$(date -u +%FT%TZ) stopped: low disk" >> "data/$CMD.log"; break
  fi
  [ $status -ne 0 ] && sleep 60
done
rm -f "data/$CMD.pid"
