#!/usr/bin/env bash
# Continuous L2 recording: one `albert record` run per day, each re-selecting the market universe
# (markets closing within 7 days). Stop with: touch data/record.stop   (or kill the loop's PID).
# On macOS, wrap in `caffeinate -is` so the machine does not sleep.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
rm -f data/record.stop
echo $$ > data/record.pid
while [ ! -f data/record.stop ]; do
  "$PY" -m albert record --minutes "${RECORD_MINUTES:-1440}" "$@" >> data/record.log 2>&1
  status=$?
  if grep -q '"stopped_reason": "disk' "$(ls -t data/runs/*-record.json | head -1)"; then
    echo "$(date -u +%FT%TZ) stopped: low disk" >> data/record.log; break
  fi
  [ $status -ne 0 ] && sleep 60
done
rm -f data/record.pid
