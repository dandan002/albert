#!/usr/bin/env bash
# Continuous live paper maker (no real orders): one `albert paper l2` run per day, each
# re-selecting its markets. Stop (and halt quoting) with: touch data/paper_l2/STOP
# On macOS, wrap in `caffeinate -is` so the machine does not sleep.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
mkdir -p data/paper_l2
rm -f data/paper_l2/STOP
echo $$ > data/paper_l2/loop.pid
while [ ! -f data/paper_l2/STOP ]; do
  "$PY" -m albert paper l2 --minutes "${PAPER_MINUTES:-1440}" "$@" >> data/paper_l2/loop.log 2>&1
  status=$?
  [ $status -ne 0 ] && sleep 60
done
rm -f data/paper_l2/loop.pid
