#!/usr/bin/env bash
# Continuous live paper maker (no real orders): one `albert paper l2` run per day, each
# re-selecting its markets. PAPER_OUT names the folder under data/ (default paper_l2; the
# illiquid universe uses paper_l2_illiquid). Stop (and halt quoting) with: touch data/$PAPER_OUT/STOP
# On macOS, wrap in `caffeinate -is` so the machine does not sleep.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OUT="${PAPER_OUT:-paper_l2}"
DIR="data/$OUT"
mkdir -p "$DIR"
rm -f "$DIR/STOP"
echo $$ > "$DIR/loop.pid"
while [ ! -f "$DIR/STOP" ]; do
  "$PY" -m albert paper l2 --out "$OUT" --minutes "${PAPER_MINUTES:-1440}" "$@" >> "$DIR/loop.log" 2>&1
  status=$?
  [ $status -ne 0 ] && sleep 60
done
rm -f "$DIR/loop.pid"
