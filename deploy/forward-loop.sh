#!/usr/bin/env bash
# Forward paper test of the three fundamental leads (no orders anywhere): research/forward.py.
# Every hour at HH:05 UTC it runs each lead's snapshot; each lead decides itself whether it is
# due and records each key at most once (mentions: speech events starting in 5-7 h, i.e. seen
# at 6-7 h; rain: the 12:05 UTC run, today's KXRAIN books <= 3c; FCS: college-football games
# kicking off in 5-7 h, from the football milestones). A failed run is retried after 2, 4, 8,
# 16 and 30 min (then every 30 min until one succeeds); snapshots are idempotent, so a retry
# never records anything twice.
# Score later (not done here): python3 -m albert.research.forward score all
# Stop with: touch data/cache/forward/STOP   (checked every 30 s)
# On macOS, wrap in `caffeinate -is` so the machine does not sleep.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
# Leads to snapshot (space-separated); e.g. LEADS="mentions fcs" to leave one out.
LEADS="${LEADS:-mentions rain fcs}"
DIR=data/cache/forward
mkdir -p "$DIR"
rm -f "$DIR/STOP"
echo $$ > "$DIR/loop.pid"
fails=0
while [ ! -f "$DIR/STOP" ]; do
  status=0
  for lead in $LEADS; do
    nice -n 10 "$PY" -m albert.research.forward snapshot "$lead" >> "$DIR/loop.log" 2>&1 || status=$?
  done
  if [ $status -ne 0 ]; then
    fails=$((fails + 1))
    wait=$(( 120 * (1 << (fails < 5 ? fails - 1 : 4)) ))
    [ $wait -gt 1800 ] && wait=1800
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) loop: snapshot exit $status ($fails in a row); retry in $((wait / 60)) min" >> "$DIR/loop.log"
  else
    fails=0
    now=$(date -u +%s)
    wait=$(( (now / 3600 + 1) * 3600 + 300 - now ))     # until the next HH:05 UTC
    [ $wait -gt 3600 ] && wait=$(( wait - 3600 ))
  fi
  end=$(( $(date -u +%s) + wait ))
  while [ ! -f "$DIR/STOP" ] && [ "$(date -u +%s)" -lt "$end" ]; do sleep 30; done
done
rm -f "$DIR/loop.pid"
