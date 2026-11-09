#!/usr/bin/env bash
# Live maker with REAL orders (Kalshi prod): one `albert live l2` process per week. The tag (one
# letter) names the maker; its runs live in data/live/prod/<tag>/. Arguments: the tag, then the
# `albert live l2` options (see deploy/live-maker-<name>.conf).
# The loop does NOT restart a maker that halted for a safety reason (loss limit, reconciliation
# mismatch, order errors, foreign or taker fill, kill switch): a tripped guard stays tripped until
# a person looks. A crash (non-zero exit) is retried after 5 minutes.
# Stop every live maker (and halt quoting): touch data/live/STOP
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
TAG="$1"; shift
DIR="data/live/prod/$TAG"
mkdir -p "$DIR"
while [ ! -f data/live/STOP ]; do
  "$PY" -m albert live l2 --env prod --live --tag "$TAG" --minutes "${LIVE_MINUTES:-10080}" "$@" >> "$DIR/loop.log" 2>&1
  status=$?
  last=$(ls -d "$DIR"/2*/ 2>/dev/null | tail -1)
  halted=$("$PY" -c "import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
end=next((r for r in reversed(rows) if r.get('kind')=='end'),{})
print(end.get('halted') or '')" "${last}ledger.jsonl" 2>/dev/null)
  if [ -n "$halted" ] && [ "$halted" != "shutdown" ]; then
    echo "$(date -u +%FT%TZ) halted: $halted; not restarting (fix, then systemctl restart)" >> "$DIR/loop.log"
    exit 0
  fi
  [ $status -ne 0 ] && sleep 300
done
