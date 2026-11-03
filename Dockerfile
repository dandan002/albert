# Albert runtime image: `albert paper run` (default) or any other `albert` command.
# State, ledger and caches live in /data (mount a persistent disk there). No credentials are
# baked in: pass them at run time (e.g. --env-file from Secret Manager) when live trading exists.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ALBERT_DATA_DIR=/data \
    ALBERT_RECORD_LIVE=0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 albert \
    && mkdir -p /data && chown albert /data

USER albert
VOLUME ["/data"]

# Healthy if the paper loop wrote a heartbeat in the last 10 minutes.
HEALTHCHECK --interval=5m --timeout=10s --start-period=5m \
  CMD python -c "import json,sys,datetime as d; h=json.load(open('/data/paper/heartbeat.json')); sys.exit(0 if (d.datetime.now(d.UTC)-d.datetime.fromisoformat(h['ts'])).total_seconds()<600 else 1)"

ENTRYPOINT ["albert"]
CMD ["paper", "run", "--strategy", "maker", "--interval", "60"]
