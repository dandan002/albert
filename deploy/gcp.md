# Running Albert on Google Cloud (Compute Engine)

The paper trader is a single long-running process with local state, so the simplest
reliable setup is **one small VM running the container with a persistent disk**. Cloud Run
is a poor fit: it scales to zero and has no durable local disk.

## What runs where

| Piece | Where |
|---|---|
| Image | Artifact Registry (`albert:<git sha>`) |
| Process | `albert paper run --strategy maker` in Docker, `--restart unless-stopped` |
| State | `/data` on the VM (`paper/state.json`, `paper/ledger.jsonl`, `paper/heartbeat.json`) |
| Logs | stdout JSON lines -> Cloud Logging (Container-Optimized OS ships them automatically) |
| Health | Docker `HEALTHCHECK` on the heartbeat file (stale after 10 min) |
| Secrets (Phase 6 only) | Secret Manager -> env file on the VM at start-up; never in the image |

Paper maker mode needs no research tables: it reads live Kalshi markets, order books and
trades. The `elo` strategy needs settled results for its 20 series (`paper elo-bootstrap`,
once; it then refreshes hourly into `/data/tables/elo_markets.parquet`). Research and backtests can stay on a laptop, or run on the same VM with more disk.

## One-time setup

```bash
PROJECT=<your-project>  REGION=us-east4  ZONE=us-east4-a   # us-east is closest to Kalshi
gcloud config set project $PROJECT
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com logging.googleapis.com
gcloud artifacts repositories create albert --repository-format=docker --location=$REGION

# Build and push (from the repo root). Cloud Build avoids cross-arch issues from Apple silicon.
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/albert/albert:$(git rev-parse --short HEAD)
```

## Create the VM

```bash
IMAGE=$REGION-docker.pkg.dev/$PROJECT/albert/albert:$(git rev-parse --short HEAD)
gcloud compute instances create albert-paper \
  --zone=$ZONE --machine-type=e2-small \
  --image-family=cos-stable --image-project=cos-cloud \
  --boot-disk-size=20GB \
  --scopes=cloud-platform \
  --metadata=google-logging-enabled=true

gcloud compute ssh albert-paper --zone=$ZONE
# on the VM:
docker-credential-gcr configure-docker --registries=$REGION-docker.pkg.dev
sudo mkdir -p /var/albert && sudo chown 10001 /var/albert
docker run --rm -v /var/albert:/data $IMAGE paper reset --kalshi-cash 50 --pmus-cash 50 --yes
docker run --rm -v /var/albert:/data $IMAGE paper elo-bootstrap --since 2026-08-01   # Elo history
docker run -d --name albert --restart unless-stopped -v /var/albert:/data $IMAGE \
  paper run --strategy maker --strategy elo --interval 60
```

`/var/albert` on a COS boot disk survives reboots. For extra safety, snapshot the disk on a
schedule, or copy `ledger.jsonl` to a GCS bucket daily.

## Operate

```bash
docker logs -f albert                                    # JSON line per tick
docker run --rm -v /var/albert:/data $IMAGE paper status # account, positions, resting orders
docker stop albert                                       # SIGTERM: finishes the tick, saves state
```

Upgrade: build and push a new tag, then `docker rm -f albert` and `docker run` again with the
new image. State is in `/data`, so nothing is lost.

## Before live trading (Phase 6)

- Order clients with per-order and daily caps and a kill switch (not built yet).
- Credentials: create secrets for the Kalshi key id/private key, write them to a root-only
  env file at boot (`gcloud secrets versions access`), pass `--env-file`. Never bake them into
  the image or log them.
- Run the VM's clock with NTP (default on GCE), because Kalshi request signatures are
  timestamped.
