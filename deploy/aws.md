# AWS host (us-east-2, same region as Kalshi's exchange)

- Instance `albert` = `i-01990679def3bfdf4` (m7i-flex.large: the account is on the AWS Free plan,
  which only allows free-tier types; Ubuntu 24.04, encrypted 100 GB gp3, IMDSv2 only), Elastic IP
  16.59.235.153. Key pair `albert` = the owner's `~/.ssh/id_ed25519`. Clock: chrony on Amazon Time
  Sync (169.254.169.123).
- **Access: Systems Manager Session Manager only.** Security group `albert-ssh` has NO inbound rules
  (port 22 closed). Instance profile `albert-ec2-ssm` (AmazonSSMManagedInstanceCore). On the Mac:
  Session Manager plugin in `~/.local/sessionmanagerplugin` (symlink `~/.local/bin`), and
  `~/.ssh/config` host `albert` (ProxyCommand `aws ssm start-session ... AWS-StartSSHSession`,
  HostKeyAlias 16.59.235.153). So: `ssh albert`, `rsync ... albert:albert/`, or a plain shell with
  `aws ssm start-session --region us-east-2 --target i-01990679def3bfdf4`.
- Code in `~/albert` (venv `.venv`), secrets `~/albert/.env` (mode 600).
- Services (systemd, `sudo systemctl status albert-*`):
  - `albert-record` (deploy/record-loop.sh --paired-days 3 --skip-vol) → data/raw/stream
  - `albert-record-vol` (RECORD_CMD=record-vol deploy/record-loop.sh --ladder-interval-ms 1000) → data/raw/stream: the
    vol-study families (`stream.vol_family`), never recorded by `albert-record` as well.
    Disk floors: record-vol stops below 20 GB free, record below 10 GB (`--min-free-gb`); while
    under the floor systemd restarts it every minute and it exits at once ("stopped: low disk" in
    data/record-vol.log). That is what stopped vol recording 2026-11-18 03:12 → 11-20.
  - `albert-record-hl` (RECORD_CMD=record-hl deploy/record-loop.sh; since 2026-11-20) →
    data/raw/stream/hl: Hyperliquid HIP-4 outcome books and trades, non-financial outcomes only,
    side 0, new markets added hourly on the same 1-2 connections (Hyperliquid allows 10
    connections and 1000 subscriptions per IP). About 150 MB a day. Public data, never traded.
  - `albert-archive.timer` (installed 2026-11-20) → `albert-archive.service` (daily 08:00 UTC,
    `albert archive-stream --keep-days 4`): closed raw stream days → S3 bucket
    `albert-archive-091914201487` (us-east-2; Block Public Access on, bucket-owner-enforced, versioned,
    SSE-S3 with a bucket key, TLS-only bucket policy, tag `project=albert`), keys
    `raw/stream/<venue>/<date>/<file>`, uploaded as `INTELLIGENT_TIERING` (no retrieval fee or
    restore step; 30 days untouched → Infrequent Access, 90 → Archive Instant Access). Lifecycle: a
    day-0 transition of `raw/` to Intelligent-Tiering (safety net) and abort of stale multipart
    uploads after 7 days. The instance role `albert-ec2-ssm` has an inline policy `albert-archive-s3`:
    Put/Get on `raw/stream/*` and List on the bucket, nothing else (no delete). The unit sets
    `AWS_CONFIG_FILE=/dev/null` so it always runs as that role, never as an `aws login` session left
    in `~/.aws`. Needs boto3 (`pip install -e '.[archive]'`). Restore a day:
    `albert restore-stream <date> [--venue pm]`; research readers raise `ArchivedError` on an
    archived day instead of reading fewer files. Run it now: `sudo systemctl start albert-archive`.
  - `albert-mentions@prod` (REAL ORDERS, running) with deploy/mentions-prod.conf containing
    `MENTIONS_ARGS=--live ...` AND `KALSHI_LIVE_ENABLED=1` in .env (both required);
    `albert-mentions@paper` (running) alongside it on the same markets with simulated fills.
  - Stopped, units kept: `albert-paper` (deploy/paper-loop.sh → data/paper_l2), `albert-paper-illiquid`,
    `albert-forward` (deploy/forward-loop.sh → data/cache/forward), `albert-mentions@demo`
    (deploy/mentions-demo.conf), `albert-live-maker@{liquid,illiquid}` (real orders; held off by
    `data/live/STOP`).
  - `albert-guard@prod` (albert live guard; args in deploy/guard-prod.conf): account-wide daily-loss
    halt and at-risk kill switch over every bot, exchange-sourced fill audit in data/live/audit/prod.
    Global kill: `albert live kill [--live]`; undo `albert live resume`; status `albert live check --env prod`.
- Install or update a unit: `sudo cp deploy/systemd/<unit> /etc/systemd/system/ && sudo systemctl
  daemon-reload`. The services run the working tree in `~/albert`, so a code change takes effect on
  restart. Restarting `albert-mentions@prod` cancels its resting orders, so check
  `albert live check --env prod` first and re-check Kalshi fees (`GET /series/fee_changes`).
- Update code: this host is the primary working copy (since 2026-11-07). From the Mac (older
  backup): `rsync -az --exclude data/ --exclude .venv/ --exclude .env ./ albert:albert/` then
  `sudo systemctl restart albert-...`.
- Pull data back: `rsync -az albert:albert/data/raw/stream/ data/raw/stream/` (append-only).
- Emergency direct SSH (only if SSM is down): `aws ec2 authorize-security-group-ingress --region us-east-2
  --group-id sg-04dc0331e14485794 --protocol tcp --port 22 --cidr <your ip>/32`, then revoke it after.

## Kalshi hosts and latency (measured 2026-11-07 from this instance)

The code uses Kalshi's recommended hosts (docs.kalshi.com/getting_started/api_environments), set in
`venues/kalshi_trade.ENVS` and `stream.py`:

| Traffic | Prod host | Demo host |
|---|---|---|
| Signed REST (orders, account) | `external-api.kalshi.com` | `external-api.demo.kalshi.co` |
| Market-data REST (unsigned lists, books) | `api.elections.kalshi.com` (CloudFront, cached) | `demo-api.kalshi.co` |
| WebSocket market data, `order_group_updates` | `external-api-ws.kalshi.com` | `external-api-ws.demo.kalshi.co` |
| WebSocket `fill`, `user_orders` (rejects other channels) | `warp-ws.kalshi.com` | `warp-ws.demo.kalshi.co` |

- `external-api*` and `warp-ws` resolve to EC2 hosts in us-east-2: TCP RTT ~1 ms from here.
- The shared host `api.elections.kalshi.com` is CloudFront and its edge varies; the recorder had
  been routed via Ashburn (iad12, ~23 ms RTT), which put ~12 ms on every frame.
- Market-data REST stays on the shared host on purpose: the direct host is uncached and returned
  429 to the recorder's startup market listing; that traffic is not latency-sensitive. Research
  `CachedClient` code also keeps the old host because the cache key includes the URL.

| Prod, keep-alive p50 | Shared (CloudFront) host | Direct host |
|---|---|---|
| REST public GET | 7–18 ms | 4.3 ms |
| REST signed GET | 15–37 ms | 7.4 ms |
| First request on a new connection | 38–69 ms | 10 ms |
| WS ping RTT | 2.7 ms (25 ms via iad) | 1.3 ms (warp-ws 1.3 ms) |
| WS exchange send → our receive | 1.8–12 ms | 1.4 ms |

Kalshi's own event → send time is ~5 ms p50. Demo runs in us-west-2: ~48–52 ms from here
whatever the host (demo create/cancel RTT p50 ~60–73 ms); it proves plumbing only.
Prod order latency (one $1 post-only order placed and cancelled by hand, 2026-11-07):

| | Create | Cancel |
|---|---|---|
| Send → engine `ts_ms` | ~5 ms | ~4 ms |
| REST round trip | 6.4 ms | 5.6 ms |
| Send → `warp-ws` `user_order` | 5.8 ms | 5.4 ms |

`warp-ws` delivers the order update ~1 ms after the engine stamp, before the REST response.
So market event → our order at the engine is ~7 ms (feed) + <1 ms (decide, sign) + ~5 ms ≈
13 ms p50. Simulator and paper default `latency_ms` is 20 (was 100). That's decision → engine,
because replay times are already our receive times. Repeated `GET /portfolio/balance` calls take
~1.6 s each after the first (other reads 5–8 ms), so keep balance reads out of any hot loop.

Live runs log per-order legs as `timing` ledger records (`albert/timing.py`): event → decide,
decide → send, REST round trip, send → engine `ts_ms`, send → `warp-ws` ack. Status lines carry
p50/p90/p99/max per leg and the asyncio loop lag.

Latency fixes (2026-11-07):

| Change | Measured effect |
|---|---|
| Live L2 bot: classifier holdback 500 → 0 ms | event → decide p50 582 → 0.22 ms (demo run) |
| Orders sent concurrently (4 in flight, pool re-warmed every 20 s) | a burst no longer waits N round trips |
| Explicit `exchange_index` from `GET /markets` (was -1, auto-route) | prod A/B, 8 pairs each: create RTT p50 6.64 → 6.16 ms, cancel 5.49 → 4.54 ms; send → engine: create 4.5 → 3.8 ms, cancel 3.5 → 2.1 ms |
| Ed25519 prod key (`KALSHI_API_KEY_FAST_ID` / `KALSHI_PRIVATE_KEY_FAST`, preferred by `signer_for`) | signing p50 0.35 → 0.03 ms |

See `.planning/research/exchange-infra-latency.md` §5.2 for where the remaining ~5 ms goes. `KalshiTrader` keeps idle connections for 300 s to avoid a new TLS handshake per
sparse order. Further options, not done: an Ed25519 prod key (signing 0.3 ms → 0.03 ms) and
AWS PrivateLink (Premier tier and above only).
