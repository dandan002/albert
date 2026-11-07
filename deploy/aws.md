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
  - `albert-record` (deploy/record-loop.sh --paired-days 3) → data/raw/stream
  - `albert-paper` (deploy/paper-loop.sh) → data/paper_l2
  - `albert-forward` (deploy/forward-loop.sh, LEADS="mentions fcs rain") → data/cache/forward
  - `albert-mentions@demo` (albert live mentions --env demo; args in deploy/mentions-demo.conf)
  - prod pilot later: `albert-mentions@prod` with deploy/mentions-prod.conf containing
    `MENTIONS_ARGS=--live ...` AND `KALSHI_LIVE_ENABLED=1` in .env (both required).
- Update code: from the Mac, `rsync -az --exclude data/ --exclude .venv/ --exclude .env ./ albert:albert/`
  then `sudo systemctl restart albert-...`.
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
p50/p90/p99/max per leg and the asyncio loop lag. Orders currently auto-route
(`exchange_index: -1`), which Kalshi says adds latency; see
`.planning/research/exchange-infra-latency.md` §5.2 for the breakdown and the free fixes. `KalshiTrader` keeps idle connections for 300 s to avoid a new TLS handshake per
sparse order. Further options, not done: an Ed25519 prod key (signing 0.3 ms → 0.03 ms) and
AWS PrivateLink (Premier tier and above only).
