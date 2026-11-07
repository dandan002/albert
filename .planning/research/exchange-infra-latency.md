# Exchange infrastructure, matching, APIs and latency: Kalshi and Polymarket US

Research date: 2026-11-04 (measurements taken 2026-11-05 02:54–03:00 UTC). Scope: what a small
maker bot (resting limit orders in many thin, non-financial markets, $100 split across both venues)
needs to know about matching, order types, APIs, rate limits, hosting and latency.

Notation: **[V]** = verified from an official doc, filing, or our own measurement (cited);
**[I]** = inference or estimate (reasoning given); **[U]** = unverified or secondary source only.

Current state of the repo (for context): `src/albert/venues/kalshi.py` and `polymarket_us.py` are
**public-REST, read-only** clients (no signing, no WebSocket). Config uses
`https://api.elections.kalshi.com/trade-api/v2` and `https://gateway.polymarket.us/v1`
(`src/albert/config.py`). The paper trader polls every 60 s (`deploy/gcp.md`). Everything below
about authenticated trading, WebSockets and order management is still to be built.

---

## 0. Executive summary

1. **Kalshi runs in AWS us-east-2 (Ohio).** Every origin hostname we resolved (REST ALB, WebSocket,
   the new low-latency `warp-ws` fill host, both FIX gateways) maps to AWS `us-east-2` EC2 ranges in
   AWS's published `ip-ranges.json` [V, §7]. `api.elections.kalshi.com` is a CloudFront front end.
   The **demo** environment runs in **us-west-2**, so demo latency tells you nothing about production [V].
2. **Polymarket US's exchange runs in AWS us-east-1 (N. Virginia).** The institutional host
   `api.prod.polymarketexchange.com` resolves to `us-east-1` EC2 IPs [V]. The retail API
   (`api.polymarket.us`, `gateway.polymarket.us`) sits behind Cloudflare, so its origin can't be seen.
   It most likely proxies to that same us-east-1 exchange [I].
3. **Both venues use price-time priority**, with a CLOB per market [V]. Both have post-only
   (`post_only` on Kalshi, `participateDontInitiate` on PM US), IOC/FOK, GTC, expiring orders,
   batch create/cancel, amend/modify, cancel-all and self-trade prevention [V].
4. **Latency is not our edge and co-location would not help.** Our main risk is adverse selection
   by takers who saw the real-world event first (stadium, official data feeds). They act about
   0.5–10 s before a broadcast viewer, and no cloud region can close that gap. What matters:
   (a) do not leave quotes resting through scheduled information releases or live play unless the
   spread pays for it; (b) use exchange-side protection (Kalshi **order groups**, short
   **expiration times**, `cancel_order_on_pause`, PM US **GTD** orders); (c) cancel within about
   100 ms of a *market-data* signal. Python asyncio does that easily: signing a request takes 28 µs
   with Ed25519 and parsing a book takes about 10 µs.
5. **Rate limits matter more than RTT for us.** Kalshi Basic is **100 write tokens/s**, which is
   10 order creates/s or 50 cancels/s, with no burst. **Advanced is free with one API call**:
   300 tokens/s with a 3 s burst bucket. PM US retail is **25 req/s per source IP**. Upgrade
   Kalshi to Advanced on day one.
6. **Kalshi sharding is an operational trap for a $50 account.** Collateral is checked per exchange
   shard. Sports on shard 3 (Tennis, Baseball, Basketball) need funds moved to that shard, and
   order groups don't work across shards [V].
7. **Recommended deployment:** one small, non-burstable VM in **AWS us-east-2 (Ohio)**. GCP
   us-east4 is an acceptable fallback, at roughly 10–15 ms to Kalshi. Use the Python 3.12 asyncio
   service as is. Budget ≤100 ms p50 / ≤300 ms p99 from "market signal on the WebSocket" to "cancel
   acknowledged". Full details in §9.

---

## 1. Kalshi: matching engine and order semantics

| Topic | Finding | Status / source |
|---|---|---|
| Matching | Rule 5.8: "Kalshi's trading algorithms execute all trades by matching orders according first by price and then time priority… orders… at the same price will be executed in order of time, from oldest to most recent." | [V] Kalshi Rulebook v1.16, CFTC filing Feb 18 2025: https://www.cftc.gov/sites/default/files/filings/orgrules/25/02/rules02172515652.pdf |
| Queue position visible | `GET /portfolio/orders/{id}/queue_position` and `/queue_positions` return `queue_position_fp` (contracts ahead of you, 0-indexed), "determined using price-time priority". | [V] https://docs.kalshi.com/api-reference/orders/get-order-queue-position.md ; changelog https://docs.kalshi.com/changelog/index.md |
| Multiple matching engines (shards) | "Kalshi splits trading across multiple matching engines." Shard 0 is the default; 1 = Exotics/Combos; 2 = Crypto, Commodities; 3 = Sports with tags Tennis, Baseball, Basketball. "Collateralization checks run within the matching engine… fund the relevant exchange shard before placing an order." Opt-in auto-rebalancing runs every 10 s. "Order groups do not function across exchange instances." "Automatic routing will incur an additional latency cost." | [V] https://docs.kalshi.com/getting_started/exchange_sharding.md |
| Order endpoint (V2) | `POST /portfolio/events/orders`. Required: `ticker, side (bid/ask, YES-leg only), count, price, time_in_force, self_trade_prevention_type`. Optional: `client_order_id, expiration_time, post_only, cancel_order_on_pause, reduce_only, subaccount, order_group_id, exchange_index`. The response includes `ts_ms`, the matching-engine timestamp. Legacy `/portfolio/orders` is deprecated "no earlier than May 6, 2026". | [V] https://docs.kalshi.com/api-reference/orders/create-order-v2.md |
| Time in force | `good_till_canceled`, `immediate_or_cancel`, `fill_or_kill`. An expiring order is GTC plus `expiration_time` (Unix seconds). IOC can't be combined with `expiration_time`. | [V] same |
| Post-only | `post_only: bool` | [V] same |
| Reduce-only | Allowed only with IOC. | [V] same |
| `buy_max_cost` | Not in the V2 schema. It belonged to the legacy endpoint. | [V] absent from create-order-v2 and llms-full.txt |
| Self-trade prevention | Required field: `taker_at_cross` (cancel the incoming order, keep partial fills) or `maker` (cancel the resting order and keep matching). | [V] same |
| YES/NO book | The V2 API quotes everything from the YES leg: "`bid` means buy YES, `ask` means sell YES (selling YES is economically equivalent to buying NO at 1 − price)". The REST orderbook shows only bids for each side: a NO bid at q is a YES ask at 1 − q (already handled in `kalshi.orderbook_top`). | [V] create-order-v2; https://docs.kalshi.com/getting_started/orderbook_responses.md |
| Amend | `POST /portfolio/events/orders/{id}/amend`: change price, max fillable count, expiry. "Amending only expiry or decreasing size preserves queue position. Increasing size or changing price forfeits queue position." | [V] https://docs.kalshi.com/api-reference/orders/amend-order-v2.md |
| Decrease | `POST …/{id}/decrease` with `reduce_by` or `reduce_to`. | [V] https://docs.kalshi.com/api-reference/orders/decrease-order-v2.md |
| Batch | `POST` and `DELETE /portfolio/events/orders/batched`. "Maximum batch size scales with your tier's write budget." Each item is billed separately (10 tokens per create, 2 per cancel), and the whole batch must fit in the bucket. | [V] batch-create / batch-cancel pages; rate limits page |
| Cancel all | `DELETE /portfolio/events/orders` cancels all resting orders across every shard. Costs 2 tokens. "Newly placed orders may also be cancelled during the minute after the request." | [V] https://docs.kalshi.com/api-reference/orders/cancel-all-orders.md |
| Order groups | Create a group with a contracts limit (1–1,000,000). If fills exceed the limit within a rolling 15 s window, the exchange cancels all resting orders in the group and rejects new ones until reset. Manual `trigger` acts as a kill switch for the group. Max 25,000 groups per user. | [V] https://docs.kalshi.com/getting_started/order_groups.md ; changelog Jul 23 2026 |
| Tick size | Per market, from `price_ranges` (source of truth) and `price_level_structure`. Structures include `linear_cent` (1¢), `deci_cent` (0.1¢), tapered variants (0.1¢ below 10¢ and above 90¢), and `center_deci_edge_centi_cent` (combos). Whole-cent prices are always valid. Off-grid prices are rejected. | [V] https://docs.kalshi.com/getting_started/fixed_point_migration.md |
| Tick size, observed | Live sample (2026-11-05): of 1,000 open non-combo markets, **998 were `linear_cent` and 2 were `tapered_deci_cent`**. The first 200 markets including combos were all `center_deci_edge_centi_cent`. | [V] measured, §7.4 |
| Fractional contracts | Counts are fixed-point with 0.01 granularity. Since Apr 17 2026, markets "support fractional trading unconditionally". | [V] fixed_point_migration; changelog Apr 17 2026 |
| Collateral | Fully collateralized ("due to the fully collateralized… nature of" contracts, Rulebook ch. 5). Resting orders reserve collateral on their shard. | [V] Rulebook (above); exchange_sharding |
| Position limits | Rule 5.17: position limits may be set per contract in its Terms & Conditions. Otherwise Rule 5.16 position accountability levels apply. Not binding at our size. | [V] Rulebook v1.16 |
| Max open orders | **No documented cap found** in docs/llms-full.txt. | [U] |
| Hours / maintenance | Scheduled maintenance every **Thursday 3:00–5:00 AM ET**. A trading pause is in effect (cancels allowed, place/amend not). An exchange pause (rare, or an outage) blocks cancels too. Resting orders stay on the book unless `cancel_order_on_pause` was set. | [V] https://docs.kalshi.com/getting_started/maintenance_and_pauses.md |
| Taker order delay | Secondary reports say Kalshi filed with regulators for order delays (courtsiding concerns). Not found in the docs. | [U] https://predictiondesk.pages.dev/research-hub/newsletters/kalshi-takes-its-first-l |

## 2. Kalshi: APIs, WebSockets, FIX, rate limits, demo

### 2.1 Hosts

| Surface | Production (recommended) | Also supported | Source |
|---|---|---|---|
| REST | `https://external-api.kalshi.com/trade-api/v2` | `https://api.elections.kalshi.com/trade-api/v2` | [V] https://docs.kalshi.com/getting_started/api_environments.md |
| WebSocket | `wss://external-api-ws.kalshi.com/trade-api/ws/v2` | `wss://api.elections.kalshi.com/trade-api/ws/v2` | same |
| Fill and order updates (lower latency, added Oct 29 2026) | `wss://warp-ws.kalshi.com/trade-api/ws/v2`. Serves **only** `fill` and `user_orders`. | – | same; changelog |
| FIX | `mm.fix.elections.kalshi.com` (order entry: 8228 NR, 8230 RT, 8229 DC, 8231 PT, 8232 RFQ), `marketdata.fix.elections.kalshi.com:8233` | – | [V] https://docs.kalshi.com/fix/connectivity.md |
| Private connectivity | AWS PrivateLink at **Premier tier or above**. VPC peering at Prime or above. | – | [V] api_environments; fix/connectivity |
| L1/L2 feed over fiber | Kalshi streams order books via the DoubleZero Edge network for its most active sports and crypto-perp markets (institutional product). | – | [U, press] https://theblock.co/news/markets/2026-08-12-kalshi-first-prediction-market-to-stream-full-order-books-via-doublezero-411589 |

`api.elections.kalshi.com` (CloudFront) **caches the `GET /markets` list for up to 15 s**
(`cache-control: public, max-age=15`, `x-cache: Hit from cloudfront`, `age: 13`). It does not cache
`/markets/{t}/orderbook` (always `Miss`). `external-api.kalshi.com` also sends `max-age=15` on
`/markets`. **Never take quotes from the markets list for trading decisions.** Use the orderbook
endpoint or the WebSocket [V, measured §7.3].

### 2.2 WebSocket channels and book maintenance [V]
Channels (enum in https://docs.kalshi.com/websockets/websocket-connection.md): `orderbook_delta`,
`ticker`, `trade`, `fill`, `market_positions`, `market_lifecycle_v2`,
`multivariate_market_lifecycle`, `communications`, `order_group_updates`, `user_orders`, plus
crypto index feeds. "Authentication is required to establish the connection", even for
public channels.

Orderbook (https://docs.kalshi.com/websockets/orderbook-updates.md):
- The channel sends `orderbook_snapshot` first, then `orderbook_delta`. Each delta carries a
  per-subscription `seq` and a per-market exchange `clock`.
- **Gap rule:** "Apply every delta in `seq` order… If you detect a gap, request `get_snapshot` for
  all subscribed markets, or `unsubscribe`… then `subscribe` again." A repeated `subscribe` alone
  does not resync.
- `clock` matches updates across redundant connections, but it can decrease across maintenance.
- Conflation is opt-in: `update_interval_ms` of 0, 100 or 1000. Your own order updates bypass
  conflation. **Use 0 (real-time) for markets we quote in.** 100 ms conflation is fine for
  "watch-only" markets.
- One `orderbook_delta` subscription per connection. Error code 26 means the per-subscription
  market limit was exceeded; the number is not published [U]. Error 25 means the outbound buffer
  overflowed (slow consumer).
- Keep-alive: Kalshi sends a Ping every **10 s** and expects a Pong. Clients may also send pings.
  https://docs.kalshi.com/websockets/connection-keep-alive.md
- Concurrent WebSocket connections: 200 (Basic/Advanced/Expert), up to 1,000 on Prime.
  [V] rate_limits page.
- Compression: `permessage-deflate` is supported. Max message size is 64 MiB. [V] llms-full changelog

### 2.3 Rate limits [V] (https://docs.kalshi.com/getting_started/rate_limits.md)
Token buckets. Default cost is **10 tokens per request**. A cancel costs **2**, and so does
cancel-all. Read and Write buckets are separate. REST and FIX share them.

| Tier | Read tokens/s | Write tokens/s | How to qualify |
|---|---|---|---|
| Basic | 200 | 100 (bucket = 1 s, **no burst**) | sign up |
| Advanced | 300 (bucket 3 s) | 300 (bucket 3 s → burst 900) | **call the Upgrade Account API Usage Level endpoint** |
| Expert | 600 | 600 | 0.075% volume share to earn / 0.05% to keep |
| Premier | 1,200 | 1,200 | 0.125% / 0.10% |
| Paragon | 2,400 | 2,400 | 0.25% / 0.20% |
| Prime | 4,800 | 4,800 | 0.50% / 0.40% |
| Prestige | 12,000 | 9,600 | 1.00% / 0.80% |

- Volume share = your trailing-30-day volume ÷ (previous month's exchange volume × 2). Tiers above
  Advanced are unreachable for us.
- What these limits mean for us: Basic = 10 creates/s **or** 50 cancels/s. Advanced = 30 creates/s
  sustained and up to 450 cancels in a burst.
- `GET /portfolio/balance` has its own bucket (1/s on Basic and Advanced).
- A 429 carries no `Retry-After`. There is no penalty; back off exponentially.
- Per-shard write buckets: an auto-routed single order is billed to the unscoped bucket **and**
  every shard's bucket. Batch writes go only to the unscoped bucket.
- I tripped a 429 on `/markets` with an unauthenticated burst of about 16 requests in a few
  seconds from one IP (see §7.3). Public-endpoint IP limits exist but are not published [V measured / U limit].

### 2.4 FIX
- Protocol FIXT.1.1 / FIX50SP2 over TLS 1.2+. One FIX connection per API key. Uses the same key
  pair as REST (Ed25519 or RSA). [V] https://docs.kalshi.com/fix/authentication.md
- Only the RT (retransmission), Drop Copy, Post-Trade and replay sessions say "contact
  institutional@kalshi.com". The non-retransmission order-entry session (KalshiNR, 8228) is listed
  with no access gate, so it appears usable with an ordinary API key [I from
  https://docs.kalshi.com/fix/connectivity.md].
- **`CancelOrdersOnDisconnect` (tag 8013) exists only on FIX.** It cancels on any disconnect,
  including a graceful logout. [V] fix/authentication
- Mass cancel (35=q, 530=6) is limited to 1/s. [V] fix/connectivity
- SendingTime must be within **30 s** of server time. [V] llms-full (FIX session rejects)
- FIX gives no rate-limit advantage: the same token model applies. Its value for us would be
  cancel-on-disconnect. **Not worth building at $50** [I].

### 2.5 Auth and keys [V] (https://docs.kalshi.com/getting_started/api_keys.md)
- Ed25519 (recommended, now the web default) or RSA-2048 PSS.
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP` (ms), `KALSHI-ACCESS-SIGNATURE`.
- Signature input = `timestamp + METHOD + path` without the query string. The same path is signed
  on every host.
- The REST timestamp tolerance is not published. FIX allows 30 s [U for REST].
- Subaccount-restricted API keys exist. Use one for the bot so it can't touch the main account
  [V, changelog].
- AGENTS.md says "RSA-PSS". **Switch to Ed25519.** Measured cost: 28 µs vs 566 µs per signature (§7.5).

### 2.6 Demo environment
- `https://external-api.demo.kalshi.co/trade-api/v2` and `wss://external-api-ws.demo.kalshi.co/...`.
  Credentials are separate from production. "The price and behavior of markets in the demo
  environment may not be reflective of those in real markets." [V] https://docs.kalshi.com/getting_started/demo_env.md
- **The demo REST NLB is in AWS us-west-2**
  (`demo-external-api-nlb-….elb.us-west-2.amazonaws.com`; IPs in us-west-2 ranges) [V measured].
  Demo is good for testing signing, the order lifecycle, WebSocket parsing and gap handling. It is
  **useless for latency or fill-rate calibration**.

## 3. Polymarket US: matching engine, order types, APIs

Polymarket US has two API surfaces:

| | Retail API (what we can use as an individual) | Institutional "Polymarket Exchange" API |
|---|---|---|
| Who | Any KYC'd app user. Keys from polymarket.us/developer | Entities only, via the institutional portal (participant agreement, beneficial owners, etc.) |
| REST | `https://api.polymarket.us` (auth), `https://gateway.polymarket.us` (public) | `https://api.prod.polymarketexchange.com`, OAuth via Auth0 (private-key JWT) |
| Streams | `wss://api.polymarket.us/v1/ws/private`, `/v1/ws/markets` (JSON) | gRPC streams; FIX only via AWS PrivateLink |
| Auth | Ed25519: `X-PM-Access-Key`, `X-PM-Timestamp` (ms, **within 30 s**), `X-PM-Signature` over `timestamp+METHOD+path` | Bearer token, 24 h lifetime |
| Test env | **None documented** | Preprod `api.preprod.polymarketexchange.com` |

Sources: [V] https://docs.polymarket.us/api-reference/introduction.md ,
https://docs.polymarket.us/api-reference/authentication.md ,
https://docs.polymarket.us/trader-guide/environments.md ,
https://docs.polymarket.us/trader-guide/onboarding.md ("Individual traders: You do not need to complete this onboarding process").

**Conclusion: we are on the Retail API, and there is no sandbox.** Testing is paper trading
against real data, then tiny live orders [I].

| Topic | Finding | Source |
|---|---|---|
| Matching | "The Exchange operates as a continuous central limit order book… according to price-time priority… Within a price level, older orders have priority over newer orders. Should an existing order increase its quantity… it… loses time priority, but quantity decreases retain time priority." | [V] https://docs.polymarket.us/institutional/fix-api/fix-order-entry-overview.md |
| Single book | "Only the long side (YES) is directly tradable. The short side (NO) is synthetic." `price.value` is always the YES price. Buying NO at X means `price.value = 1 − X`. Buying YES at 0.60 and NO at 0.40 at once self-matches. | [V] https://docs.polymarket.us/api-reference/orders/overview.md |
| Order types | `ORDER_TYPE_LIMIT`, `ORDER_TYPE_MARKET` (with optional `slippageTolerance`). Intents: `BUY_LONG / SELL_LONG / BUY_SHORT / SELL_SHORT`, or `outcomeSide`+`action`. | [V] same |
| TIF | DAY, GTC, GTD (`goodTillTime`), IOC, FOK. Warning: **"DAY orders do not automatically cancel at 5pm"**, so use GTD. | [V] same |
| Post-only | `participateDontInitiate: true` means the order is "rejected if it would immediately match". | [V] https://docs.polymarket.us/api-reference/orders/create-order.md |
| Manual flag | Field `MANUAL_ORDER_INDICATOR_AUTOMATIC` must be set for bots. | [V] orders overview |
| Price / qty grid | Price must be between 0.01 and 0.99. Out-of-bounds orders still get an orderId but are rejected. Each market sets its own `orderPriceMinTickSize` (e.g. 0.005) and `minimumTradeQty` (e.g. 0.01). Off-grid input may be normalized. | [V] orders overview |
| Modify | `POST /v1/order/{id}/modify` is a cancel-replace. `/v1/orders/batched/modify` handles up to 20. | [V] orders overview |
| Batch | `/v1/orders/batched` (create ≤20) and `/v1/orders/batched/cancel` (≤20). **Batch responses do not confirm per-entry success.** Unknown IDs are silently ignored, so confirm outcomes on the private WebSocket. | [V] orders overview |
| Cancel all | `POST /v1/orders/open/cancel`, optionally filtered by market slugs. | [V] https://docs.polymarket.us/api-reference/orders/cancel-all-open-orders.md |
| Self-match prevention | Institutional/FIX: per-order SMP ID with cancel-newest (default) or cancel-oldest. Retail: the docs only mention a "self-match error". | [V] https://docs.polymarket.us/institutional/fix-api/fix-self-match-prevention.md ; [U] retail behaviour |
| Mass-quote protection | Exchange-side: if an account's traded quantity within a rolling interval reaches a threshold, its eligible resting orders are cancelled (like Kalshi order groups). Configured per account. Unclear whether retail accounts can set it. | [V] https://docs.polymarket.us/institutional/fix-api/fix-mass-quote-protection.md ; [U] retail |
| Latency stopgap | Orders not processed within **5 s** are rejected with message "Global Rate Limit Exceeded". This is *not* a rate limit. "Pure cancels are not affected." You can cancel before receiving an ack. | [V] https://docs.polymarket.us/api-reference/rate-limits.md |
| Hours | Nearly 24/7. Weekly maintenance **Thursday 6:00–8:00 AM ET**. Emergency pauses happen without notice. The trading day ends at 5 PM ET. | [V] https://docs.polymarket.us/learn/trading/access-and-limits/trading-hours.md |
| Collateral | See https://docs.polymarket.us/market-structure/collateral-and-margin.md (not analyzed here). | – |

### 3.1 Rate limits [V]
- **Retail API: 25 requests/s general limit**, "enforced per source IP and Cloudflare location, so
  requests using different API keys from the same IP share counters". Combos and RFQ creation are
  1/s. 429 → wait at least 1 s, then back off. More capacity: email support@polymarket.us.
  https://docs.polymarket.us/api-reference/rate-limits.md
- Institutional (for reference): InsertOrder 30–400/s and CancelOrder 90–1,200/s by tier. Batch
  calls are 50/s with up to 20 operations each. Public endpoints allow 20/s per IP.
  https://docs.polymarket.us/trader-guide/rate-limits.md
- Implication: 25 req/s is plenty for one bot. Prefer batch cancel (20 IDs per request) or
  cancel-all.

### 3.2 WebSockets (retail) [V]
- `/v1/ws/markets` subscription types: `MARKET_DATA` (full book plus stats per message),
  `MARKET_DATA_LITE` (BBO plus depth counts), `TRADE`. `/v1/ws/private`: `ORDER` (snapshot then
  updates), `POSITION`, `ACCOUNT_BALANCE`.
  https://docs.polymarket.us/api-reference/websocket/overview.md , …/markets.md , …/private.md
- Both endpoints require API-key headers in the handshake.
- The server sends a JSON `{"heartbeat":{}}`. The docs say "Reconnect if heartbeats stop" but give
  no interval [U].
- **No sequence numbers are documented on the retail streams.** The docs say only "Messages are
  delivered in sequence". `MARKET_DATA` messages carry whole books, so a reconnect simply takes the
  next full message as the snapshot [V doc / I handling].
- Institutional gRPC: market data snapshot then updates. Up to 1,000 symbols per stream and 20
  streams per firm. Drop Copy resumes from a `resume_token`. https://docs.polymarket.us/streaming-endpoints/streaming-best-practices.md

### 3.3 Fees and maker economics [V] (https://docs.polymarket.us/fees.md, effective 2026-10-28)
- Taker fee = Θ·C·p·(1−p) with Θ = 0.0695 (0.10 for table tennis). **Makers pay nothing.**
- **Maker rebates start only at $10M of maker volume** (Tier 2, Θ = −0.0035). Tier 1 (all of us)
  gets **no rebate**.
- From **Nov 6 2026, 1 PM ET**, tiers use a rolling 30-day window, set daily.
- **Liquidity Incentive Program:** resting orders near the best price are scored every second
  (Score = DiscountFactor^ticks × size). Each market has a pool, a Target Size and a Max Spread.
  This could matter more to a small maker than rebates do. Live schedule:
  https://polymarket.us/rewards . Program rules:
  https://docs.polymarket.us/incentives/liquidity.md
- AGENTS.md says "Polymarket US… pays maker rebates by tier". That is true, but it is irrelevant
  below $10M per 30 days.

### 3.4 Cancel-on-disconnect (PM US) [V]
Exists only on **FIX**, and even there it "cancels open (unexecuted) **DAY** orders… GTC and GTD
orders continue to rest" (https://docs.polymarket.us/institutional/fix-api/fix-session-management.md).
**There is nothing for the retail API.** Our dead-man's switch has to be short **GTD** expiries
that we keep refreshing.

## 4. Operational checklist (both venues)

| Concern | Kalshi | Polymarket US (retail) |
|---|---|---|
| Keys | Ed25519 key, PEM stored in `.env` or Secret Manager. Use a **subaccount-restricted key** for the bot. | Ed25519 secret, base64; first 32 bytes are the seed. Shown once. Revocable at polymarket.us/developer. |
| Clock | REST tolerance unpublished; FIX 30 s. Run chrony against the cloud's time service (Amazon Time Sync 169.254.169.123 / Google metadata NTP) and alert if offset > 1 s. | 30 s tolerance [V]. Same chrony setup. |
| Book gaps | `seq` per subscription. On a gap, `get_snapshot` for affected markets. On reconnect, a fresh snapshot. Discard state across Thursday maintenance (clock may go backwards). | Full-book messages; on reconnect, wait for the first message. Every N minutes, cross-check against REST `/v1/markets/{slug}/book` (public gateway, 20/s per IP). |
| Liveness | Server pings every 10 s. If no ping or data for 15–20 s, cancel all via REST, then reconnect. | Heartbeat JSON (interval undocumented). Same policy with a 30 s timeout [I]. |
| Cancel-on-disconnect | FIX only (8013). REST: none. Substitute: `expiration_time` = now + 60–300 s and re-amend expiry. Amending expiry keeps queue priority [V]. | None on retail. Substitute: GTD `goodTillTime` = now + 60–300 s. Modify is a cancel-replace and **loses priority**, so plan refresh cadence around that. |
| Fill burst protection | **Order group** with a `contracts_limit` per 15 s per shard. The exchange auto-cancels with zero latency on our side. | MQP exists (account-level, configured by the exchange). Retail availability unknown. |
| Kill switch | `DELETE /portfolio/events/orders` (all shards, 2 tokens); `order_groups/{id}/trigger`. | `POST /v1/orders/open/cancel`. |
| Pauses | `cancel_order_on_pause: true` on every order. Maintenance Thu 3–5 AM ET. | Maintenance Thu 6–8 AM ET. Pull quotes at 5:55. |
| Order acks | Use `warp-ws` `user_orders` and `fill` (lowest latency). REST response `ts_ms` lets us measure engine time vs local time. | Private WS `ORDER`. Batch REST responses are not proof of outcome. |
| Shards | Track `exchange_index` per market. Fund shard 3 before sports trading, or use auto-rebalancing. Pass `exchange_index` explicitly to avoid the auto-routing latency cost and multi-bucket billing. | n/a |

## 5. Latency requirements analysis

### 5.1 Who picks off a resting maker in a thin non-financial market?
1. **Event-informed takers.** Someone in the stadium, someone on an official league data feed,
   someone reading a press release or NWS report the moment it posts. Sportradar's claim:
   official data reaches books "without delay" while a TV feed runs about 7 s behind [U/vendor
   claim, via https://cdcgaming.com/commentary/potential-sportradar-listing-raises-official-data-stakes/].
   The UK Gambling Commission defines courtsiding as exploiting exactly that delay
   (https://www.gamblingcommission.gov.uk/licensees-and-businesses/print/in-play-or-in-running-betting).
   Sub-500 ms official feeds exist [U] (https://sbcnews.co.uk/uncategorized/2026/05/20/cost-stream-delay-live-betting).
   **Gap to us: seconds.** No VM placement closes a seconds-scale information gap.
2. **Cross-market takers** reacting to a move on another venue or a correlated Kalshi market.
   Their gap to us is tens of milliseconds to a few seconds. **This is where our speed matters,
   and an asyncio bot in the right region is competitive with anyone not co-located.**
3. **Latency-arb HFT** on queue priority in liquid markets. Not our game. Kalshi offers
   PrivateLink from Premier tier and a DoubleZero fiber feed for institutions. Co-location-grade
   speed is only worth it in liquid markets where queue position is the edge.

### 5.2 Concrete latency budget (target for the live bot)
Numbers marked [I] are estimates. The rest come from §7 measurements or docs.

| Leg | Same region (AWS us-east-2 → Kalshi) | Cross region (GCP us-east4 → Kalshi) | Home NYC → Kalshi (measured) |
|---|---|---|---|
| Network RTT to origin | ~1 ms [I] | ~10–15 ms [I: Ashburn–Columbus fiber about 500 km; AWS: "highly likely… exceed 10 ms" between us-east-1 and us-east-2, https://repost.aws/questions/QU04hC9l_ITLabW555J9iQ6Q] | **38–46 ms** TCP-connect to the ALB [V §7.2] |
| WS delta: exchange → our process | +0.5 RTT ≈ 1 ms | ≈ 6–8 ms | ≈ 20–25 ms |
| Decide (parse + strategy) | < 1 ms (JSON parse ≈ 10 µs) [V §7.5] | same | same |
| Sign cancel | 0.03 ms Ed25519 / 0.6 ms RSA [V §7.5] | same | same |
| Cancel request → engine ack (warm keep-alive) | RTT + server ≈ 5–20 ms [I] | ≈ 20–40 ms [I] | **≈ 47 ms p50** on a warm uncached GET (`/exchange/status` via ALB) [V §7.3]. Order writes unmeasured. |
| **Signal → cancel acked** | **≈ 10–30 ms** | **≈ 30–60 ms** | ≈ 80–120 ms |

**Measured 2026-11-07 (AWS us-east-2, direct hosts):** exchange event → our receive p50 7.0 ms /
p99 12.8 ms (2.8M frames: 6 ms Kalshi-side, 1.4 ms network). Prod create and cancel each took
~4-5 ms from our send to the engine `ts_ms`, with REST RTT 5.6-6.4 ms. So signal → cancel acked
≈ 13 ms, the low end of the same-region estimate. Details: `deploy/aws.md`.

**Budget to enforce:** p50 ≤ 100 ms, p99 ≤ 300 ms from receiving a market-data trigger to the
cancel being acknowledged (measure with Kalshi `ts_ms` and our monotonic clock). Every venue/region
option above passes, so **region choice is a tie-breaker, not a requirement**. Anything slower than
about 1 s (for example, the current 60 s polling paper trader) is **not acceptable for resting live
quotes** in markets with in-play information.

**Policy matters more than speed (for event-driven markets):**
- *Scheduled releases* (NWS climate reports, poll closings, award announcements, economic-style
  release times): cancel or widen **before** the timestamp, T−5 s, using the exchange clock. This
  removes the race entirely.
- *Live sports:* don't rest quotes during live play unless the expected spread pays for event risk.
  Prefer pre-game quoting and pull at scheduled start (Kalshi lifecycle events help). Order groups
  cap the damage of one burst of fills.
- *Slow-information markets* (long-dated politics, culture): latency is nearly irrelevant. Stale
  quotes matter only at news events, which you can't out-race anyway.

### 5.3 Is Python asyncio enough? Yes [I, with measured inputs]
- CPU per message is microseconds (§7.5). A single process can hold several WebSocket connections
  across hundreds of markets.
- Our binding constraints are exchange rate limits (10 creates/s on Kalshi Basic) and the
  2–4 decimal tick grid, not interpreter speed.
- Practical rules: one asyncio event loop; `websockets` or `aiohttp` with `permessage-deflate`;
  `orjson` if parsing ever shows up in profiles; never block the loop with Parquet writes (use a
  thread or queue); `httpx.AsyncClient` with HTTP keep-alive and connection pre-warming (a cold
  TLS handshake added about 90 ms from NYC, §7.2); one process per venue to isolate failures.
- Rewriting in Rust/Go/C++ would only pay off with FIX/PrivateLink and a liquid-market strategy.
  Neither applies at $100.

### 5.4 When co-location would matter
Only if all of these hold: (a) quoting liquid markets where queue position decides fills, (b)
Premier tier or above (needed for PrivateLink, about 0.125% of exchange volume), and (c) a
cross-venue strategy where tens of ms decide who gets the fill. None applies now [I].

## 6. Hosting findings (who runs where)

| Host | Resolves to | Provider / region | Status |
|---|---|---|---|
| `api.elections.kalshi.com` | 13.32.x / 13.33.x / 13.249.x | AWS **CloudFront** (global). POP JFK50/JFK52 from NYC. | [V] |
| `external-api.kalshi.com` → `external-api-alb.elections.kalshi.com` | 3.133.2.254, 3.150.151.204, 16.58.229.24, 18.119.184.78, 77.112.155.46… | AWS **us-east-2** EC2 (ALB) | [V] |
| `external-api-ws.kalshi.com`, `warp-ws.kalshi.com` | same ALB, us-east-2 IPs | AWS **us-east-2** | [V] |
| `trading-api.kalshi.com` (legacy) | 3.17.202.178, 3.146.177.187 (`server: awselb/2.0`) | AWS **us-east-2** | [V] |
| `mm.fix.elections.kalshi.com`, `marketdata.fix…` | 18.223.244.34, 18.119.210.162, 18.220.63.230 | AWS **us-east-2** | [V] |
| `demo-api.kalshi.co` | CloudFront | – | [V] |
| `external-api.demo.kalshi.co` | `demo-external-api-nlb-…elb.us-west-2.amazonaws.com` | AWS **us-west-2** | [V] |
| `api.polymarket.us`, `gateway.polymarket.us`, `ws.polymarket.us` | 104.18.38.40, 172.64.149.216 | **Cloudflare** (`cf-ray …-BOS` from NYC). Origin hidden. | [V] |
| `api.prod.polymarketexchange.com` | 184.198.18.247, 100.25.127.120, 54.205.113.173, 3.231.185.247, 34.232.142.231, 32.193.87.174 | AWS **us-east-1** EC2 | [V] |
| `api.preprod.polymarketexchange.com` | 50.16.199.198, 3.82.180.221, 100.51.23.42 | AWS us-east-1 | [V] |

Third-party corroboration: Glassnode's prediction-market latency monitor states "Origin servers
are in AWS us-east-2 (Ohio)" for Kalshi, behind CloudFront
(https://latency.glassnode.com/prediction-markets/about). A forum post claims "Polymarket US:
us-east-1, Kalshi: us-east-2" [U] (https://reddit.sentinel-team.org/posts/1r68xm3/snapshots/2026-02-18T17%3A21%3A01.932339Z).
A bot-hosting blog that says Kalshi is in us-east-1 is **contradicted by our DNS evidence**
(https://www.botforkalshi.com/blog/kalshi-bot-hosting-guide).

Note: Polymarket *international* (clob.polymarket.com, data-only for us) is reported to have its
origin in AWS eu-west-2 London, behind Cloudflare (Glassnode, same page). This does not matter for
our data pulls.

---

## 7. Raw measurements (from the dev laptop: New York City, ISP AS6079 RCN, Apple M1 Pro)

### 7.1 DNS → IP → AWS region (AWS `ip-ranges.json`, fetched 2026-11-05)
```
api.elections.kalshi.com.  A 13.33.67.104 / .16 / .65 / .53   -> 13.32.0.0/15 GLOBAL CLOUDFRONT
demo-api.kalshi.co.        A 13.32.241.76 / .56 / .74 / .115  -> 13.32.0.0/15 GLOBAL CLOUDFRONT
trading-api.kalshi.com.    A 3.146.177.187, 3.17.202.178      -> us-east-2 EC2
external-api.kalshi.com.   CNAME external-api-alb.elections.kalshi.com.
  3.133.2.254   -> 3.132.0.0/14 us-east-2 EC2
  3.150.151.204 -> 3.144.0.0/13 us-east-2 EC2
  3.18.236.102  -> 3.16.0.0/14  us-east-2 EC2
  3.147.23.30   -> 3.144.0.0/13 us-east-2 EC2
  16.58.229.24  -> 16.58.0.0/16 us-east-2 EC2
  18.119.184.78 -> 18.116.0.0/14 us-east-2 EC2
  18.189.201.245-> 18.189.0.0/16 us-east-2 EC2
  16.58.86.116  -> 16.58.0.0/16 us-east-2 EC2
external-api-ws.kalshi.com: external-api-alb.elections.kalshi.com. 77.112.190.83 13.184.92.40 ...
  77.112.190.83 -> 77.112.0.0/14 us-east-2 EC2 ; 13.184.92.40 -> 13.184.0.0/13 us-east-2 EC2
warp-ws.kalshi.com: external-api-alb.elections.kalshi.com. 3.130.77.23 ... 77.112.134.125 -> us-east-2
mm.fix.elections.kalshi.com: 18.223.244.34 18.119.210.162 -> 18.220.0.0/14, 18.116.0.0/14 us-east-2
marketdata.fix.elections.kalshi.com: 18.220.63.230 -> us-east-2
external-api.demo.kalshi.co: demo-external-api-nlb-b0adf6094204c0bd.elb.us-west-2.amazonaws.com.
  44.226.28.91 -> 44.224.0.0/11 us-west-2 ; 54.71.122.8 -> 54.68.0.0/14 us-west-2
gateway.polymarket.us / api.polymarket.us / ws.polymarket.us: 104.18.38.40, 172.64.149.216
  whois 104.18.38.40: NetName CLOUDFLARENET, OrgName Cloudflare, Inc.
api.prod.polymarketexchange.com:
  184.198.18.247 -> 184.192.0.0/12 us-east-1 ; 100.25.127.120 -> 100.24.0.0/13 us-east-1
  54.205.113.173 -> 54.204.0.0/15 us-east-1 ; 3.231.185.247 -> 3.224.0.0/12 us-east-1
  34.232.142.231 -> 34.224.0.0/12 us-east-1 ; 32.193.87.174 -> 32.192.0.0/13 us-east-1
api.preprod.polymarketexchange.com: 50.16.199.198 -> 50.16.0.0/15 us-east-1
```

### 7.2 curl cold-connection timing (seconds; new TCP+TLS per request)
```
== https://api.elections.kalshi.com/trade-api/v2/exchange/status
x-cache: Hit from cloudfront ; x-amz-cf-pop: JFK52-P9
dns=0.007 tcp=0.028 tls=0.055 ttfb=0.083 code=200
dns=0.003 tcp=0.020 tls=0.043 ttfb=0.067 code=200
dns=0.003 tcp=0.017 tls=0.047 ttfb=0.071 code=200
dns=0.003 tcp=0.021 tls=0.050 ttfb=0.076 code=200
dns=0.003 tcp=0.014 tls=0.038 ttfb=0.061 code=200
== https://external-api.kalshi.com/trade-api/v2/exchange/status   (us-east-2 ALB, direct)
dns=0.004 tcp=0.043 tls=0.093 ttfb=0.139 code=200
dns=0.004 tcp=0.039 tls=0.081 ttfb=0.118 code=200
dns=0.005 tcp=0.042 tls=0.087 ttfb=0.131 code=200
dns=0.004 tcp=0.044 tls=0.090 ttfb=0.141 code=200
dns=0.004 tcp=0.040 tls=0.085 ttfb=0.128 code=200
== https://trading-api.kalshi.com/trade-api/v2/exchange/status    (server: awselb/2.0)
tcp=0.034 tls=0.076 ttfb=0.110 code=401 ; tcp=0.040 ... ttfb=0.133 ; tcp=0.046 ... ttfb=0.147
tcp=0.063 ... ttfb=0.170 ; tcp=0.045 ... ttfb=0.129
== https://demo-api.kalshi.co/trade-api/v2/exchange/status        (x-amz-cf-pop: JFK50-P13, Miss)
ttfb=0.138 (miss), then 0.073 0.065 0.073 0.069
== https://gateway.polymarket.us/v1/markets?limit=1               (server: cloudflare, cf-ray ...-BOS)
tcp=0.024 tls=0.059 ttfb=0.089 ; tcp=0.030 tls=0.063 ttfb=0.094 ; tcp=0.029 tls=0.061 ttfb=0.094
tcp=0.021 tls=0.050 ttfb=0.082 ; tcp=0.024 tls=0.060 ttfb=0.096
== https://api.polymarket.us/v1/markets?limit=1                   (cf-ray ...-BOS)
ttfb=0.098 0.095 0.091 0.088 0.085
== https://api.prod.polymarketexchange.com/v1/health               (us-east-1, {"status":"healthy"})
tcp=0.062 tls=0.092 ttfb=0.118 ; tcp=0.032 tls=0.070 ttfb=0.098 ; tcp=0.027 tls=0.056 ttfb=0.091
tcp=0.026 tls=0.059 ttfb=0.084 ; tcp=0.031 tls=0.064 ttfb=0.095
== https://external-api.kalshi.com/.../markets/KXNBAGAME-26NOV08MIASAC-SAC/orderbook
tcp=0.038 tls=0.090 ttfb=0.134 ; tcp=0.048 tls=0.095 ttfb=0.144 ; tcp=0.039 tls=0.079 ttfb=0.116
tcp=0.039 tls=0.076 ttfb=0.114 ; tcp=0.037 tls=0.082 ttfb=0.121
```
ICMP ping (5 packets): CloudFront edge 13.33.67.104 min/avg/max = 15.7/20.1/24.9 ms. Cloudflare
104.18.38.40 = 19.0/20.9/23.1 ms. Kalshi ALB 3.146.177.187: 100% loss (ICMP filtered). Traceroute
to the ALB showed only `*` after hop 13, so TCP-connect time is the RTT proxy. **Reading:** the
home last mile adds about 15 ms (the edge POP is in the same metro but pings 16–25 ms). The NYC
home → Ohio origin RTT is about 38–46 ms.

### 7.3 Warm keep-alive request latency (Python httpx 0.27.2, 15 requests per target, one connection)
```
kalshi api.elections (CloudFront) /markets?limit=1: n=15 code=429 min=48.3 p50=51.6 max=73.0 ms  x-cache=Error from cloudfront pop=JFK52-P9
kalshi api.elections /exchange/status: n=15 code=200 min=42.0 p50=48.8 max=55.5 ms  x-cache=Miss from cloudfront pop=JFK52-P9
kalshi external-api (ALB us-east-2) /exchange/status: n=15 code=200 min=40.8 p50=46.7 max=84.4 ms
pmus gateway (Cloudflare) /markets?limit=1: n=15 code=200 min=30.4 p50=69.2 max=85.0 ms  pop=...-BOS
```
(The 429 came after about 16 unauthenticated `/markets` calls within a few seconds from this IP.)

Caching headers:
```
api.elections /markets/{t}/orderbook : x-cache: Miss from cloudfront (3/3)
api.elections /markets?limit=1       : cache-control: public, max-age=15 ; x-cache: Hit from cloudfront ; age: 13
external-api  /markets?limit=1       : cache-control: public, max-age=15
```

### 7.4 Tick-structure sample (public `GET /markets`, 2026-11-05)
```
status=open, limit=200 (first page, combos)        : center_deci_edge_centi_cent x200
status=open, limit=1000, mve_filter=exclude         : linear_cent x998, tapered_deci_cent x2
  top series: KXNCAAFSPREAD 191, KXNCAAFTOTAL 166, KXNCAAMBTOTAL 121, KXNCAAMBSPREAD 90,
              KXESOCCERGAME 54, KXCS2GAME 40, KXNHLTOTAL 27, KXNCAAMB1H 24
```

### 7.5 CPU cost (Apple M1 Pro, cryptography 44.0.1, CPython 3 via anaconda)
```
ed25519: 28 us/sign
rsa-pss-2048-sha256: 566 us/sign
json.loads ~100-level book (2070 B): 10 us
```

---

## 8. Open questions / not verified
- Kalshi REST timestamp tolerance; max open orders per user; per-subscription WebSocket market
  limit (error 26); public-endpoint IP rate limits.
- Polymarket US retail: heartbeat interval; whether retail accounts get self-match prevention or
  mass-quote protection; whether the retail order path goes Cloudflare → us-east-1 exchange (likely).
- Real order-write ack latency on both venues. Measure it at go-live with Kalshi `ts_ms` and PM US
  `transactTime` against local monotonic time.
- Status of any Kalshi taker speed bump filing for sports.
- Cross-cloud RTTs (GCP us-east4 ↔ AWS us-east-1/us-east-2) are estimates. Measure them from the VM
  with `curl -w` before committing.

## 9. Recommended deployment and latency budget

**Provider/region:** **AWS us-east-2 (Ohio)**, one AZ.
- Why: it is in the same region as Kalshi's REST/WS ALB, `warp-ws` fill host and FIX gateways
  (≈1 ms). It is about 10–15 ms from Polymarket US's us-east-1 exchange [I]. It keeps the door open
  to Kalshi PrivateLink if we ever reach Premier.
- Kalshi is the venue with order groups, queue-position APIs and the broader non-financial
  catalogue, so it gets the shorter path.
- Fallback: keep the existing **GCP us-east4** plan (`deploy/gcp.md`). It costs about 10–15 ms
  more to Kalshi and is probably ~2–5 ms to PM US [I]. That difference does not change any decision
  in §5. Don't spend time migrating unless convenient.

**Instance:** a non-burstable 2-vCPU small instance, for example AWS `c7g.medium` / `m7g.medium`
(arm64) or `t3.small` with unlimited credits. On GCP, upgrade from `e2-small` (shared core) to
`e2-standard-2` or `c3-standard-4` only if CPU steal shows up. Add 20–30 GB of gp3 disk for the
ledger and raw capture, and chrony with Amazon Time Sync (169.254.169.123). Keep secrets in AWS
Secrets Manager (or GCP Secret Manager) loaded into the env at start.

**Runtime:** Python 3.12, a single asyncio process per venue. Two Kalshi WebSocket connections:
`external-api-ws` for `orderbook_delta` (real-time), `ticker`, `trade`, `market_lifecycle_v2`;
`warp-ws` for `fill` and `user_orders`. Two PM US WebSockets (markets, private). `httpx.AsyncClient`
with pre-warmed keep-alive to `external-api.kalshi.com` (not the CloudFront host) and
`api.polymarket.us`. Ed25519 keys on both venues. Upgrade Kalshi to the **Advanced** tier.

**Exchange-side safety:**
- Kalshi: every order `post_only`, `cancel_order_on_pause`, `self_trade_prevention_type=maker`,
  an `expiration_time` 1–5 min ahead (refresh by amending expiry, which keeps priority), and an
  `order_group_id` with a small `contracts_limit` per shard.
- PM US: `participateDontInitiate`, GTD 1–5 min, `MANUAL_ORDER_INDICATOR_AUTOMATIC`.
- Both: cancel-all on stale WebSocket (>15–20 s with no ping or data), on clock skew > 1 s, and
  before Thursday maintenance windows (Kalshi 03:00 ET, PM US 06:00 ET).

**Latency budget (SLOs to log and alert on):**

| Metric | Target p50 | Target p99 | Hard fail |
|---|---|---|---|
| WS message age (exchange ts → local receive) | < 10 ms | < 50 ms | > 1 s → cancel all, resync |
| Trigger → cancel sent | < 2 ms | < 10 ms | – |
| Cancel sent → ack (REST or `user_orders`) | < 30 ms (us-east-2) / < 60 ms (us-east4) | < 250 ms | > 2 s → cancel-all, pause quoting |
| Signal → cancel acknowledged, end to end | **≤ 100 ms** | **≤ 300 ms** | – |
| Pre-scheduled-release pull | T − 5 s | T − 2 s | quote still live at T → incident |
| Heartbeat silence | – | – | 15 s (Kalshi), 30 s (PM US) → cancel all |

Bottom line: a Python asyncio bot on a small us-east-2 VM is about 1–2 orders of magnitude faster
than it needs to be for this strategy class. The work that protects the bankroll is quote policy
around information events plus exchange-side auto-cancels, not shaving milliseconds.
