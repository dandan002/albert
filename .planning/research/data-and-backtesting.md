# Data sources and backtest / live-test infrastructure for maker strategies

Research date: 2026-11-04. Scope: Kalshi and Polymarket US (tradable), Polymarket international
(data only). Companion to `.planning/research/market-making.md`, which covers strategy theory,
incentive programs and fee detail. This file covers **data, simulators, and how to test**.

**Labels.** **[V]** = verified on the cited page on 2026-11-04. **[V-3p]** = stated by a third
party or vendor marketing page and not independently checked. **[I]** = my inference or design
recommendation. **[L]** = measured from this repo's local data.

---

## 0. Summary

1. **Nobody sells a long, complete Kalshi L2 history, and Kalshi does not publish one.** Kalshi's
   historical API covers markets, candlesticks (1 min / 1 h / 1 d), trades, and your own
   orders/fills/positions. It has no order book history [V]. Third-party Kalshi book archives all
   start in 2026, are captured by the vendor themselves, and several say coverage is partial
   (section 1.2). For Polymarket intl there is a free hourly L2 archive (pmxt) from 2026-04-13 [V-3p].
   For Polymarket US, Allium has a snapshot table; nothing else was found.
2. **Self-recording is cheap and required.** Kalshi's `orderbook_delta` WebSocket gives a snapshot,
   then deltas with a per-subscription `seq` (to detect gaps) and an exchange `clock` [V]. Polymarket
   US gives full-book messages over WebSocket, and over gRPC it offers an optional *unaggregated*
   ("raw") book [V]. Expected volume for a few hundred markets is tens to low hundreds of MB/day
   compressed [I, measure in a pilot]. A small VM is enough.
3. **Build the simulator; borrow the queue models.** hftbacktest has the best-documented queue
   models (risk-averse; probabilistic power/log, which split cancellations between ahead of and
   behind your order) and latency models [V]. But it is market-by-price only, has no settlement or
   binary payoff, and its stock fee models are linear in trade value, while Kalshi and Polymarket US
   charge `θ·C·p(1−p)` with rounding. NautilusTrader's Kalshi adapter is "building" (not usable) and
   it has no Polymarket US adapter [V]. Recommendation: a ~1-2k-line event-replay simulator inside
   Albert that reuses `fees.py`, `risk.py` and the engine's settlement logic, and implements
   hftbacktest-style queue models [I].
4. **Live testing must calibrate the fill model against reality, and Kalshi makes this unusually
   easy.** Kalshi exposes your **true queue position** (`GET /portfolio/orders/{id}/queue_position`,
   "number of preceding shares") [V], and book deltas caused by your own order carry your
   `client_order_id` [V]. A 1-contract shadow-live phase can therefore measure the queue model's
   error directly, not just fill rates. The Kalshi demo environment is for API plumbing only:
   Kalshi warns that its prices and behavior "may not be reflective of those in real markets" [V].
5. **Go/no-go:** fix the strategy in advance, then require positive event-clustered PnL with t ≥ 2,
   simulated-versus-live fill rates and markout curves that agree, and a majority of positive weeks.
   That takes about **6-10 weeks of recording** (dev, then holdout) before tiny-live, then **4-8
   weeks of tiny-live** (section 4.4).

---

## 1. Historical data availability

### 1.1 Venues' own data

| Venue | What history exists | Order book history? | Source |
|---|---|---|---|
| Kalshi | Live/historical split, with cutoffs from `GET /historical/cutoff`. Historical endpoints: markets, candlesticks, trades, and the user's own fills/orders/positions. | **No.** `GET /markets/{ticker}/orderbook` is current state only. | [V] [Historical Data](https://docs.kalshi.com/getting_started/historical_data.md), [docs index](https://docs.kalshi.com/llms.txt) |
| Kalshi | Candlesticks at 1, 60 or 1440 minutes | n/a | [V] [Get Market Candlesticks](https://docs.kalshi.com/api-reference/market/get-market-candlesticks.md) |
| Kalshi | Trades include block trades, flagged `is_block_trade` | n/a | [V] [Get Trades](https://docs.kalshi.com/api-reference/market/get-trades.md) |
| Polymarket US | REST book/BBO (12 req/min each for `GetOrderBook`/`GetBBO`), candlestick guide, CSV reports of *your own* orders/trades | **No** public history found | [V] [rate limits](https://docs.polymarket.us/trader-guide/rate-limits.md), [docs index](https://docs.polymarket.us/llms.txt) |
| Polymarket intl | On-chain `OrderFilled` events (every fill, attributable), Data API trades, `prices-history` | **No.** The CLOB book is off-chain, so fills are on-chain but resting orders and cancels are not. | [V] [Goldsky Polymarket datasets](https://docs.goldsky.com/chains/polymarket); [V-3p] [microstructure note](https://avdheshcharjan.github.io/notes/polymarket-mm.html) ("no resting-book data") |

Oddpool, which sells captured Kalshi/Polymarket books, states: "There is no backfill API," and its
archive exists only because it has been capturing in real time [V-3p, vendor page]
([oddpool institutional](https://www.oddpool.com/institutional)).

**Polymarket intl on-chain fills.** Goldsky says the old public subgraph endpoints "will return
incomplete or incorrect data" and points to its v2 Turbo Pipelines datasets (`polymarket.order_filled`)
[V-3p] ([Goldsky](https://docs.goldsky.com/chains/polymarket)). A community repo reports that
Polymarket moved to new CTF Exchange contracts on 2026-04-28 [V-3p]
([poly_data](https://github.com/pranksteess/poly_data)). The Hugging Face `SII-WANGZJ/Polymarket_data`
dump has about 293M `OrderFilled` rows, but a user reports missing events for a known transaction
[V-3p] ([discussion](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/discussions/8)). Use it
for trade-tape research, not for queue modeling.

### 1.2 Third-party L2 / tick vendors

All of these are vendor captures, so their coverage starts when the vendor began recording.

| Vendor | Venues | Granularity | Coverage start | Price | Notes / source |
|---|---|---|---|---|---|
| **Oddpool** | Kalshi, Polymarket | Snapshots, deltas, trades, ms timestamps; daily Hive-partitioned Parquet to S3, REST/WS | Kalshi from **2026-03-19** | Kalshi endpoints "available on all plans (Free tier included)"; institutional pricing on request | [V-3p] [Kalshi overview](https://docs.oddpool.com/kalshi/overview.md), [institutional](https://www.oddpool.com/institutional). About 1 h delay. Also publishes PredictionMarketBench (section 3.4). |
| **Predexon** | Kalshi, Polymarket, Dflow | Kalshi book snapshots (whole-cent rounding; the endpoint is "to be deprecated" in favour of a sub-cent history endpoint); Polymarket per-token L2 snapshots; bulk Parquet "raw tick stream" | Kalshi from **2026-01-07**; Polymarket L2 from 2026-01-01 | Kalshi orderbook endpoint "Free & Unlimited"; $50 signup credits; tick slices quoted on request | [V-3p] [Kalshi orderbooks](https://docs.predexon.com/api-reference/kalshi/orderbooks), [quant guide](https://docs.predexon.com/start-here/quant-guide). Snapshot frequency not stated. |
| **EntityML** | Kalshi, Polymarket | Order book for a *subset* of markets | Polymarket 2025-09-01, Kalshi 2026-02-17; "coverage is not continuous" | n/a | [V-3p] from search snippet of docs.entityml.com (site unreachable on 2026-11-04) |
| **KalshiBook** | Kalshi | L2 deltas, trades, point-in-time book reconstruction (Python SDK) | not stated | not stated | [V-3p] [PyPI](https://pypi.org/project/kalshibook/) |
| **Dome API** | Polymarket, Kalshi | `/polymarket/orderbooks`, `/kalshi/orderbooks` historical snapshots | not stated | Aggregators list Starter $39, Pro $129, **Orderbook $249/mo**; reportedly acquired by Polymarket in Feb 2026 | [V-3p] [SDK](https://pypi.org/project/dome-api-sdk/), [pm.wiki comparison](https://pm.wiki/uk/compare/dome-vs-probalytics). Unverified. |
| **Allium** | Kalshi, **Polymarket US** | Snapshot tables, one row per price level (`kalshi_market_orderbook`, `polymarket_us_market_orderbook`) | not stated | Enterprise (Snowflake etc.) | [V] schema only: [Kalshi](https://docs.allium.so/historical-data/predictions/kalshi/market-orderbook.md), [PM US](https://docs.allium.so/historical-data/predictions/polymarket-us/market-orderbook.md). Frequency and depth not documented. **Only Polymarket US book history found anywhere.** |
| **pmxt archive** | Polymarket intl (folders for Kalshi and Opinion also listed) | Hourly Parquet of `book_snapshot` + `price_change` events, sorted by market/asset/receipt time | Polymarket v2 from **2026-04-13 19:00 UTC** | **Free** | [V-3p] [v2 overview](https://archive.pmxt.dev/docs/v2-data-overview) (site refused connections on 2026-11-04; figures from search snippets): ~31 GB for 94 hourly files, 100-400 MB per file. |
| **Telonex** | Polymarket intl (not Kalshi or Polymarket US) | `book_snapshot_5/25/full`, "every websocket frame recorded tick-by-tick" | not stated | Free trial of 5 files, paid tiers | [V-3p] [docs](https://telonex.io/docs), [FAQ](https://telonex.io/faq) |
| **FinFeedAPI (CoinAPI)** | "Prediction markets" (Polymarket in examples) | Historical raw add/update/delete book events | not stated | free credits, then paid | [V-3p] [dataset page](https://www.finfeedapi.com/datasets/prediction-markets-order-books) |
| **fable5-kalshi-l2** (Hugging Face, gated) | Kalshi | L2 snapshots, trades, settlements | not stated | free, gated | [V-3p] [dataset](https://huggingface.co/datasets/naratron33/fable5-kalshi-l2) |
| Kaggle | none found for Kalshi L2 | | | | search, 2026-11-04 |

**Implications [I]:**
- No vendor can give Albert multi-year Kalshi L2. The best case is about 7-10 months (Jan-Mar 2026
  onward). Every vendor needs a gap/completeness audit before use, and snapshot-only feeds (Allium,
  Predexon Kalshi, Dome) are too coarse for queue modeling.
- **Cheapest useful backfill:** Oddpool's free tier for Kalshi (snapshots + top-of-book + trades
  since 2026-03-19). Use it to *pre-screen* which categories and price regions have spreads and
  depth worth quoting, and to cross-check our own recorder on overlapping days. Do not use it as
  the backtest source of record unless its deltas pass the same seq/gap audit as our own capture.
- pmxt (Polymarket intl, free) is a good free sandbox for building and debugging the queue
  simulator before our own Kalshi data has accumulated. Its sports markets mirror Polymarket US
  games, and intl prices track US within about 1c (README §4) [L]. It is not tradable and its
  queue dynamics differ, so it is a development dataset, not evidence.
- Ground rule 1 applies to vendor data: record a real vendor response into `tests/fixtures/<vendor>/`
  before writing a parser.

### 1.3 Sportsbook / odds reference prices

| Source | What | Price | Source |
|---|---|---|---|
| **The Odds API** | Many books including Pinnacle (paid plans: "All bookmakers"). Historical snapshots from 2020-06-06, at 10-min then **5-min** intervals (from Sep 2022). Historical costs 10 credits per region per market. | Free 500 credits; $30 (20k), $59 (100k), $119 (5M), $249 (15M) per month | [V] [home/pricing](https://the-odds-api.com/), [v4 guide](https://the-odds-api.com/liveapi/guides/v4/) |
| Pinnacle direct | Public API closed **2025-07-23**; now only commercial or "select high value bettors" | n/a | [V-3p] [dev.to](https://dev.to/ryankr/pinnacle-killed-its-public-api-heres-how-to-get-pinnacle-odds-in-2026-with-code-39pe), [odds-api.io](https://odds-api.io/blog/pinnacle-api-shutdown-alternatives) |
| Pinnacle relays (pinnapi, PinnOdds, SharpOddz) | Live Pinnacle odds (REST/SSE/WS); generally **no history** | pinnapi free 100 req/day, paid from $99/mo | [V-3p] [sportsapis.dev pinnapi](https://sportsapis.dev/apis/pinnapi), [PinnOdds](https://sportsapis.dev/apis/pinnodds) |
| OddsJam API | Historical opening/closing lines on a quote-only API tier | not published | [V-3p] [oddspapi blog](https://oddspapi.io/blog/?p=2860) |
| Betfair Historical Data | BASIC (free, 1-min LTP), ADVANCED (1 s, top 3 levels), **PRO (50 ms, full ladder)** | Paid tiers; prices on Betfair's site | [V-3p] [betcode FAQ](https://betcode-org.github.io/faq/using-data), [Betfair data scientists](https://betfair-datascientists.github.io/data/usingHistoricDataSite) |

**[I]** For a maker strategy, a sharp reference price matters mostly as a *cancel trigger*: pull
quotes when the sharp line moves. Its value is not prediction. Five-minute historical snapshots
(The Odds API) are too coarse to backtest sub-minute cancel logic. For that, record the
reference live alongside the book. Cheapest options: The Odds API at $30-59/mo, polled every 30-60 s
on the handful of leagues traded; or the Polymarket intl CLOB WebSocket (free, no auth) as the
reference for Polymarket-listed games. Adding a paid odds feed can wait until a free-reference
version shows promise.

---

## 2. Self-recording L2 data

### 2.1 Feed facts

**Kalshi** (production `wss://external-api.kalshi.com/...`, demo `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2`)
- The connection **requires authentication even for public channels** [V]
  ([WebSocket Connection](https://docs.kalshi.com/websockets/websocket-connection.md)). The recorder
  therefore needs `KALSHI_API_KEY_ID`/`KALSHI_PRIVATE_KEY` from `.env`. This is the first
  authenticated code in the repo, so key material must never reach raw files, logs or manifests
  (ground rule 8).
- `orderbook_delta`: an `orderbook_snapshot` first (`yes_dollars_fp`, `no_dollars_fp` as
  `[price, count]` string pairs, plus `clock`), then `orderbook_delta` messages (`price_dollars`,
  `delta_fp` signed, `side` yes/no, `clock`, optional `ts_ms`, and `client_order_id` *only when
  your own order caused the change*) [V]
  ([Orderbook Updates](https://docs.kalshi.com/websockets/orderbook-updates)).
- **Book semantics:** the book lists **bids only** for YES and NO. A YES ask at X is a NO bid at
  1−X [V] ([Orderbook Responses](https://docs.kalshi.com/getting_started/orderbook_responses.md)).
  Prices are dollar strings that allow sub-penny levels; sizes are fixed-point strings with
  fractional contracts possible [V]
  ([Fixed-Point](https://docs.kalshi.com/getting_started/fixed_point_migration.md)).
- **Sequencing:** `seq` counts messages *per subscription* and stays contiguous. `clock` is the
  exchange clock, identical across connections in real-time mode. On a gap, send `get_snapshot`
  for all subscribed markets, or unsubscribe and resubscribe. A repeated `subscribe` does **not**
  resync [V].
- **Real-time vs conflated:** `update_interval_ms` = 0 (real-time, the default), 100 or 1000.
  Conflated mode sends net change per level per window, and its clocks can go backwards [V].
  **Record real-time (0).** Conflation hides the add-then-cancel churn that queue models need [I].
- **Limits:** one `orderbook_delta` subscription per connection; a per-subscription market limit
  (error 26, number not documented); error 25 "Subscription buffer overflow" if the client reads
  too slowly; 200 concurrent connections at Basic tier [V]
  ([asyncapi](https://docs.kalshi.com/asyncapi.yaml), [Rate Limits](https://docs.kalshi.com/getting_started/rate_limits.md)).
- Other channels to record: `trade` (public trades: `trade_id`, prices, `count_fp`, `taker_side`,
  `taker_outcome_side`, `taker_book_side`, `ts_ms`), `ticker`, `market_lifecycle_v2` [V]
  ([Public Trades](https://docs.kalshi.com/websockets/public-trades.md), [docs index](https://docs.kalshi.com/llms.txt)).

**Polymarket US**
- WebSocket `wss://api.polymarket.us/v1/ws/markets`, **authenticated**. Subscription types
  `SUBSCRIPTION_TYPE_MARKET_DATA` ("full order book and market stats", top levels only),
  `_MARKET_DATA_LITE` (BBO plus depth counts) and `_TRADE` (price, qty, `tradeTime`, maker/taker side
  and intent). Up to 100 markets per subscription. An optional `responsesDebounced` batches updates
  (avoid it for recording). **No sequence number is documented** [V]
  ([Markets WebSocket](https://docs.polymarket.us/api-reference/websocket/markets.md)).
- gRPC `MarketDataSubscriptionAPI`: needs only an Auth0 JWT with `read:marketdata` scope, with no
  participant ID or KYC. Options: `symbols=[]` subscribes to all instruments; `depth` (default 10);
  `unaggregated=True` for the "raw order book" rather than levels. Up to 1,000 symbols per stream and
  20 concurrent streams per firm [V]
  ([Market Data Streaming](https://docs.polymarket.us/streaming-endpoints/market-data-stream.md),
  [rate limits](https://docs.polymarket.us/trader-guide/rate-limits.md)).
  **[I] If "raw" means per-order (L3) data, Polymarket US queue positions could be reconstructed
  exactly. Capture a sample first to confirm.**
- Stream semantics: **at-least-once delivery, so deduplicate**; market data starts with a snapshot
  and then updates; "some streams send periodic snapshots", and you should replace state on each
  snapshot, not merge [V]
  ([Streaming](https://docs.polymarket.us/trader-guide/streaming-apis.md)).

**Polymarket intl** (data only): `wss://ws-subscriptions-clob.polymarket.com/ws/market`, no auth;
messages `book`, `price_change` (with optional `hash`, `bestBid`/`bestAsk`), `last_trade_price`,
`tick_size_change`. **No sequence numbers documented.** Heartbeat is a PING every 10 s [V]
([market channel](https://docs.polymarket.com/developers/CLOB/websocket/market-channel)).

### 2.2 Storage design that fits the append-only rule [I]

The current raw cache stores one file per HTTP response, keyed by sha256(url+params)
(`src/albert/http.py`, README "Data layout") [L]. That does not fit a stream. Proposed layout:

```
data/raw/stream/<venue>/<channel>/<YYYY-MM-DD>/<HH>-<conn_id>.jsonl.zst
    one line per received frame: {"recv_ns": ..., "conn": ..., "frame": <verbatim JSON text>}
    written to <name>.part, closed hourly, fsync'd, then renamed. Never touched again.
data/runs/<ts>-record.json      manifest per recorder session: params, market list per hour,
                                closed files + sha256 + line counts, seq ranges, gap list
data/tables/book_events/venue=kalshi/date=YYYY-MM-DD/part-<HH>.parquet   (derived, rebuildable)
    columns: recv_ns, exch_ms (ts_ms), clock, conn, sid, seq, market, kind (snapshot|delta|trade),
             side (yes|no), price (Decimal/int ticks), delta_or_qty, trade_id, taker_side
data/tables/book_gaps.parquet    (market, gap_start_recv_ns, gap_end_recv_ns, reason)
data/tables/book_checkpoints/... full book every 5 min per market (derived), for fast seeking
```

- **Raw text first, Parquet second.** Raw frames are the source of truth that the manifest
  references (ground rules 1 and 3). The Parquet tables are a normalization that can be rebuilt
  from them, the same as the current `tables/` versus `raw/` split.
- Store **prices as integer ticks** (for example price × 10,000) and sizes as integer centi-contracts,
  so that repeated delta accumulation stays exact with sub-penny prices and fractional contracts.
  Float accumulation produces phantom levels, a pitfall another builder also reports [V-3p]
  ([codefixsolution](https://codefixsolution.com/work-with-kalshi-order-book-data-python/)).
- **Gap handling (Kalshi).** Per connection, track the expected `seq`. On a gap, write a gap row,
  call `get_snapshot`, and mark the book invalid from the last good `seq` to the next snapshot.
  The simulator refuses to rest or fill orders in invalid intervals (it cancels them, which is
  conservative). Also request a `get_snapshot` every N minutes and **assert that the reconstructed
  book equals the snapshot**. This is the recorder's main smoke test, and any mismatch is a bug
  [I]. Run two redundant connections for the most important markets and use `clock` to
  cross-check [V: clock is comparable across real-time connections].
- **Gap handling (Polymarket US/intl).** No `seq` is documented, so rely on full-book messages and
  periodic snapshots. Deduplicate (at-least-once). Treat a silence longer than the heartbeat
  interval as a gap. Record the `hash` field where present, and test whether it is a book checksum
  [I].
- **Market universe and survivorship.** The recorder subscribes from a market list chosen *before*
  the market's outcome or final volume is known (for example every open game moneyline in leagues
  X, Y, Z from listing until settlement). Write that list into the hourly manifest. Lesson 2 in
  `docs/HANDOFF.md` (sampling on final volume fabricated edges) applies with full force [L].
- **Consumer speed.** Read the socket in an asyncio task that only timestamps and enqueues frames.
  A separate task compresses and writes. Error 25 (buffer overflow) is the failure to watch for [V].

### 2.3 Expected data volume [I, verify with a 1-hour pilot]

- Local baseline [L]: in `data/tables/kalshi_trade_windows.parquet` (15,427 order windows on
  major-league game markets), total trades divided by total window hours = **~26 trades per
  market-hour** in pre-game windows.
- Book deltas usually outnumber trades several to many times over, because of adds, cancels and
  re-quotes. Assume 5-20× → ~130-520 deltas per market-hour.
- 300 markets × 24 h → **~1-4M frames/day**. At ~250 B of JSON per frame, that is 0.25-1 GB/day
  raw, or **~15-100 MB/day zstd** (the repo's zstd ratio on JSON is ~20× per README [L]). In-game
  periods and live news will spike well above this.
- External reference: pmxt's full Polymarket intl archive is ~100-400 MB of Parquet per hour, which
  is a few GB/day for the *entire* venue [V-3p]. A few-hundred-market Kalshi + Polymarket US capture
  should be one to two orders of magnitude smaller.
- Conclusion: storage is a non-issue, at a few GB/month. The binding constraints are recorder uptime
  and gap rate. Measure bytes/hour, frames/hour and gaps/day in the pilot and record them in the
  manifest.

### 2.4 How long to record [I]

- **1-2 weeks:** recorder stability, gap statistics, snapshot-equality checks, first simulator
  runs. No conclusions about edge.
- **Edge evaluation needs independent events, not hours.** Previous maker results flipped sign by
  month (+2.1% / +4.2% / −8.7% Aug/Sep/Oct, README §5) [L], and a t ≈ 1 result needed thousands of
  fills. For each category tested, plan on **≥ 4 weeks of development data plus ≥ 4 weeks of
  untouched holdout**, and at least ~300 independent events (games or dates) in each. With weather
  brackets (24 cities × daily), 4 weeks is ~670 city-days; with major leagues it is several hundred
  games. Sports seasons change the mix, so the holdout should be the *later* period.
- **Recording is the critical path.** Start it now, before the simulator exists. Every day not
  recorded is lost permanently, because no vendor backfills (section 1.1).

---

## 3. Backtesting engines

### 3.1 Requirements specific to Albert

1. **Binary payoff and settlement:** positions settle at 0 or 1 at a resolution time; separate cash
   per venue (as in `backtest/engine.py`) [L].
2. **Mirrored books:** Kalshi YES/NO bid-only books → one YES book (YES ask = 1 − NO bid). Kalshi's
   V2 order API uses a "single-book `bid`/`ask` side" [V]
   ([Create Order V2](https://docs.kalshi.com/api-reference/orders/create-order-v2.md)). Polymarket US
   has long/short intents on one book [V] (markets WS order intents).
3. **Fees:** Kalshi `0.07·mult·C·P(1−P)` taker, maker fee only on maker-fee series, with six-decimal
   trade fees and **balance rounding to $0.01 for non-direct members, carried by a per-order fee
   accumulator** [V] ([Fee Rounding](https://docs.kalshi.com/getting_started/fee_rounding.md)).
   Polymarket US taker `0.0695·C·p(1−p)`, with maker rebates only from tier 2 ($10M+ maker volume)
   [V] ([PM US fees](https://docs.polymarket.us/fees.md)). Neither fee is linear in notional.
4. **Queue position** under price-time priority. Kalshi says queue position "is determined using a
   price-time priority" [V]
   ([queue position](https://docs.kalshi.com/api-reference/orders/get-order-queue-position.md)).
5. **Latency:** feed latency (exchange `ts_ms` versus our receive time) and order round trip.
6. **Liquidity rewards** (Kalshi incentive programs, `incentive_type` = `liquidity`, with
   `target_size_fp` and `discount_factor_bps` fields [V]
   ([Get Incentives](https://docs.kalshi.com/api-reference/incentive-programs/get-incentives.md));
   Polymarket US Liquidity Incentive Program, which rewards "resting orders, whether they fill or
   not" [V] ([incentives](https://docs.polymarket.us/incentives/overview.md))). These can be a large
   part of maker PnL, and **no open simulator models them** [I; prediction-market-backtesting says
   "Liquidity Rewards ... are not modeled", V-3p].

### 3.2 Candidates

| Engine | Strengths | Gaps for Albert | Source |
|---|---|---|---|
| **hftbacktest** (Rust core, Python/Numba API) | Queue models: `RiskAverseQueueModel` (advance only on trades at your price; cancels assumed behind you) and `ProbQueueModel` with power/log functions (decreases split ahead of and behind you). `NoPartialFill` / `PartialFill` exchange models. Latency: constant, `IntpOrderLatency` (from recorded order RTTs), feed-latency-derived. Event format with both `exch_ts` and `local_ts`. The author stresses that queue modeling matters most for **large-tick assets**, which 1c ticks on binaries are. | "Currently only supports Market-By-Price" (L3 only via Rust traits). Fee models are `TradingValueFeeModel` (rate × value) and flat-per-trade, so p(1−p) needs a custom Rust `FeeModel`. No settlement or binary payoff; no multi-venue cash. Requires exch_ts < local_ts on every event. The author's advice: pick the queue model "by comparing backtest and real trading results". | [V] [order fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html), [latency](https://hftbacktest.readthedocs.io/en/latest/latency_models.html), [data](https://hftbacktest.readthedocs.io/en/latest/data.html), [prob queue tutorial](https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html), [large tick tutorial](https://hftbacktest.readthedocs.io/en/latest/tutorials/Queue-Based%20Market%20Making%20in%20Large%20Tick%20Size%20Assets.html), [fee source](https://docs.rs/crate/hftbacktest/latest/source/src/backtest/models/fee.rs) |
| **NautilusTrader** | Polymarket (intl) adapter "stable": `BinaryOption` instruments, L2 MBP deltas, `PolymarketFeeModel`, `PolymarketDataLoader`. Backtests support `trade_execution` + `queue_position=True` (displayed size ahead, reduced by trades and book decreases) and `liquidity_consumption`. | **Kalshi adapter status "building" ("likely not in a usable state")**; **no Polymarket US** adapter. Heavy framework. Its queue heuristic is MBP-only, and it warns that `NO_AGGRESSOR` trades "can clear a queue and fill an order earlier than reality". | [V] [integrations](https://nautilustrader.io/docs/nightly/integrations), [Polymarket guide](https://nautilustrader.io/docs/latest/integrations/polymarket/), [trade execution](https://nautilustrader.io/docs/latest/concepts/backtesting/trade-execution/) |
| **prediction-market-backtesting** (evan-kolberg; NautilusTrader-based) | Full L2 replay of Polymarket from the pmxt archive and Telonex; gap → book reset until the next snapshot; static 75 ms latency + insert/cancel delays; maker rebate modeled. | Kalshi is trade-tick replay only; queue is "an MBP heuristic, not true venue FIFO reconstruction"; rewards not modeled. | [V-3p] [execution modeling](https://evan-kolberg.github.io/prediction-market-backtesting/execution-modeling/), [repo](https://github.com/evan-kolberg/prediction-market-backtesting) |
| **PredictionMarketBench** (Oddpool, MIT) | Kalshi episodes (book + trades + settlement); maker/taker modes; Kalshi fee formula. | Only 4 episodes. Queue is "displayed size and **pro-rata** fill allocation", which **contradicts Kalshi's price-time priority**. No latency. Built as an LLM-agent benchmark. | [V-3p] [paper](https://arxiv.org/html/2602.00133), [repo](https://github.com/Oddpool/PredictionMarketBench) |
| pm-hftbacktest (PyPI) | Described as a Polymarket HFT/market-making backtester | Page would not load; maintenance unknown | [V-3p] [PyPI](https://pypi.org/project/pm-hftbacktest/) |
| Backtrader / vectorbt | n/a | Bar- or vector-based. No order book, no queue, no binary settlement. **Unsuitable** [I]. | |
| Albert's `backtest/maker.py` | Already has trade-driven fills, improve-versus-join queue rules, settlement, fees, game clustering. | Hourly candles + trades; no book state, so the queue ahead when joining is unknown; it assumes "other makers did not jump ahead of us" [L]. | `src/albert/backtest/maker.py` |

### 3.3 Recommendation: build, borrowing hftbacktest's models [I]

Build `albert/sim/` (≈1-2k lines) as an event-replay simulator over the recorded `book_events`
Parquet:

- **Book:** integer-tick YES book per market, reconstructed from snapshots and deltas, with gap
  intervals respected.
- **Order lifecycle:** strategy decision at `recv_ts` → order arrives at `recv_ts + entry_latency`
  (sampled from measured RTTs, so `IntpOrderLatency`-style) → resting → cancel arrives at
  `t + cancel_latency`. The order sees only events with `recv_ts` ≤ its decision time (no
  look-ahead), while fills happen in exchange time (`ts_ms`/`clock`).
- **Queue models, pluggable:**
  1. `RiskAverse`: queue ahead = displayed size at arrival; only trades at our price reduce it;
     all cancels at the level come from behind us.
  2. `Prob(f)`: a level decrease not explained by a trade is attributed ahead of us with probability
     `f(ahead/(ahead+behind))`, with power-n and log variants (hftbacktest's formulation).
  3. `Improve`: when our price creates a new level, queue ahead = 0 (already used in `maker.py`).
  - Fills come only from trades on the passive side at our price. Trades *through* our price fill
    us fully up to their size. Exclude `is_block_trade` prints. Allow partial fills.
- **Reuse:** `fees.py` (extended to Kalshi's fee accumulator rounding per order), `risk.py`
  sizing/caps, the engine's per-venue cash and settlement, and `game_key_expr` clustering.
- **Outputs:** fills with predicted queue-ahead at fill, markouts (section 4.3), PnL by event, and
  rewards accrual (a separate, explicitly estimated line item).
- **Cross-check:** port one simple quoting rule to hftbacktest on a few days of data (custom fee
  model, or fees applied post hoc) and require fill counts within ~10% of Albert's simulator under
  the same queue model. This catches implementation bugs cheaply.

Why not adopt: (a) neither mature engine has a usable Kalshi or any Polymarket US path [V]; (b) the
data is small, so Python/polars is fast enough at a few million events/day [I]; (c) the binary and
fee specifics would need custom code in either engine anyway; (d) Albert's ground rules (manifests,
recorded fixtures, no look-ahead tests) are easier to enforce in its own code [I].

---

## 4. Live-testing methodology

### 4.1 Kalshi demo: plumbing only

Kalshi: "The price and behavior of markets in the demo environment may not be reflective of those in
real markets" [V] ([demo env](https://docs.kalshi.com/getting_started/demo_env.md)). A report on
Kalshi's *margin* demo says its activity "appears to be synthetic" [V-3p]
([litecs](https://litecs.ubi.pt/en/?p=17591)). No source documents real flow in the event-contract
demo. **Use demo only to test** signing, order create/amend/cancel, `user_orders`/`fill` channels,
the queue-position endpoint, order groups, kill-switch behaviour and rate-limit handling. Demo fill
statistics are meaningless [I].

### 4.2 Three stages

1. **Recorded replay** (simulator on recorded data): development on weeks 1-4, holdout on weeks 5-8.
2. **Forward paper on the live book** [I]: run the same strategy code inside the recorder process.
   Orders are virtual, and fills come from the simulator's queue model applied to *live* events.
   This tests the strategy against data that did not exist when it was designed, and it tests
   operations (restarts, gaps, decision latency). The existing `paper.py` loop (polling REST every
   `--interval` seconds) is too slow for sub-minute logic. The paper strategy should subscribe to the
   recorder's in-memory book.
3. **Shadow-live at 1 contract** [I]: real orders, minimum size, same signals. In parallel, run the
   *virtual* twin of each real order through the simulator. Real fills are compared with virtual
   fills order by order.

### 4.3 What to measure in shadow-live

- **True queue position.** Poll `GET /portfolio/orders/queue_positions` (filter by `market_tickers`
  or `event_ticker`; `queue_position_fp` = "number of preceding shares") [V]
  ([endpoint](https://docs.kalshi.com/api-reference/orders/get-queue-positions-for-orders.md)) at a
  fixed cadence, within the Read token budget (Basic tier: 200 tokens/s, 10 per request by default
  [V]). Compare with the simulator's `ahead` estimate over time. The fraction of non-trade level
  decreases that turned out to be *ahead* of us is the parameter of the probabilistic queue model,
  so fit `f` to it directly [I]. Our own order's add and cancel appear in the delta stream with our
  `client_order_id` [V], which anchors our position in the recorded book exactly.
- **Fill-rate calibration:** bin orders by predicted fill probability (or by queue-ahead at
  placement and time-to-cancel) and plot realized against predicted, with a reliability curve and
  Brier score. Accept the model if realized/predicted fill counts are within ±20% overall and in the
  main bins, and if errors are not one-sided (one-sided means optimism) [I].
- **Adverse selection: markout curves.** For each fill, from the maker's side, compute
  `sign × (mid(t+h) − fill_price)` at h = 1 s, 10 s, 60 s, 5 min, plus `payoff − fill_price` to
  settlement. Use mid-to-mid and say which quote is used (last at or before t+h) [V-3p, methodology]
  ([kx markout](https://kx.com/glossary/what-is-markout-and-how-is-it-calculated/),
  [QuestDB cookbook](https://questdb.com/docs/cookbook/sql/finance/markout/)). Report curves with CIs,
  medians and percentiles, not one horizon [V-3p]. Remember that maker fills are *conditioned* on
  the price moving into the order [V-3p], which is exactly the adverse selection that sank the
  hourly maker test (README §5) [L]. **Compare live markout curves with the simulator's curves
  for the virtual twins.** If live is worse, real fills are more toxic than the simulator assumes;
  typically you are filled when faster participants have already cancelled [I].
- **Benchmarks for scale:** on Polymarket intl on-chain data, every decile of makers had positive
  volume-weighted markouts at 10 s / 1 min / 10 min, but those markouts are measured against later
  *trades*, so they mix spread capture with selection [V-3p]
  ([note](https://avdheshcharjan.github.io/notes/polymarket-mm.html)). Bürgi, Deng & Whelan find
  Kalshi makers lose much less than takers on average (VoxEU summary: takers ≈ −32%, makers ≈ −10%)
  [V-3p] ([VoxEU](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market),
  [paper](https://www.karlwhelan.com/Papers/Kalshi.pdf)). In other words, being the maker is
  necessary but not sufficient.
- **Latency:** log `t_decision`, `t_sent`, the REST ack time, `ts_ms` of our own delta, and the fill
  time. Feed this RTT distribution into the simulator as an `IntpOrderLatency`-style empirical
  latency [V: hftbacktest recommends collecting it by "regularly submitting orders that cannot
  execute"].

### 4.4 Go/no-go statistics for "consistent positive returns" [I]

- **Pre-register:** freeze the strategy, its parameters, the market universe rule, and the metrics
  before the holdout or live window starts. Count every variant tried and apply a deflated-Sharpe
  style haircut ([Bailey & López de Prado, DSR](https://papers.ssrn.com/abstract=2460551)) [V for
  the method's existence].
- **Unit of observation:** PnL per **event** (one game or one city-day, with all its legs and quotes
  combined, as in `game_key_expr`) [L]. Aggregate to daily PnL per dollar of peak capital at risk.
  Cluster SEs by event, and block-bootstrap by day for daily metrics.
- **Minimum length:** with a daily Sharpe `s`, t ≈ s·√N, so t ≥ 2 needs N ≈ 4/s². An annualised
  Sharpe of 3 (s ≈ 0.19/day) needs **~110 trading days**; Sharpe 5 (s ≈ 0.31) needs **~40 days**.
  Hence: 8 weeks of holdout replay plus 4-8 weeks of shadow-live, pooled only when the fill model
  has passed calibration.
- **Go criteria (all required):**
  1. Event-clustered mean PnL > 0 with t ≥ 2 on holdout replay, *and* the same sign in shadow-live.
  2. ≥ 60% of weeks positive; max drawdown within the existing `RiskConfig` limits.
  3. Fill-model calibration within ±20%; the live markout curve is not significantly worse than
     the simulated one at 10 s and 60 s.
  4. PnL net of real fees, including rounding. Rewards are reported separately, and the strategy
     must not depend on them unless the program's terms are stable.
- **Kill criteria:** live markouts at 60 s are worse than simulated by more than the expected edge,
  or fill counts exceed the simulator by more than 30%. Too many fills means toxic flow [I].

---

## 5. Pitfalls specific to maker backtests

| Pitfall | Why it bites | Mitigation |
|---|---|---|
| **Queue-position optimism** | Assuming front of queue, or counting a trade *at* your price as a fill when you joined behind others. Moallemi & Yuan find queue position value is the same order of magnitude as the bid-ask spread [V-3p] ([paper](https://www.econbiz.de/Record/a-model-for-queue-position-valuation-in-a-limit-order-book/10012953025), [arXiv follow-up](https://arxiv.org/pdf/1902.10743)). | Report results under RiskAverse **and** Prob models; calibrate with Kalshi's queue-position endpoint. |
| **Cancellations ahead of you** | RiskAverse assumes every cancel is behind you, which is pessimistic. Assuming every cancel is ahead of you is very optimistic. | Probabilistic split (hftbacktest `ProbQueueModel`) fitted to live queue data. |
| **Touch versus trade-through** | A trade at your price may fill only the queue ahead; a trade through your price implies you were filled. | Fill on touch only once the modeled queue ahead is 0. Always fill on trade-through, up to trade size. |
| **Trade/book ordering and `NO_AGGRESSOR` prints** | Double-counting a trade *and* the book decrease it caused, or letting ambiguous prints clear both sides; NautilusTrader warns this "can clear a queue and fill an order earlier than reality" [V]. | Match trades to deltas by `clock`/time. Use `taker_side`, and drop prints without an aggressor side. Exclude block trades. |
| **Latency** | Decisions use data you would not yet have; cancels land after the adverse move. | Decide on `recv_ts`, use exchange time for matching, and use empirical entry/cancel latency. Zero latency "is optimistic for CLOB strategies" [V-3p, evan-kolberg]. |
| **Self-impact and other bots' reactions** | Joining or improving changes others' behaviour, which replay cannot show [V-3p, PredictionMarketBench lists "strategic interaction" as not modeled]. | Keep size at 1-5 contracts; check against shadow-live. |
| **Survivorship / selection** | Choosing markets by final volume or by "had trades" selects outcomes. This produced fake 2-4× edges here before [L]. | Freeze the universe at listing time and store it in the manifest. |
| **Settlement timing** | Kalshi `close_time` of a settled market is the *actual* close [L]. Resting into live games or into resolution gets filled exactly when informed flow arrives (6h row in README §5) [L]. | Cancel at a scheduled cutoff (ticker start − buffer); model the market lifecycle channel; simulate halts and suspensions (Polymarket US states `SUSPENDED`/`HALTED`) [V]. |
| **Fee rounding at tiny size** | Kalshi non-direct members' balances align to $0.01 per fill, with a per-order accumulator and rebate [V]. A 1-contract maker fill on a maker-fee series can pay a full cent against a ~0.4c model fee [I, from the documented formula]. | Implement the accumulator exactly; compare against the first live fills' reported fees. |
| **Tick, sub-penny and fractional sizes** | Kalshi allows sub-penny prices and fractional contracts; Predexon's Kalshi history rounds to whole cents and merges levels [V-3p]. Polymarket emits `tick_size_change` [V]. | Use integer ticks in storage; handle tick changes as book resets. |
| **Gaps** | Reconstructed books drift silently. | Seq audit, periodic snapshot equality tests, and no simulated activity inside gap intervals. |
| **Rewards double-counting** | Reward programs change weekly and pay pro rata against other makers. | Model rewards separately and conservatively; never let them carry a go decision. |
| **At-least-once streams** | Polymarket US can deliver duplicate messages [V]. | Deduplicate before building books. |

---

## 6. Recommended pipeline (phased for this repo)

Each phase ends with a live smoke run that produces non-empty, sane output (AGENTS.md process).

**Phase R0: pilot capture (2-3 days of work).**
- `albert/venues/kalshi_ws.py`: signed WS connect (keys from `.env` via `config.secret()`), subscribe
  `orderbook_delta` (real-time), `trade`, `market_lifecycle_v2` for a frozen universe (for example
  all open game moneylines in 6 leagues plus all `KXHIGH*` legs for the next 2 days).
- `albert record` CLI: append-only hourly `.jsonl.zst` files plus manifest; seq audit; `get_snapshot`
  every 10 min with a reconstructed-book equality check.
- `albert capture` gains WS fixtures: a trimmed real snapshot, deltas and trades into
  `tests/fixtures/kalshi_ws/` (ground rule 1). Parser tests use them.
- Smoke test: 1 hour on ~50 markets. Report frames/hour, bytes/hour, gaps and equality-check pass
  rate. Zero frames counts as a failure.

**Phase R1: continuous recording (start immediately after R0; runs throughout).**
- Deploy on the existing `deploy/gcp.md` VM path under systemd/Docker, with heartbeat and daily
  manifest. Add Polymarket US (WebSocket `MARKET_DATA` + `TRADE`; trial gRPC `unaggregated=True`
  on a few symbols to see whether it is L3).
- Optional: the Polymarket intl CLOB WS for matched games (free reference mid; data only).
- Normalizer: `albert normalize-stream` → `book_events`, `book_gaps`, `book_checkpoints` Parquet.

**Phase S1: simulator (1-2 weeks of work, in parallel with recording).**
- `albert/sim/`: book reconstruction, order lifecycle with latency, RiskAverse/Prob/Improve queue
  models, Kalshi fee accumulator, settlement via existing engine code, markout and fill reports.
- Develop and debug on the free pmxt Polymarket intl archive (optional) and then on our own
  Kalshi data. Cross-check one rule against hftbacktest.
- Tests: no-look-ahead (a decision never reads events with `recv_ts` > decision time), queue
  invariants, a fee-rounding test with documented worked examples, and gap refusal.

**Phase S2: research on recorded data (weeks 2-8 of recording).**
- Weeks 1-4: development. Weeks 5-8: holdout, run once. Strategies to try first come from
  `market-making.md` §6 (rewarded slow markets, weather brackets overnight, pre-game quotes with a
  reference-price cancel trigger).

**Phase P: forward paper on the live book (≥ 2 weeks, overlapping S2's holdout).**
- The strategy runs inside the recorder against the in-memory book, with virtual fills from the
  simulator. Logged in `data/paper/ledger.jsonl`, as today.

**Phase L0: demo plumbing (2-3 days).** Order client, kill switch, per-order and daily caps,
cancel-all on disconnect, all tested on the Kalshi demo. No fill statistics are taken from demo.

**Phase L1: shadow-live at 1 contract (4-8 weeks, after explicit user enablement).**
- Real 1-contract orders with virtual twins. Collect queue-position polls, latency logs and markouts.
  Apply the go/no-go rules in section 4.4. Scale only after passing them.

**Rough cost [I]:**

| Item | Cost |
|---|---|
| Recorder VM (small always-on instance; existing GCP guide) | low tens of $/month (check current GCP pricing) |
| Storage | a few GB/month; negligible |
| Oddpool free tier (Kalshi backfill cross-check) | $0 |
| pmxt archive (Polymarket intl dev data) | $0 |
| The Odds API (optional reference) | $0-59/month |
| Dome Orderbook tier (if a longer Kalshi/Polymarket book history is wanted) | ~$249/month [V-3p, unverified] |
| Shadow-live capital at risk | 1-contract orders: well under $1 per order; adverse-selection losses during calibration likely a few dollars, within the $50 Kalshi allocation |
| **Total, default path** | **≈ $15-80/month plus a few dollars of live losses** |

**Open questions to verify first:** the per-subscription market limit on Kalshi (error 26); whether
Polymarket US gRPC `unaggregated=True` is per-order (L3); Polymarket US WS message rates and whether
`marketData` messages are full snapshots or deltas; the meaning of Polymarket's `hash` field; and
current Kalshi maker-fee series and active liquidity programs (re-check before L1, ground rule 5).
