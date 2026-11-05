# Live (cash) maker pilot: plan

Status (2026-11-05, updated): **order layer built and tested offline; waiting for Kalshi demo
keys in `.env`** to run the demo plumbing test. Nothing has placed an order yet. Prod is gated
(`KALSHI_LIVE_ENABLED` is not set). The live paper maker and the recorder keep running.

Built (commit 8437d8a), see "Software to build" for the design:

- `venues/kalshi_trade.py`: demo/prod scoped client. Write allowlist: single create/cancel and
  order groups only (Cancel All and batch cancel unreachable); `cancel_own(prefix)`; client-side
  Write bucket (100 tokens/s, create 10, cancel 2, verified with `GET /account/endpoint_costs`).
- `live_l2.py`: `LiveBroker` (Simulator interface, own orders netted from the book),
  `run_actions` worker, `reconcile`, `LiveMaker` (feeds, guards, ledger, queue-position
  sampling), `plumbing` (demo-only end-to-end test that saves real responses for fixtures).
- CLI: `albert live check` (read-only; works on prod now), `albert live cancel-own`,
  `albert live plumbing` (demo), `albert live l2 --env demo|prod [--live]`.
- Tests: parsers on recorded read-only portfolio payloads; fake exchange for the broker. The
  private WebSocket shapes (`user_order`, `fill`) are built from the recorded REST objects and
  must be checked against the demo recordings.

Next, once demo keys exist: `albert live check --env demo`, `albert live plumbing`, trim its
recordings into fixtures and fix any parser drift, then a short `albert live l2 --env demo`.

Decisions taken (user, 2026-11-05): build the order layer now, test it on Kalshi demo, then run a
supervised 1-contract live pilot (about 20 markets, $10 daily loss cap) after about a week of
paper data. Kalshi only. Stay on the Mac for the supervised pilot (VM deferred). In-play sports
stay in for paper; decide separately for live. Polymarket US on hold.

## Account facts (checked 2026-11-05, read-only calls)

- The key in `.env` (`KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY`) is a **production** key:
  `GET /portfolio/balance` returns 200 on prod and 401 `NOT_FOUND` on demo. Kalshi does not share
  credentials between environments, so demo needs its own account and key.
- Prod balance: $95.54 cash, all on shard 0 (shards 1-3 are $0; NBA/MLB/tennis need shard 3).
- **Pre-existing manual position and order: `KXIMPEACH-29-JAN20`** (67.22 contracts, about $49
  exposure) with a resting order at 0.92 for 67.22. The bot must never cancel or trade anything
  it did not create: never call Cancel All Orders (it cancels across every shard and subaccount),
  only cancel orders whose `client_order_id` carries the bot's prefix or that sit in the bot's
  order group, and exclude markets where we hold a manual position.
- Consider a dedicated subaccount (1-63) for the bot so positions and orders are separated by the
  exchange itself.

## What the user needs to do

1. Create a demo account at demo.kalshi.co and an API key there.
2. Add to `.env`: `KALSHI_DEMO_API_KEY_ID` and `KALSHI_DEMO_PRIVATE_KEY` (same formats as prod).
3. Before the live pilot: confirm the prod key has trading permission, choose the money at risk
   and the loss caps, and decide on in-play sports for live.

## Kalshi API facts for the order layer (docs.kalshi.com, fetched 2026-11-05)

- Environments: prod REST `https://api.elections.kalshi.com/trade-api/v2` (or
  `external-api.kalshi.com`), WS `wss://api.elections.kalshi.com/trade-api/ws/v2`; demo REST
  `https://demo-api.kalshi.co/trade-api/v2`, WS `wss://demo-api.kalshi.co/trade-api/ws/v2`.
- **Create Order V2**: `POST /portfolio/events/orders`. Required: `ticker`, `side` (`bid` = buy
  YES at `price`; `ask` = sell YES, equivalent to buying NO at 1 - price), `count` (string, 0.01
  granularity), `price` (string dollars), `time_in_force` (`good_till_canceled` |
  `immediate_or_cancel` | `fill_or_kill`), `self_trade_prevention_type` (`taker_at_cross` |
  `maker`). Optional: `client_order_id`, `post_only`, `expiration_time` (Unix seconds, with
  good_till_canceled), `cancel_order_on_pause`, `reduce_only` (IOC only), `subaccount`,
  `order_group_id`, `exchange_index` (-1 = auto-route by ticker). Response 201:
  `order_id`, `client_order_id`, `fill_count`, `remaining_count`, `ts_ms`. The legacy
  `/portfolio/orders` endpoint is being deprecated.
- **Cancel Order V2**: `DELETE /portfolio/events/orders/{order_id}?market_ticker=...` (auto-route);
  returns `order_id`, `client_order_id`, `reduced_by`, `ts_ms`. Batch cancel exists; batches cost
  the same tokens per item.
- **Order groups**: `POST /portfolio/order_groups/create` with `contracts_limit` (matched
  contracts per rolling 15 s). When hit, every order in the group is cancelled and new orders are
  blocked until reset. Also delete (cancels all its orders), trigger, reset, update limit.
- **Private WS channels**: `user_orders` (order created/updated: `order_id`, `ticker`, `status`
  resting/canceled/executed, `book_side`, `yes_price_dollars`, `fill_count_fp`,
  `remaining_count_fp`, `maker_fees_dollars`, `client_order_id`, `order_group_id`), `fill`
  (`trade_id`, `order_id`, `client_order_id`, `market_ticker`, `is_taker`, `yes_price_dollars`,
  `count_fp`, `fee_cost`, `post_position_fp`, `purchased_side`, `ts_ms`), `market_positions`,
  `order_group_updates`.
- Queue position: `GET` queue positions for all resting orders (for measuring fill-model error).
- **Rate limits**: token buckets, most requests cost 10 tokens; separate Read and Write buckets;
  batch cancel is 2 tokens per order. Basic-tier Write bucket holds one second of budget (no
  burst). `GET /portfolio/balance` has its own bucket, once per second on Basic. 429 on limit, no
  Retry-After: back off. 200 concurrent WebSocket connections on Basic. An Advanced usage level
  can be granted via `POST` upgrade once one of the last 100 orders came from the API.
- Kalshi notes that REST state lags the exchange slightly; combine order responses with the WS
  channels for the current state.

## Software to build

1. `venues/kalshi_trade.py`: environment-scoped client (`demo` | `prod`), create (post-only, GTC
   with `expiration_time`, `cancel_order_on_pause`, bot `client_order_id` prefix, order group),
   cancel, list own resting orders, positions, balance, order-group management. Prod requires
   an explicit enable (a `.env` flag plus a CLI flag) on every run.
2. **Live exchange adapter** with the Simulator's interface (`place`, `cancel`, `live_orders`,
   `position`, `fills`, `markets`, `ext_at`), so `Quoter` runs unchanged. It must **subtract our
   own resting orders from the book the strategy sees**. Otherwise improve logic treats our quote
   as the external best and bids against itself.
3. **Shadow simulation**: the paper Simulator runs on the same feed. Log predicted versus actual
   fills, plus queue positions, to measure the fill model's error. Built so far: exchange queue
   position of every resting order vs the size the net book shows at its price (ledger `queue`
   records, every 30 s). Per-order shadow fills are still to do; until then compare live fills
   with the concurrent paper run.
4. **Guards**:
   - per-order, per-market and gross-contract caps, plus a daily loss cap;
   - kill switch file;
   - on startup and shutdown, cancel the bot's own orders only;
   - cancel all of the bot's orders on a feed gap, disconnect, stale book or exception;
   - reconcile positions and balance each minute and halt on any mismatch;
   - exchange-side backstops: an order group contracts limit, `expiration_time` on every order
     (dead-man's switch while on the Mac), and `cancel_order_on_pause`.
5. **Demo plumbing test**: place, cancel, partial fill, expiry, order-group trigger, reconnect, and
   the kill switch. Demo does not behave like the real market; it proves plumbing only.

## Rollout

1. Demo plumbing test passes.
2. Live, 1 contract, about 20 markets, $10 daily loss cap, supervised, after about a week of
   paper data.
3. After 1-2 weeks, widen if shadow fills match real fills and PnL tracks paper.
4. Scale only on the go/no-go: event-clustered t ≥ 2 and ≥ 60% positive weeks, on settlement.
