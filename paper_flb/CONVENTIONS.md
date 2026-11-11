# Study conventions: "Who Buys the Longshot?" (favourite–longshot bias and retail flow)

This is an academic, descriptive study written up for arXiv in `paper_flb/`. We are documenting
how prediction markets price longshots and favourites, and who trades them. We are not looking for
a trading strategy. Several workstreams run in parallel; this file is the contract between them.

## Research questions
1. **Cross-section.** How large is the favourite–longshot bias (FLB) on Kalshi? How does it vary
   by category, horizon (time to resolution), contract structure (single binary, multi-outcome
   mutually-exclusive event, strike-ladder leg, "by date" deadline contract), liquidity and
   calendar time? *(workstream `xs`)*
2. **Who pays.** Using trade-level data, how do returns differ by liquidity role (taker against the
   resting order), by price bought, by trade size (retail proxy) and by timing (evenings,
   weekends, in-play, attention spikes)? Does small-trade order flow predict returns?
   *(workstream `flow`)*
3. **Retail traders.** With identified wallets, which wallets buy longshots? Are profits
   concentrated in a few wallets, and does skill persist? How does the bias compare across
   venues (Kalshi, Polymarket international, Polymarket US) on matched events?
   *(workstream `wallets`)*
4. **Explanations.** How much of the bias is mechanical (1c tick floor, fees, collateral), how
   much is preference (probability weighting, risk-loving), and how much is information
   (Shin-type insiders)? *(workstream `structural`, which starts later from the `xs` panel)*

## Shared definitions: `src/albert/research/flb_common.py`
Use this module; do not redefine its contents.
- **Side price** `q`: the price paid for the side bought (YES at p gives q = p; NO gives
  q = 1 − p). `won` is 1 if that side paid out.
- **Longshot and favourite.** A longshot is q ≤ 10c; a favourite is q ≥ 90c.
- **Price buckets:** `BUCKET_LABELS`, computed with `bucket_expr()`.
- **Returns per contract:** `won − q − fee`. **Returns per dollar:** `(won − q − fee)/(q + fee)`.
  Report gross returns (fee 0) and net-of-taker-fee returns, and say which one each table shows.
- **Fees:** `fee_pc()`, exact Kalshi fees with series multipliers. Per-order rounding is a
  robustness variant only.
- **Categories:** `coarse_category()` (Kalshi) and `pm_coarse_category()` (Polymarket). Groups are
  Sports, Weather, Politics, Entertainment, Mentions, Science & Tech, Other. **Financial contracts
  (crypto, indices, rates, economics, companies) are out of the project's scope and are
  excluded.** State this in your data section.
- **Trade size:** dollar notional buckets `SIZE_LABELS` (≤ $10, $10–100, $100–1k, > $1k), via
  `size_bucket_expr()`.
- **Inference:** `clustered()` gives means with standard errors clustered by event (or by market
  where an event has one market). Report n, clusters and t.

## Measurement rules (binding)
- **No post-outcome prices.** Never measure calibration with a price observed after the outcome
  was effectively known:
  - Use prices at fixed leads before the *scheduled* resolution or start time: pre-game for
    sports, and before `occurrence_time` or the expected expiration otherwise.
  - Drop markets that closed early because the event happened, unless you use the scheduled
    time.
  - `last_price` at settlement is not a pre-event price.
- **Clustering and weights.** Cluster by event. Report both market-weighted and volume-weighted
  results where they differ.
- **Subsamples.** Show split-half or first-half/second-half robustness for the main results.
- **Multiple testing.** Many cells are reported. Emphasise patterns that hold across categories
  or both halves, not single cells.
- **Real data only.** Every number in the paper must trace to a file under `data/research/flb_*`.

## LaTeX contract (`paper_flb/`)
Each workstream `<ws>` (`xs`, `flow`, `wallets`, `structural`) owns these files. Edit nothing
else, and never edit `main.tex`, `macros.tex` or another workstream's files.
- `sections/data_<ws>.tex`: at most half a page on the data you use: sources, coverage, filters,
  counts.
- `sections/method_<ws>.tex`, `sections/results_<ws>.tex`, `sections/appendix_<ws>.tex`.
- `bib/<ws>.bib`. Before adding an entry, grep `paper_flb/references.bib` and every
  `paper_flb/bib/*.bib` for the title or DOI. Reuse an existing key and never create a
  duplicate. **Verify every reference you add** (web search; journal, year, volume or arXiv
  id). Never invent a citation.
- `figures/fig_<ws>_*.pdf`, produced by `scripts/fig_<ws>.py` with
  `/home/ubuntu/.venvs/paper/bin/python`.
  - `import figstyle; figstyle.apply()`.
  - Widths `figstyle.FULL` (6.5 in) or `figstyle.HALF` (3.15 in).
  - Colours `figstyle.SERIES` / `figstyle.REF`.
  - Save with `figstyle.save(fig, name)`.
  - Read data only from `data/research/flb_<ws>/`.
- Labels are prefixed by workstream, e.g. `tab:xs-calib`, `fig:flow-size`, `eq:wallets-skill`.
  Cross-reference other sections only through these stable labels: `sec:data`, `sec:framework`,
  `sec:method`, `sec:method-<ws>`, `sec:res-<ws>`, `app:<ws>`.
- **Every number and table carries a `% source: data/research/flb_<ws>/<file>` comment.**
- Use `booktabs` and `threeparttable` with notes stating the sample, weights, clustering and
  units. Use the macros in `macros.tex` (`\q`, `\won`, `\fee`, `\ret`, `\LS`, `\FAV`, `\shinz`,
  `\E`, `\Prob`, `\cents`, …).
- **Style:** academic, plain and concise; present tense for results.
  - Define every symbol.
  - No trading-advice language.
  - Do not overclaim: give t-statistics or confidence intervals.
  - Use subsections and short paragraphs, with lists where they help.
- **Compile check without clobbering other workstreams:**
  `cd paper_flb && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=check_<ws> main.tex`.
  The build must produce no errors and no undefined references from your files.

## Code, data and resources
- Code goes in `src/albert/research/flb_<ws>.py`, with a CLI `python -m albert.research.flb_<ws> run <step>`.
  Tests go in `tests/test_flb_<ws>.py`. Outputs go under `data/research/flb_<ws>/`, including a
  `summary.json` and a short `verdict.md`. Keep `.venv/bin/python -m ruff check src tests` and
  your tests clean. Do not commit.
- **Data fetching:**
  - Public endpoints only, through the repo's cached clients (raw files are append-only;
    `AGENTS.md` rule 3).
  - Stay at or below **3 requests/s** per workstream.
  - Polymarket hosts only where their own robots.txt allows us.
  - No authentication-only data and no orders of any kind.
- **Shared machine:** 2 vCPU and 7 GB RAM, shared with live trading bots, a recorder and other
  agents.
  - Run heavy steps as `nice -n 15 env POLARS_MAX_THREADS=1 .venv/bin/python -m ...` in the
    background with logs.
  - Keep peak RSS **≤ 1 GB**, using lazy scans and chunking.
  - Check `free -m` before a heavy step and wait while available memory is under 2 GB.
- **Never touch** `albert-live-maker@*`, `albert-mentions@prod`, `albert-record`, `data/live/`,
  `deploy/` or `.env`.
