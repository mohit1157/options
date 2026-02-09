# Autonomous Trading Bot (Stocks + Options Trading Bot (Alpaca + X/RSS + Sentiment))

⚠️ **Educational use only. Not financial advice.** Automated trading carries significant risk, including rapid losses. 
Start with **paper trading** and thoroughly test before using real money.

## What this project is
A modular, **event-driven** Python project scaffold that:
- Pulls market-moving info from:
  - **X (Twitter) handles** (e.g., deltaone, elonmusk, donaldtrump) via **X API v2** (Basic $200/mo tier assumed)
  - **News/RSS feeds** via free RSS sources
- Runs **sentiment analysis** via a pluggable **LLM sentiment adapter** (e.g., Grok or any other model)
- Trades via **Alpaca**:
  - Orders, positions, account, paper/live
  - Risk limits, stop loss / take profit
  - Market data (bars/trades/quotes depending on your Alpaca plan)

The code includes:
- A working RSS ingestor
- A working Alpaca broker adapter (paper trading by default)
- A 3‑minute EMA pop‑pullback‑hold options strategy (calls + puts)
- A clean interface to plug in Grok sentiment & X ingestion once you add credentials

## Folder structure
```
autonomous_trading_bot/
  src/tradebot/
    app.py                    # CLI entrypoints
    config.py                 # Env config
    core/
      events.py               # Event bus models
      logger.py               # Logging setup
      timeutils.py
    ingestion/
      rss.py                  # RSS feed ingestor (working)
      x_client.py             # X API v2 ingestor (skeleton)
    sentiment/
      base.py                 # Sentiment interface
      grok.py                 # Grok adapter skeleton (HTTP)
      local_rule.py           # Simple rule-based fallback (working)
    broker/
      alpaca_broker.py        # Alpaca adapter (working)
      models.py
    data/
      marketdata.py           # Market data helpers (bars, ema)
      store.py                # SQLite store
    strategy/
      base.py
      ema_sentiment.py        # Legacy example strategy
      pop_pullback_hold.py    # 3‑min EMA pop‑pullback‑hold options strategy
    risk/
      risk_manager.py         # Position sizing, limits, SL/TP helpers
  scripts/
    run_paper.sh
  .env.example
  requirements.txt
  pyproject.toml
  Dockerfile
  docker-compose.yml
```

## Quick start (paper trading)
### 1) Create a virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
Copy and edit:
```bash
cp .env.example .env
```
Then set:
- `ALPACA_API_KEY` and `ALPACA_API_SECRET`
- `ALPACA_BASE_URL` to the paper endpoint
- `USE_SENTIMENT=false` for pure technical mode (default)
- Optional (only when `USE_SENTIMENT=true`):
  - `X_BEARER_TOKEN` for X API
  - `GROK_API_KEY` + `GROK_BASE_URL` (if you expose Grok inference via an endpoint)

### 3) Run the bot
```bash
python -m tradebot.app run --paper
```

## How it works (high level)
1. **Ingestion (optional)**
   - If `USE_SENTIMENT=true`, RSS ingestor pulls items every N seconds and emits `NewsEvent`s.
   - If `USE_SENTIMENT=true`, X ingestor fetches recent tweets for configured handles and emits `SocialEvent`s.
2. **Sentiment (optional)**
   - If `USE_SENTIMENT=true`, events are routed through a `SentimentClient` implementation.
   - Default provided: `LocalRuleSentiment` (simple keyword scoring).
   - Plug-in provided: `GrokSentimentClient` (HTTP skeleton).
3. **Strategy**
   - The active strategy runs on **3‑minute candles** of the **underlying** and computes EMA9.
   - It trades continuation after a pop above/below EMA, a pullback to EMA, and a 1–2 candle hold.
   - Entry triggers on a break of the confirmation candle’s high/low, with partial profits and EMA exits.
4. **Risk**
   - Central `RiskManager` enforces max daily loss, max position size, and sets SL/TP brackets.
5. **Execution**
   - `AlpacaBroker` places bracket orders (where supported) and monitors open positions.

## Plugging in Grok sentiment
This project keeps Grok integration as an **adapter** (so you can swap models).
- If Grok has a direct API you can call: implement it in `src/tradebot/sentiment/grok.py`.
- If you’re using Grok through another service, point `GROK_BASE_URL` to it and implement the request shape.

## Plugging in X handles (deltaone, elonmusk, donaldtrump)
The X ingestor is in `src/tradebot/ingestion/x_client.py`. You’ll need:
- `X_BEARER_TOKEN` (X API v2)
- Decide whether you query by **user ID** or **username**
- Rate limit logic (Basic tier limits apply)

## Safety checks you should add before real money
- Paper trade 2–4 weeks with logging on.
- Add circuit breakers: spread filter, volatility filter, max trades/day, cooldown after loss.
- Add monitoring: Prometheus/Grafana or even simple alerts.
- Add robust backtesting (separate module) before enabling live.

## Disclaimer
This repository is a **starter scaffold**. You are responsible for compliance with broker terms, exchange rules, 
and all legal/regulatory requirements.


## Stocks + options only
This version is set up for **US stocks/ETFs + US equity/ETF options** via Alpaca. Alpaca supports options trading in paper by default and offers options trading levels for live accounts. (See Alpaca Options Trading docs.)

### Options flow in this repo
1) Detect an underlying signal (3‑min EMA pop → pullback → hold confirmation)
2) Select an option contract (nearest expiry within your DTE window + ATM-ish strike)
3) Place a single‑leg options order via Alpaca Trading API

## Strategy Configuration (Pop‑Pullback‑Hold)
These are configurable via env vars (see `src/tradebot/config.py`).

- `POP_PULLBACK_EMA_LENGTH` (default `9`)
- `POP_PULLBACK_HOLD_CANDLES_REQUIRED` (default `2`, allowed `1` or `2`)
- `POP_PULLBACK_STRENGTH_FILTER` (default `false`)
- `POP_PULLBACK_STOP_BUFFER` (default `0.0`)
- `POP_PULLBACK_STOP_BUFFER_MODE` (`price` or `percent`)
- `POP_PULLBACK_TARGET_PROFIT_PCT` (default `0.07`, allowed `0.06`–`0.08`)
- `POP_PULLBACK_ENTRY_TIMEOUT_CANDLES` (default `3`)
- `POP_PULLBACK_EMA_EXIT_MODE` (`close` or `intrabar`)
- `POP_PULLBACK_PROFIT_CALC_ON_UNDERLYING` (default `true`)

Additional options selection knobs:
- `OPTION_DTE_MAX`, `OPTION_STRIKE_TOLERANCE`, `OPTION_ORDER_QTY`, `OPTION_USE_DYNAMIC_QTY`
