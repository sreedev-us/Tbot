# tbot

Trading bot scaffold with a Spring Boot execution backend, a Python alpha engine, and a Flutter telemetry dashboard.

## Structure

- `execution-backend/`: Spring Boot risk, execution orchestration, control commands, and dashboard APIs.
- `alpha-engine/`: Python CCXT signal engine that honors backend halt state before submitting signals.
- `sentiment-oracle/`: Independent Python sentiment microservice that filters news events, scores them locally, and posts risk modifiers into Spring Boot.
- `telemetry_dashboard/`: Flutter observer app using Supabase realtime plus backend override APIs.
- `supabase/realtime_setup.sql`: SQL to expose the ledger tables to Supabase realtime.
- `docker-compose.yml`: local PostgreSQL for development.

## Implemented now

- PostgreSQL-backed ACID persistence with Flyway migrations for `orders`, `executions`, `system_telemetry`, and `control_commands`.
- HikariCP-style datasource pooling via Spring Boot datasource config.
- `POST /api/v1/execute` for signal intake with idempotency and risk gating.
- `POST /api/v1/sentiment` and `GET /api/v1/sentiment?asset=...` for the sentiment oracle state channel.
- `GET /api/v1/dashboard/snapshot` for telemetry aggregates.
- `GET /api/v1/dashboard/market-chart` for the demo candle graph with trade entry and exit markers.
- `GET /api/v1/dashboard/trades` for detailed paper-trade reports with profit or loss classification.
- `POST /api/v1/market-data/candles` and `/api/v1/paper-trades/close` so the Python engine can keep the demo ledger and chart up to date.
- `GET /api/v1/control/state` and `POST /api/v1/control/commands` for halt, resume, and liquidate-all commands.
- Flutter dashboard cards, a live demo graph, a profit/loss trade journal, and a manual override terminal that routes only to Spring Boot.
- A sentiment-aware risk layer that:
  rejects `BUY` signals when fresh severe negative sentiment is active,
  and boosts aligned bullish `BUY` notional by a configurable multiplier.

## Local run

1. Copy `.env.example` to `.env`.
2. For local Postgres development:
   `docker compose up -d`
3. For Supabase-backed development:
   set `TBOT_DB_URL`, `TBOT_DB_USERNAME`, and `TBOT_DB_PASSWORD` to your Supabase pooler JDBC values.
   The placeholders `SUPABASE_DB_*` in `.env.example` show the expected format.
4. Start the backend:
   `cd execution-backend`
   `.\gradlew.bat bootRun`
5. Start the alpha engine:
   `cd alpha-engine`
   `python -m pip install -r requirements.txt`
   `python main.py`
6. Start the Flutter dashboard:
   `cd telemetry_dashboard`
   `flutter pub get`
   `flutter run --dart-define=TBOT_BACKEND_URL=http://localhost:8080 --dart-define=SUPABASE_URL=https://your-project.supabase.co --dart-define=SUPABASE_ANON_KEY=your-public-anon-key`
7. Optional: start the sentiment oracle:
   `cd sentiment-oracle`
   `python -m pip install -r requirements.txt`
   `python main.py`

## Demo paper trading

- The current stack can execute demo trades end to end without touching a live exchange account.
- The backend still simulates the execution hop, while the Python engine now:
  syncs candles into Spring Boot,
  opens a paper trade on accepted signals,
  and closes it on stop loss or take profit.
- The dashboard shows:
  a live candle graph,
  trade start and end points,
  and a detailed history with realized profit or loss.

## One-command local startup

- Start everything:
  `.\start.ps1`
- Stop everything:
  `.\stop.ps1`
- To include the sentiment oracle in the one-command startup, set `TBOT_ENABLE_SENTIMENT_ORACLE=true` in `.env`.

For local testing, the startup script builds the Flutter app with `--pwa-strategy=none` and serves the built web output on `http://localhost:3000`. This avoids the loading-bar issue you can get from a stale service worker or a detached `flutter run` web session that never finishes bootstrapping.

## Strategy Optimization

- The Python side now includes an Optuna-based Bayesian optimization scaffold for mean reversion.
- Run it with:
  `cd alpha-engine`
  `python -m pip install -r requirements.txt`
  `python optimize.py path\\to\\ohlcv.csv --trials 300 --storage sqlite:///optuna.db --return-weight 0.5 --sharpe-weight 0.3 --drawdown-weight 0.2`
- The optimizer enforces the hard `take_profit_pct <= stop_loss_pct * 2` constraint, which keeps the search inside your 1:2 risk/reward boundary.
- Its objective is now a weighted combination of:
  total return,
  Sharpe ratio,
  and max drawdown penalty.

## Sentiment Oracle

- The oracle is an independent microservice and does not place trades.
- Its role is to publish fresh sentiment state that the backend risk module uses as a circuit breaker or size multiplier.
- The default local model is a lightweight heuristic financial scorer so the service runs immediately.
- If you have Ollama running locally, set `TBOT_SENTIMENT_MODEL_BACKEND=ollama` to use local LLM inference instead.
- The included `feed/news_feed.jsonl` file is the development seam where a real Benzinga / FinancialJuice / filtered X adapter can plug in later.

## Supabase setup

1. Open the Supabase SQL editor for project `wmpkqdbftotydbmgagxw`.
2. Replace `YOUR_ADMIN_UUID` in [realtime_setup.sql](C:/Users/Sreedev/AndroidStudioProjects/tbot/supabase/realtime_setup.sql) with the Supabase Auth user ID for your admin account.
3. Run [realtime_setup.sql](C:/Users/Sreedev/AndroidStudioProjects/tbot/supabase/realtime_setup.sql).
4. Use the Supabase pooler connection string for Spring Boot JDBC, not the REST or GraphQL APIs.
5. The Flutter dashboard now requires an authenticated Supabase admin session before it can subscribe to realtime streams.
6. The anon key is still public by design, but backend writes must stay on trusted server credentials only.

## Notes

- Sandbox mode is enabled for Binance and Bybit when `TBOT_USE_SANDBOX=true`.
- Kraken does not expose a CCXT sandbox, so keep its credentials empty during paper trading.
- The backend still simulates the final exchange hop; the dashboard and control plane are already wired to the production-shaped contracts.
