# tbot

Trading bot scaffold with a Spring Boot execution backend, a Python alpha engine, and a Flutter telemetry dashboard.

## Structure

- `execution-backend/`: Spring Boot risk, execution orchestration, control commands, and dashboard APIs.
- `alpha-engine/`: Python CCXT signal engine that honors backend halt state before submitting signals.
- `telemetry_dashboard/`: Flutter observer app using Supabase realtime plus backend override APIs.
- `supabase/realtime_setup.sql`: SQL to expose the ledger tables to Supabase realtime.
- `docker-compose.yml`: local PostgreSQL for development.

## Implemented now

- PostgreSQL-backed ACID persistence with Flyway migrations for `orders`, `executions`, `system_telemetry`, and `control_commands`.
- HikariCP-style datasource pooling via Spring Boot datasource config.
- `POST /api/v1/execute` for signal intake with idempotency and risk gating.
- `GET /api/v1/dashboard/snapshot` for telemetry aggregates.
- `GET /api/v1/control/state` and `POST /api/v1/control/commands` for halt, resume, and liquidate-all commands.
- Flutter dashboard cards, live Supabase table streams after admin sign-in, and a manual override terminal that routes only to Spring Boot.

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
