# Sentiment Oracle

Independent microservice that filters a low-latency headline stream, scores it locally, and publishes normalized market sentiment into the Spring Boot backend.

## Current shape

- Prefilters only watched entities such as `BTC`, `SEC`, `Federal Reserve`, and `Inflation`.
- Supports a local heuristic financial model out of the box.
- Supports local Ollama inference by setting `TBOT_SENTIMENT_MODEL_BACKEND=ollama`.
- Publishes accepted sentiment events to `POST /api/v1/sentiment`.

## Run

```powershell
cd sentiment-oracle
python -m pip install -r requirements.txt
python main.py
```

## Feed input

The default feed source is a newline-delimited JSON file:

`sentiment-oracle/feed/news_feed.jsonl`

Each line should look like:

```json
{
  "source": "financialjuice",
  "headline": "Federal Reserve signals easing bias as inflation cools",
  "body": "Risk assets including BTC react positively to softer macro expectations.",
  "observedAt": "2026-04-25T08:00:00Z",
  "tags": ["BTC", "Federal Reserve", "Inflation"]
}
```

This is the local integration seam for a real Benzinga / FinancialJuice / X filtered stream adapter later.

## Environment

- `TBOT_BACKEND_URL`
- `TBOT_SENTIMENT_FEED_FILE`
- `TBOT_SENTIMENT_POLL_SECONDS`
- `TBOT_SENTIMENT_WATCHED_ENTITIES`
- `TBOT_SENTIMENT_MODEL_BACKEND`
- `TBOT_SENTIMENT_OLLAMA_MODEL`
- `TBOT_SENTIMENT_OLLAMA_URL`
- `TBOT_SENTIMENT_ASSET`
- `TBOT_SENTIMENT_MIN_CONFIDENCE`
