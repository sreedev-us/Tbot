from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class OracleConfig:
    backend_base_url: str
    feed_file: str
    poll_interval_seconds: int
    watched_entities: tuple[str, ...]
    model_backend: str
    ollama_model: str
    ollama_url: str
    asset: str
    min_confidence: float


def load_config() -> OracleConfig:
    load_dotenv()
    watched = os.getenv("TBOT_SENTIMENT_WATCHED_ENTITIES", "BTC,SEC,Federal Reserve,Inflation")
    return OracleConfig(
        backend_base_url=os.getenv("TBOT_BACKEND_URL", "http://localhost:8080"),
        feed_file=os.getenv("TBOT_SENTIMENT_FEED_FILE", "sentiment-oracle/feed/news_feed.jsonl"),
        poll_interval_seconds=int(os.getenv("TBOT_SENTIMENT_POLL_SECONDS", "2")),
        watched_entities=tuple(token.strip() for token in watched.split(",") if token.strip()),
        model_backend=os.getenv("TBOT_SENTIMENT_MODEL_BACKEND", "heuristic").strip().lower(),
        ollama_model=os.getenv("TBOT_SENTIMENT_OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
        ollama_url=os.getenv("TBOT_SENTIMENT_OLLAMA_URL", "http://localhost:11434"),
        asset=os.getenv("TBOT_SENTIMENT_ASSET", "BTC/USDT"),
        min_confidence=float(os.getenv("TBOT_SENTIMENT_MIN_CONFIDENCE", "0.55")),
    )
