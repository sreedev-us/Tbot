from __future__ import annotations

import hashlib
import time

from sentiment_oracle.backend_client import publish_sentiment
from sentiment_oracle.config import load_config
from sentiment_oracle.feed import load_feed_events
from sentiment_oracle.filters import matches_entities
from sentiment_oracle.local_models import build_model


def run() -> None:
    config = load_config()
    model = build_model(
        backend=config.model_backend,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    seen_event_ids: set[str] = set()

    print(
        f"Sentiment Oracle started for {config.asset} "
        f"(model={config.model_backend}, feed={config.feed_file})"
    )

    while True:
        for event in load_feed_events(config.feed_file):
            fingerprint = hashlib.sha256(
                f"{event.source}|{event.headline}|{event.observed_at}".encode("utf-8")
            ).hexdigest()
            if fingerprint in seen_event_ids:
                continue
            seen_event_ids.add(fingerprint)

            if not matches_entities(event, config.watched_entities):
                continue

            result = model.evaluate(event)
            if result.confidence < config.min_confidence:
                print(
                    f"Skipped low-confidence headline from {event.source}: "
                    f"score={result.score:.2f} confidence={result.confidence:.2f}"
                )
                continue

            publish_sentiment(
                config.backend_base_url,
                asset=config.asset,
                event=event,
                result=result,
            )
            print(
                f"Published sentiment {result.score:.2f} "
                f"(confidence={result.confidence:.2f}) from {event.source}: {event.headline}"
            )

        time.sleep(config.poll_interval_seconds)
