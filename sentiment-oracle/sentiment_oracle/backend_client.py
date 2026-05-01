from __future__ import annotations

import requests

from sentiment_oracle.models import FeedEvent, SentimentResult


def publish_sentiment(base_url: str, *, asset: str, event: FeedEvent, result: SentimentResult) -> None:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/v1/sentiment",
        json={
            "asset": asset,
            "source": event.source,
            "headline": event.headline,
            "sentimentScore": result.score,
            "confidence": result.confidence,
            "tags": ",".join(event.tags),
            "observedAt": event.observed_at,
        },
        timeout=5,
    )
    response.raise_for_status()
