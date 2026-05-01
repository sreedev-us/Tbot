from __future__ import annotations

from sentiment_oracle.models import FeedEvent


def matches_entities(event: FeedEvent, watched_entities: tuple[str, ...]) -> bool:
    haystack = " ".join([event.headline, event.body, *event.tags]).lower()
    return any(entity.lower() in haystack for entity in watched_entities)
