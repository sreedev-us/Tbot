from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sentiment_oracle.models import FeedEvent


def load_feed_events(feed_file: str) -> Iterable[FeedEvent]:
    path = Path(feed_file)
    if not path.exists():
        return []

    events: list[FeedEvent] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        events.append(
            FeedEvent(
                source=str(payload.get("source", "unknown")),
                headline=str(payload.get("headline", "")),
                body=str(payload.get("body", "")),
                observed_at=str(payload.get("observedAt", payload.get("observed_at"))),
                tags=tuple(str(tag) for tag in payload.get("tags", [])),
            )
        )
    return events
