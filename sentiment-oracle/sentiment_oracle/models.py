from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FeedEvent:
    source: str
    headline: str
    body: str
    observed_at: str
    tags: tuple[str, ...]

    @property
    def text(self) -> str:
        return f"{self.headline}\n{self.body}".strip()


@dataclass(slots=True)
class SentimentResult:
    score: float
    confidence: float
    rationale: str
