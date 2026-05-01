from __future__ import annotations

import json
from abc import ABC, abstractmethod

import requests

from sentiment_oracle.models import FeedEvent, SentimentResult


class LocalSentimentModel(ABC):

    @abstractmethod
    def evaluate(self, event: FeedEvent) -> SentimentResult:
        raise NotImplementedError


class HeuristicFinancialModel(LocalSentimentModel):

    POSITIVE_TERMS = {
        "approval": 0.45,
        "approved": 0.45,
        "bullish": 0.60,
        "easing": 0.35,
        "cooling inflation": 0.30,
        "rate cut": 0.40,
        "etf inflow": 0.40,
        "support": 0.25,
        "breakout": 0.40,
        "surge": 0.30,
    }
    NEGATIVE_TERMS = {
        "lawsuit": -0.55,
        "rejection": -0.45,
        "bearish": -0.60,
        "inflation spike": -0.40,
        "rate hike": -0.45,
        "liquidation": -0.50,
        "ban": -0.60,
        "hack": -0.70,
        "probe": -0.35,
        "selloff": -0.45,
    }

    def evaluate(self, event: FeedEvent) -> SentimentResult:
        text = event.text.lower()
        score = 0.0
        hits = 0

        for term, weight in self.POSITIVE_TERMS.items():
            if term in text:
                score += weight
                hits += 1

        for term, weight in self.NEGATIVE_TERMS.items():
            if term in text:
                score += weight
                hits += 1

        clipped_score = max(min(score, 1.0), -1.0)
        confidence = min(0.35 + hits * 0.15, 0.95) if hits > 0 else 0.25
        rationale = f"heuristic hits={hits}"
        return SentimentResult(score=clipped_score, confidence=confidence, rationale=rationale)


class OllamaFinancialModel(LocalSentimentModel):

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def evaluate(self, event: FeedEvent) -> SentimentResult:
        prompt = (
            "You are a financial sentiment classifier. "
            "Return strict JSON with keys score, confidence, rationale. "
            "Score must be between -1.0 and 1.0. Confidence must be between 0.0 and 1.0. "
            "Focus on short-term market impact for crypto traders.\n\n"
            f"Headline: {event.headline}\n"
            f"Body: {event.body}\n"
            f"Tags: {', '.join(event.tags)}"
        )
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        payload = json.loads(body["response"])
        return SentimentResult(
            score=max(min(float(payload["score"]), 1.0), -1.0),
            confidence=max(min(float(payload["confidence"]), 1.0), 0.0),
            rationale=str(payload.get("rationale", "ollama")),
        )


def build_model(*, backend: str, ollama_url: str, ollama_model: str) -> LocalSentimentModel:
    if backend == "ollama":
        return OllamaFinancialModel(base_url=ollama_url, model=ollama_model)
    return HeuristicFinancialModel()
