package com.tbot.execution.dto;

import java.math.BigDecimal;
import java.time.Instant;

public record MarketSentimentStateResponse(
        String asset,
        BigDecimal sentimentScore,
        BigDecimal confidence,
        String source,
        String headline,
        String tags,
        Instant observedAt
) {
}
