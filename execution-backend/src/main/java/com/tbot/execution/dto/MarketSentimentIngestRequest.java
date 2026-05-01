package com.tbot.execution.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.Instant;

public record MarketSentimentIngestRequest(
        @NotBlank String asset,
        @NotBlank String source,
        @NotBlank String headline,
        @NotNull @DecimalMin("-1.0") @DecimalMax("1.0") BigDecimal sentimentScore,
        @NotNull @DecimalMin("0.0") @DecimalMax("1.0") BigDecimal confidence,
        @NotBlank String tags,
        @NotNull Instant observedAt
) {
}
