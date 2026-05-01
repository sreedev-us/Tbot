package com.tbot.execution.dto;

import com.tbot.execution.domain.OrderStatus;
import java.math.BigDecimal;
import java.time.Instant;

public record ExecutionDecisionResponse(
        String signalId,
        OrderStatus status,
        String reason,
        String venueOrderId,
        BigDecimal approvedNotional,
        BigDecimal executedNotional,
        BigDecimal slippageFee,
        BigDecimal sentimentScore,
        BigDecimal sentimentConfidence,
        Instant processedAt
) {
}
