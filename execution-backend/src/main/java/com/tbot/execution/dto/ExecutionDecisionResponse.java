package com.tbot.execution.dto;

import com.tbot.execution.domain.OrderStatus;
import java.math.BigDecimal;
import java.time.Instant;

public record ExecutionDecisionResponse(
        String signalId,
        OrderStatus status,
        String reason,
        String venueOrderId,
        BigDecimal executedNotional,
        BigDecimal slippageFee,
        Instant processedAt
) {
}
