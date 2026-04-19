package com.tbot.execution.dto;

import java.math.BigDecimal;
import java.time.Instant;

public record DashboardSnapshotResponse(
        long totalOrders,
        long executedOrders,
        long rejectedOrders,
        BigDecimal currentExposure,
        BigDecimal totalExecutedNotional,
        BigDecimal averageExecutionLatencyMs,
        boolean engineHalted,
        boolean liquidationRequested,
        Instant generatedAt
) {
}
