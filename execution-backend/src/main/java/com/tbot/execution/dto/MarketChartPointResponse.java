package com.tbot.execution.dto;

import java.math.BigDecimal;
import java.time.Instant;

public record MarketChartPointResponse(
        Instant timestamp,
        BigDecimal open,
        BigDecimal high,
        BigDecimal low,
        BigDecimal close,
        BigDecimal volume
) {
}
