package com.tbot.execution.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.Instant;

public record CandlePointRequest(
        @NotNull Instant timestamp,
        @NotNull @DecimalMin("0.0") BigDecimal open,
        @NotNull @DecimalMin("0.0") BigDecimal high,
        @NotNull @DecimalMin("0.0") BigDecimal low,
        @NotNull @DecimalMin("0.0") BigDecimal close,
        @NotNull @DecimalMin("0.0") BigDecimal volume
) {
}
