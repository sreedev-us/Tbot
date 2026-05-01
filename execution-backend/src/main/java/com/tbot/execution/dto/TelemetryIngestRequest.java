package com.tbot.execution.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;

public record TelemetryIngestRequest(
        @NotBlank String metricGroup,
        @NotBlank String metricName,
        @NotNull @DecimalMin("0.0") BigDecimal metricValue,
        @NotBlank String metricUnit,
        @NotBlank String tags
) {
}
