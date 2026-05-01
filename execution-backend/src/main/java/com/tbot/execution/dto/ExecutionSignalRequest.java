package com.tbot.execution.dto;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.Instant;

public record ExecutionSignalRequest(
        @NotBlank String signalId,
        @NotBlank String correlationId,
        @NotBlank String asset,
        @NotNull ExchangeName exchange,
        @NotNull OrderAction action,
        @NotNull @DecimalMin("0.0") @DecimalMax("1.0") BigDecimal confidence,
        @NotNull @DecimalMin("0.00000001") BigDecimal requestedNotional,
        @NotBlank String strategyName,
        @NotNull Instant generatedAt,
        @NotNull @DecimalMin("0.00000001") BigDecimal marketPrice,
        @NotNull @DecimalMin("0.00000001") BigDecimal stopLossPrice,
        @NotNull @DecimalMin("0.00000001") BigDecimal takeProfitPrice
) {
}
