package com.tbot.execution.dto;

import com.tbot.execution.domain.ExchangeName;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.Instant;

public record PaperTradeCloseRequest(
        @NotBlank String asset,
        @NotNull ExchangeName exchange,
        @NotNull @DecimalMin("0.0") BigDecimal exitPrice,
        @NotNull Instant closedAt,
        @NotBlank String closeReason
) {
}
