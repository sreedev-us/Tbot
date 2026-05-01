package com.tbot.execution.dto;

import com.tbot.execution.domain.ExchangeName;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.util.List;

public record MarketCandleSyncRequest(
        @NotBlank String asset,
        @NotNull ExchangeName exchange,
        @NotBlank String timeframe,
        @NotEmpty List<@Valid CandlePointRequest> candles
) {
}
