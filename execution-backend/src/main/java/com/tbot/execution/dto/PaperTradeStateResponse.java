package com.tbot.execution.dto;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import java.math.BigDecimal;
import java.time.Instant;

public record PaperTradeStateResponse(
        Long tradeId,
        String signalId,
        String asset,
        ExchangeName exchange,
        OrderAction action,
        TradeStatus status,
        TradeOutcome outcome,
        BigDecimal entryPrice,
        BigDecimal stopLossPrice,
        BigDecimal takeProfitPrice,
        BigDecimal quantity,
        Instant openedAt
) {
}
