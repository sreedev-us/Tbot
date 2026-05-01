package com.tbot.execution.dto;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import java.math.BigDecimal;
import java.time.Instant;

public record PaperTradeReportResponse(
        Long tradeId,
        String signalId,
        String asset,
        ExchangeName exchange,
        OrderAction action,
        String strategyName,
        BigDecimal confidence,
        BigDecimal requestedNotional,
        BigDecimal quantity,
        BigDecimal entryPrice,
        BigDecimal exitPrice,
        BigDecimal stopLossPrice,
        BigDecimal takeProfitPrice,
        Instant openedAt,
        Instant closedAt,
        TradeStatus status,
        TradeOutcome outcome,
        BigDecimal realizedPnl,
        BigDecimal realizedPnlPct,
        String closeReason
) {
}
