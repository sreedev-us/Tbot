package com.tbot.execution.dto;

import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import java.math.BigDecimal;
import java.time.Instant;

public record TradeMarkerResponse(
        Long tradeId,
        String signalId,
        OrderAction action,
        TradeStatus status,
        TradeOutcome outcome,
        Instant openedAt,
        Instant closedAt,
        BigDecimal entryPrice,
        BigDecimal exitPrice,
        BigDecimal realizedPnl,
        BigDecimal realizedPnlPct,
        String closeReason
) {
}
