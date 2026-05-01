package com.tbot.execution.entity;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.Instant;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "paper_trades")
public class PaperTradeRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "order_id", nullable = false, unique = true)
    private OrderRecord order;

    @Column(nullable = false, length = 24)
    private String asset;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private ExchangeName exchangeName;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 8)
    private OrderAction action;

    @Column(nullable = false, length = 64)
    private String strategyName;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal entryPrice;

    @Column(precision = 18, scale = 8)
    private BigDecimal exitPrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal quantity;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal requestedNotional;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal stopLossPrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal takeProfitPrice;

    @Column(nullable = false, precision = 10, scale = 6)
    private BigDecimal confidence;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private TradeStatus status;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private TradeOutcome outcome;

    @Column(nullable = false)
    private Instant openedAt;

    private Instant closedAt;

    @Column(precision = 18, scale = 8)
    private BigDecimal realizedPnl;

    @Column(precision = 10, scale = 4)
    private BigDecimal realizedPnlPct;

    @Column(length = 32)
    private String closeReason;
}
