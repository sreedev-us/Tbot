package com.tbot.execution.entity;

import com.tbot.execution.domain.ExecutionStatus;
import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.Instant;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "executions")
public class ExecutionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "order_id", nullable = false)
    private OrderRecord order;

    @Column(nullable = false, length = 32)
    private String venueOrderId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private ExchangeName exchangeName;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 8)
    private OrderAction action;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private ExecutionStatus status;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal requestedNotional;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal executedNotional;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal slippageFee;

    @Column(nullable = false)
    private Instant exchangeAckAt;

    @Column(nullable = false)
    private Instant fillConfirmedAt;
}
