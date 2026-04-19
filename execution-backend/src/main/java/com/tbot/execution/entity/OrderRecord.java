package com.tbot.execution.entity;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.OrderStatus;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.Instant;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "orders")
public class OrderRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 64)
    private String signalId;

    @Column(nullable = false, length = 32)
    private String correlationId;

    @Column(nullable = false, length = 24)
    private String asset;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private ExchangeName exchangeName;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 8)
    private OrderAction action;

    @Column(nullable = false, precision = 10, scale = 6)
    private BigDecimal confidence;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal requestedNotional;

    @Column(nullable = false, length = 64)
    private String strategyName;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private OrderStatus status;

    @Column(nullable = false)
    private Instant generatedAt;

    @Column(nullable = false)
    private Instant receivedAt;

    private Instant routedAt;

    private Instant completedAt;

    @Column(length = 256)
    private String rejectionReason;
}
