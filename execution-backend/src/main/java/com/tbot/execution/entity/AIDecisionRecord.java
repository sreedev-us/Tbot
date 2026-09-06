package com.tbot.execution.entity;

import com.tbot.execution.domain.AIDecisionType;
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
@Table(name = "ai_decisions")
public class AIDecisionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "order_id", nullable = false)
    private OrderRecord order;

    @Column(nullable = false, length = 64)
    private String modelVersion;

    @Column(nullable = false, length = 64)
    private String modelHash;

    @Column(nullable = false, length = 32)
    private String featureVersion;

    @Column(nullable = false)
    private Instant inputTimestamp;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 8)
    private AIDecisionType decision;

    @Column(nullable = false, precision = 10, scale = 6)
    private BigDecimal confidence;

    @Column(type = "jsonb", nullable = false)
    private String featuresJson;

    @Column(type = "jsonb")
    private String serverAIResultJson;

    @Column(type = "jsonb", nullable = false)
    private String localAIResultJson;

    @Column(type = "jsonb")
    private String riskDecisionJson;

    @Column(nullable = false)
    private Instant decidedAt;

    @Column(nullable = true)
    private Instant executedAt;

    @Column(length = 256)
    private String rejectReason;
}
