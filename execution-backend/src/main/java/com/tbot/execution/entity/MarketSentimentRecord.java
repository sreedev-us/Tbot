package com.tbot.execution.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
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
@Table(name = "market_sentiment_records")
public class MarketSentimentRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 24)
    private String asset;

    @Column(nullable = false, precision = 6, scale = 4)
    private BigDecimal sentimentScore;

    @Column(nullable = false, precision = 6, scale = 4)
    private BigDecimal confidence;

    @Column(nullable = false, length = 64)
    private String source;

    @Column(nullable = false, length = 512)
    private String headline;

    @Column(nullable = false, length = 256)
    private String tags;

    @Column(nullable = false)
    private Instant observedAt;

    @Column(nullable = false)
    private Instant recordedAt;
}
