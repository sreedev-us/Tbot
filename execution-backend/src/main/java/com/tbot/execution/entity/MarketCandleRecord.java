package com.tbot.execution.entity;

import com.tbot.execution.domain.ExchangeName;
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
@Table(name = "market_candles")
public class MarketCandleRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 24)
    private String asset;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private ExchangeName exchangeName;

    @Column(nullable = false, length = 8)
    private String timeframe;

    @Column(nullable = false)
    private Instant candleTime;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal openPrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal highPrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal lowPrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal closePrice;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal volume;
}
