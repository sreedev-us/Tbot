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
@Table(name = "system_telemetry")
public class SystemTelemetry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 32)
    private String metricGroup;

    @Column(nullable = false, length = 64)
    private String metricName;

    @Column(nullable = false, precision = 18, scale = 8)
    private BigDecimal metricValue;

    @Column(nullable = false, length = 16)
    private String metricUnit;

    @Column(nullable = false, length = 256)
    private String tags;

    @Column(nullable = false)
    private Instant recordedAt;
}
