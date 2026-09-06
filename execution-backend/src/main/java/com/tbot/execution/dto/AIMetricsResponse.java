package com.tbot.execution.dto;

import java.math.BigDecimal;
import java.util.Map;

public record AIMetricsResponse(
    String modelVersion,
    String modelHash,
    String featureVersion,
    String localAIDecision,
    BigDecimal localAIConfidence,
    String serverAIAnalysis,
    BigDecimal serverAIConfidence,
    long serverAILatencyMs,
    long localAILatencyMs,
    long totalAILatencyMs,
    String riskDecision,
    String riskReason,
    Map<String, Object> detailedMetrics
) {}
