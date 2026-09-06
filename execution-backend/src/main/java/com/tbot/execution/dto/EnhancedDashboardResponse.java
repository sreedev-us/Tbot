package com.tbot.execution.dto;

import java.math.BigDecimal;
import java.util.Map;

public record EnhancedDashboardResponse(
    DashboardTradeInfo trades,
    AISystemInfo aiSystem,
    RiskSystemInfo riskSystem,
    PerformanceMetricsInfo performance,
    LatencyMetricsInfo latency,
    SystemHealthInfo systemHealth
) {
    
    public record DashboardTradeInfo(
        BigDecimal currentExposure,
        int openTradesCount,
        java.util.List<TradeSnapshot> recentTrades,
        BigDecimal dailyPnL,
        BigDecimal totalPnL
    ) {}

    public record TradeSnapshot(
        long tradeId,
        String asset,
        BigDecimal entryPrice,
        BigDecimal currentPrice,
        BigDecimal pnlPct,
        String status
    ) {}

    public record AISystemInfo(
        String activeModel,
        String modelHealth,
        String lastDecision,
        BigDecimal lastConfidence,
        int decisionsIn1Hour,
        double avgConfidence,
        String driftStatus
    ) {}

    public record RiskSystemInfo(
        String status,
        BigDecimal maxExposure,
        BigDecimal currentExposure,
        BigDecimal maxDrawdown,
        BigDecimal currentDrawdown,
        long rejectionCount,
        double rejectionRate
    ) {}

    public record PerformanceMetricsInfo(
        double winRate,
        double profitFactor,
        double sharpeRatio,
        double maxDrawdown,
        int totalTrades,
        BigDecimal avgTradeSize,
        String mostRecentOutcome
    ) {}

    public record LatencyMetricsInfo(
        long avgTotalLatencyMs,
        long maxTotalLatencyMs,
        long avgServerAILatencyMs,
        long avgLocalAILatencyMs,
        long avgRiskLatencyMs,
        long avgExecutionLatencyMs
    ) {}

    public record SystemHealthInfo(
        boolean engineRunning,
        boolean connectionHealthy,
        String lastMarketDataTime,
        String lastAIUpdateTime,
        java.util.List<SystemError> recentErrors
    ) {}

    public record SystemError(
        String type,
        String message,
        String timestamp
    ) {}
}
