package com.tbot.execution.service;

import java.math.BigDecimal;

public record RiskDecision(
        boolean approved,
        String reason,
        BigDecimal approvedNotional,
        BigDecimal sentimentScore,
        BigDecimal sentimentConfidence
) {

    public static RiskDecision approve(BigDecimal approvedNotional, BigDecimal sentimentScore, BigDecimal sentimentConfidence) {
        return new RiskDecision(true, "APPROVED", approvedNotional, sentimentScore, sentimentConfidence);
    }

    public static RiskDecision reject(String reason, BigDecimal approvedNotional, BigDecimal sentimentScore, BigDecimal sentimentConfidence) {
        return new RiskDecision(false, reason, approvedNotional, sentimentScore, sentimentConfidence);
    }
}
