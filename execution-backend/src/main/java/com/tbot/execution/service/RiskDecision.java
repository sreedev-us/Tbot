package com.tbot.execution.service;

public record RiskDecision(boolean approved, String reason) {

    public static RiskDecision approve() {
        return new RiskDecision(true, "APPROVED");
    }

    public static RiskDecision reject(String reason) {
        return new RiskDecision(false, reason);
    }
}
