package com.tbot.execution.dto;

import java.util.Map;

public record LatencyBreakdownResponse(
    long marketDataReceivedMs,
    long featuresCalculatedMs,
    long serverAIStartedMs,
    long serverAICompletedMs,
    long localAIStartedMs,
    long localAICompletedMs,
    long riskValidationStartedMs,
    long riskValidationCompletedMs,
    long orderSentMs,
    long exchangeAcknowledgedMs,
    long orderFilledMs,
    Map<String, Long> componentLatencies
) {}
