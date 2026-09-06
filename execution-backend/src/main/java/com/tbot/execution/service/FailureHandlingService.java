package com.tbot.execution.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
public class FailureHandlingService {

    private final Map<String, FailureContext> failureContexts = new ConcurrentHashMap<>();
    private final ControlCommandService controlCommandService;

    public FailureHandlingService(ControlCommandService controlCommandService) {
        this.controlCommandService = controlCommandService;
    }

    public void handleNetworkTimeout(String component) {
        log.error("Network timeout detected in {}", component);
        failureContexts.put(component + "_network_timeout", new FailureContext(
            "NETWORK_TIMEOUT",
            component,
            "Network request did not complete within timeout",
            Instant.now(),
            FailureSeverity.HIGH
        ));

        if (shouldTriggerEmergencyStop("network")) {
            triggerEmergencyStop("Network timeout in " + component);
        }
    }

    public void handleModelFailure(String modelVersion, String error) {
        log.error("Model failure in {}: {}", modelVersion, error);
        failureContexts.put(modelVersion + "_model_failure", new FailureContext(
            "MODEL_FAILURE",
            modelVersion,
            error,
            Instant.now(),
            FailureSeverity.CRITICAL
        ));

        triggerEmergencyStop("Model failure: " + error);
    }

    public void handleDatabaseFailure(String error) {
        log.error("Database failure: {}", error);
        failureContexts.put("database_failure", new FailureContext(
            "DATABASE_FAILURE",
            "database",
            error,
            Instant.now(),
            FailureSeverity.CRITICAL
        ));

        triggerEmergencyStop("Database failure: " + error);
    }

    public void handleExchangeError(String exchange, String error) {
        log.error("Exchange error in {}: {}", exchange, error);
        failureContexts.put(exchange + "_error", new FailureContext(
            "EXCHANGE_ERROR",
            exchange,
            error,
            Instant.now(),
            FailureSeverity.HIGH
        ));

        if (shouldTriggerEmergencyStop("exchange")) {
            triggerEmergencyStop("Exchange error in " + exchange);
        }
    }

    public void handleDuplicateSignal(String signalId) {
        log.warn("Duplicate signal detected: {}", signalId);
        failureContexts.put("duplicate_" + signalId, new FailureContext(
            "DUPLICATE_SIGNAL",
            "signal_processing",
            "Signal " + signalId + " already processed",
            Instant.now(),
            FailureSeverity.LOW
        ));
    }

    public void handleStaleSentiment(String asset, long ageMs) {
        log.warn("Stale sentiment for {}: {}ms old", asset, ageMs);
        failureContexts.put("stale_sentiment_" + asset, new FailureContext(
            "STALE_SENTIMENT",
            asset,
            "Sentiment data " + ageMs + "ms old",
            Instant.now(),
            FailureSeverity.MEDIUM
        ));
    }

    public void handleInvalidModelResponse(String modelVersion) {
        log.error("Invalid response from model: {}", modelVersion);
        failureContexts.put("invalid_response_" + modelVersion, new FailureContext(
            "INVALID_MODEL_RESPONSE",
            modelVersion,
            "Model returned invalid decision structure",
            Instant.now(),
            FailureSeverity.HIGH
        ));
    }

    public void handlePartialExecution(String orderId, String reason) {
        log.warn("Partial execution of order {}: {}", orderId, reason);
        failureContexts.put("partial_exec_" + orderId, new FailureContext(
            "PARTIAL_EXECUTION",
            orderId,
            reason,
            Instant.now(),
            FailureSeverity.MEDIUM
        ));
    }

    public void handleRejectedOrder(String orderId, String reason) {
        log.warn("Order rejected: {} - {}", orderId, reason);
        failureContexts.put("rejected_" + orderId, new FailureContext(
            "REJECTED_ORDER",
            orderId,
            reason,
            Instant.now(),
            FailureSeverity.LOW
        ));
    }

    public void handleModelDrift(String modelVersion) {
        log.warn("Model drift detected in: {}", modelVersion);
        failureContexts.put("drift_" + modelVersion, new FailureContext(
            "MODEL_DRIFT",
            modelVersion,
            "Prediction distribution has changed significantly",
            Instant.now(),
            FailureSeverity.MEDIUM
        ));

        // Consider reducing trading or requiring human approval
    }

    public void triggerEmergencyStop(String reason) {
        log.error("EMERGENCY STOP TRIGGERED: {}", reason);
        
        // Step 1: Stop new orders
        controlCommandService.halt("Emergency stop: " + reason);
        
        // Step 2: Liquidate all positions (should be done by control service)
        // This is the fail-closed architecture
    }

    private boolean shouldTriggerEmergencyStop(String failureType) {
        // Count consecutive failures of same type
        long recentFailures = failureContexts.values().stream()
            .filter(ctx -> ctx.type.contains(failureType))
            .filter(ctx -> Instant.now().minusSeconds(3600).isBefore(ctx.timestamp))
            .count();

        return recentFailures >= 3; // 3 failures in 1 hour triggers stop
    }

    public Map<String, FailureContext> getRecentFailures(int lastMinutes) {
        Map<String, FailureContext> recent = new HashMap<>();
        Instant cutoff = Instant.now().minusSeconds(lastMinutes * 60L);

        for (Map.Entry<String, FailureContext> entry : failureContexts.entrySet()) {
            if (entry.getValue().timestamp.isAfter(cutoff)) {
                recent.put(entry.getKey(), entry.getValue());
            }
        }

        return recent;
    }

    public void clearOldFailures(int olderThanMinutes) {
        Instant cutoff = Instant.now().minusSeconds(olderThanMinutes * 60L);
        failureContexts.entrySet().removeIf(entry -> entry.getValue().timestamp.isBefore(cutoff));
    }

    public static class FailureContext {
        public final String type;
        public final String component;
        public final String error;
        public final Instant timestamp;
        public final FailureSeverity severity;
        public int retries = 0;

        public FailureContext(String type, String component, String error, Instant timestamp, FailureSeverity severity) {
            this.type = type;
            this.component = component;
            this.error = error;
            this.timestamp = timestamp;
            this.severity = severity;
        }

        public boolean canRetry() {
            return retries < 3;
        }

        public void recordRetry() {
            retries++;
        }
    }

    public enum FailureSeverity {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }
}
