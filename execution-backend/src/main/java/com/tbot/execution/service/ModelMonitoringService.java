package com.tbot.execution.service;

import com.tbot.execution.entity.AIDecisionRecord;
import com.tbot.execution.repository.AIDecisionRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
public class ModelMonitoringService {

    private final AIDecisionRepository aiDecisionRepository;

    public ModelMonitoringService(AIDecisionRepository aiDecisionRepository) {
        this.aiDecisionRepository = aiDecisionRepository;
    }

    public ModelHealthMetrics getModelHealth(String modelVersion, int windowMinutes) {
        Instant since = Instant.now().minusSeconds(windowMinutes * 60L);
        List<AIDecisionRecord> recentDecisions = aiDecisionRepository.findByDecisionDateRange(
            since,
            Instant.now()
        ).stream()
            .filter(d -> d.getModelVersion().equals(modelVersion))
            .collect(Collectors.toList());

        if (recentDecisions.isEmpty()) {
            return new ModelHealthMetrics(modelVersion, "NO_DATA", 0, 0, 0, Map.of());
        }

        // Calculate metrics
        long rejectionCount = recentDecisions.stream()
            .filter(d -> d.getRejectReason() != null)
            .count();

        double rejectionRate = (double) rejectionCount / recentDecisions.size();

        // Confidence distribution
        Map<String, Integer> confidenceDistribution = new HashMap<>();
        for (AIDecisionRecord decision : recentDecisions) {
            String bucket = getConfidenceBucket(decision.getConfidence());
            confidenceDistribution.merge(bucket, 1, Integer::sum);
        }

        // Average confidence
        BigDecimal avgConfidence = recentDecisions.stream()
            .map(AIDecisionRecord::getConfidence)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(recentDecisions.size()), 6, java.math.RoundingMode.HALF_UP);

        String health = determineHealth(rejectionRate, avgConfidence);

        return new ModelHealthMetrics(
            modelVersion,
            health,
            recentDecisions.size(),
            rejectionCount,
            avgConfidence.doubleValue(),
            confidenceDistribution
        );
    }

    public ModelDriftMetrics detectDrift(String modelVersion, int comparisonWindowMinutes) {
        Instant since = Instant.now().minusSeconds(comparisonWindowMinutes * 60L);
        List<AIDecisionRecord> decisions = aiDecisionRepository.findByDecisionDateRange(since, Instant.now())
            .stream()
            .filter(d -> d.getModelVersion().equals(modelVersion))
            .collect(Collectors.toList());

        if (decisions.size() < 10) {
            return new ModelDriftMetrics(modelVersion, "INSUFFICIENT_DATA", Map.of());
        }

        // Prediction distribution
        Map<String, Integer> predictionDistribution = new HashMap<>();
        for (AIDecisionRecord decision : decisions) {
            String decision_type = decision.getDecision().toString();
            predictionDistribution.merge(decision_type, 1, Integer::sum);
        }

        // Detect drift: if distribution significantly different from baseline
        boolean driftDetected = isPredictionDriftSignificant(predictionDistribution);
        String driftStatus = driftDetected ? "DRIFT_DETECTED" : "STABLE";

        return new ModelDriftMetrics(modelVersion, driftStatus, predictionDistribution);
    }

    public Map<String, Object> getModelMetrics(String modelVersion) {
        long modelCount = aiDecisionRepository.countByModelVersion(modelVersion);
        long recentRejections = aiDecisionRepository.countRejectionsAfter(Instant.now().minusSeconds(3600));

        return Map.of(
            "model_version", modelVersion,
            "total_decisions", modelCount,
            "recent_rejections_1h", recentRejections,
            "rejection_rate", modelCount > 0 ? (double) recentRejections / modelCount : 0,
            "timestamp", Instant.now().toString()
        );
    }

    private String getConfidenceBucket(BigDecimal confidence) {
        double conf = confidence.doubleValue();
        if (conf >= 0.9) return "very_high";
        if (conf >= 0.75) return "high";
        if (conf >= 0.6) return "medium";
        if (conf >= 0.5) return "low";
        return "very_low";
    }

    private String determineHealth(double rejectionRate, BigDecimal avgConfidence) {
        if (rejectionRate > 0.5) {
            return "UNHEALTHY_HIGH_REJECTION";
        }
        if (avgConfidence.doubleValue() < 0.55) {
            return "UNHEALTHY_LOW_CONFIDENCE";
        }
        if (rejectionRate > 0.3 || avgConfidence.doubleValue() < 0.65) {
            return "DEGRADED";
        }
        return "HEALTHY";
    }

    private boolean isPredictionDriftSignificant(Map<String, Integer> distribution) {
        // Simple drift detection: if any prediction is > 60% of all predictions, it's drifted
        int total = distribution.values().stream().mapToInt(Integer::intValue).sum();
        return distribution.values().stream()
            .anyMatch(count -> (double) count / total > 0.6);
    }

    public static class ModelHealthMetrics {
        public final String modelVersion;
        public final String healthStatus;
        public final int decisionCount;
        public final long rejectionCount;
        public final double averageConfidence;
        public final Map<String, Integer> confidenceDistribution;

        public ModelHealthMetrics(String modelVersion, String healthStatus, int decisionCount, long rejectionCount,
                                 double averageConfidence, Map<String, Integer> confidenceDistribution) {
            this.modelVersion = modelVersion;
            this.healthStatus = healthStatus;
            this.decisionCount = decisionCount;
            this.rejectionCount = rejectionCount;
            this.averageConfidence = averageConfidence;
            this.confidenceDistribution = confidenceDistribution;
        }
    }

    public static class ModelDriftMetrics {
        public final String modelVersion;
        public final String driftStatus;
        public final Map<String, Integer> predictionDistribution;

        public ModelDriftMetrics(String modelVersion, String driftStatus, Map<String, Integer> predictionDistribution) {
            this.modelVersion = modelVersion;
            this.driftStatus = driftStatus;
            this.predictionDistribution = predictionDistribution;
        }
    }
}
