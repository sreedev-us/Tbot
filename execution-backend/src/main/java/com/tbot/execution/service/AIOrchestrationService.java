package com.tbot.execution.service;

import com.tbot.execution.domain.AIDecisionType;
import com.tbot.execution.dto.LocalAIDecisionRequest;
import com.tbot.execution.dto.LocalAIDecisionResponse;
import com.tbot.execution.dto.ServerAIAnalysisRequest;
import com.tbot.execution.dto.ServerAIAnalysisResponse;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import org.springframework.stereotype.Service;

@Service
public class AIOrchestrationService {

    private final MarketAnalysisService marketAnalysisService;
    private final LocalDecisionService localDecisionService;
    private final ModelLoaderService modelLoaderService;

    public AIOrchestrationService(
            MarketAnalysisService marketAnalysisService,
            LocalDecisionService localDecisionService,
            ModelLoaderService modelLoaderService
    ) {
        this.marketAnalysisService = marketAnalysisService;
        this.localDecisionService = localDecisionService;
        this.modelLoaderService = modelLoaderService;
    }

    public AIDecisionPipeline executeAIPipeline(
            String asset,
            BigDecimal currentPrice,
            BigDecimal[] recentPrices,
            long[] timestamps,
            BigDecimal volume,
            BigDecimal sentiment,
            BigDecimal sentimentConfidence,
            String modelVersion,
            String featureVersion,
            Map<String, Object> positionContext
    ) {
        Instant t0Start = Instant.now();

        AIDecisionPipeline.AIDecisionPipelineBuilder builder = AIDecisionPipeline.builder()
            .asset(asset)
            .modelVersion(modelVersion)
            .featureVersion(featureVersion)
            .pipelineStartedAt(t0Start);

        try {
            // T1: Server AI Analysis
            Instant t2Start = Instant.now();
            ServerAIAnalysisRequest serverRequest = new ServerAIAnalysisRequest(
                asset,
                currentPrice,
                recentPrices,
                timestamps,
                volume,
                sentiment,
                sentimentConfidence
            );
            ServerAIAnalysisResponse serverAnalysis = marketAnalysisService.analyzeMarketContext(serverRequest);
            Instant t2End = Instant.now();
            builder.serverAnalysisStartedAt(t2Start)
                    .serverAnalysisCompletedAt(t2End)
                    .serverAnalysis(serverAnalysis);

            // T3: Local AI Decision
            Instant t4Start = Instant.now();
            Map<String, Double> marketFeatures = extractMarketFeatures(recentPrices, volume, sentiment);
            LocalAIDecisionRequest localRequest = new LocalAIDecisionRequest(
                asset,
                modelVersion,
                featureVersion,
                marketFeatures,
                serverAnalysis,
                sentiment,
                positionContext
            );
            LocalAIDecisionResponse localDecision = localDecisionService.makeDecision(localRequest);
            Instant t4End = Instant.now();
            builder.localDecisionStartedAt(t4Start)
                    .localDecisionCompletedAt(t4End)
                    .localDecision(localDecision)
                    .marketFeatures(marketFeatures);

            // Calculate model hash
            String modelHash = modelLoaderService.getModelHash(modelVersion.split("-")[0], modelVersion);
            builder.modelHash(modelHash);

            return builder.build();
        } catch (Exception e) {
            builder.error(e.getMessage());
            // Return HOLD decision on error
            builder.localDecision(new LocalAIDecisionResponse(
                AIDecisionType.HOLD,
                BigDecimal.ZERO,
                Map.of("error", e.getMessage()),
                "Error during AI pipeline: " + e.getMessage()
            ));
            return builder.build();
        }
    }

    private Map<String, Double> extractMarketFeatures(BigDecimal[] prices, BigDecimal volume, BigDecimal sentiment) {
        Map<String, Double> features = new HashMap<>();

        if (prices != null && prices.length > 0) {
            features.put("current_price", prices[prices.length - 1].doubleValue());
            features.put("price_change_pct", calculatePriceChange(prices).doubleValue());

            if (prices.length >= 20) {
                features.put("sma_20", calculateSMA(prices, 20).doubleValue());
            }
            if (prices.length >= 50) {
                features.put("sma_50", calculateSMA(prices, 50).doubleValue());
            }

            features.put("volatility", calculateVolatility(prices).doubleValue());
        }

        if (volume != null) {
            features.put("volume", volume.doubleValue());
        }

        if (sentiment != null) {
            features.put("sentiment", sentiment.doubleValue());
        }

        return features;
    }

    private BigDecimal calculatePriceChange(BigDecimal[] prices) {
        if (prices.length < 2) {
            return BigDecimal.ZERO;
        }
        BigDecimal current = prices[prices.length - 1];
        BigDecimal previous = prices[0];
        return current.subtract(previous).divide(previous, 8, java.math.RoundingMode.HALF_UP);
    }

    private BigDecimal calculateSMA(BigDecimal[] prices, int period) {
        int actualPeriod = Math.min(period, prices.length);
        return java.util.Arrays.stream(prices, prices.length - actualPeriod, prices.length)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(actualPeriod), 8, java.math.RoundingMode.HALF_UP);
    }

    private BigDecimal calculateVolatility(BigDecimal[] prices) {
        if (prices.length < 2) {
            return BigDecimal.ZERO;
        }

        BigDecimal mean = java.util.Arrays.stream(prices)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(prices.length), 8, java.math.RoundingMode.HALF_UP);

        BigDecimal sumSquaredDiff = java.util.Arrays.stream(prices)
            .map(p -> p.subtract(mean).pow(2))
            .reduce(BigDecimal.ZERO, BigDecimal::add);

        BigDecimal variance = sumSquaredDiff.divide(new BigDecimal(prices.length), 8, java.math.RoundingMode.HALF_UP);
        double stdDev = Math.sqrt(variance.doubleValue());
        return new BigDecimal(stdDev / mean.doubleValue());
    }

    @Builder
    @Getter
    public static class AIDecisionPipeline {
        private String asset;
        private String modelVersion;
        private String modelHash;
        private String featureVersion;
        private Instant pipelineStartedAt;
        private ServerAIAnalysisResponse serverAnalysis;
        private Instant serverAnalysisStartedAt;
        private Instant serverAnalysisCompletedAt;
        private LocalAIDecisionResponse localDecision;
        private Instant localDecisionStartedAt;
        private Instant localDecisionCompletedAt;
        private Map<String, Double> marketFeatures;
        private String error;

        public long getServerAILatencyMs() {
            if (serverAnalysisStartedAt == null || serverAnalysisCompletedAt == null) {
                return 0;
            }
            return java.time.Duration.between(serverAnalysisStartedAt, serverAnalysisCompletedAt).toMillis();
        }

        public long getLocalAILatencyMs() {
            if (localDecisionStartedAt == null || localDecisionCompletedAt == null) {
                return 0;
            }
            return java.time.Duration.between(localDecisionStartedAt, localDecisionCompletedAt).toMillis();
        }

        public long getTotalAILatencyMs() {
            if (pipelineStartedAt == null || localDecisionCompletedAt == null) {
                return 0;
            }
            return java.time.Duration.between(pipelineStartedAt, localDecisionCompletedAt).toMillis();
        }

        public String toJson() {
            return String.format(
                "{\"model_version\":\"%s\",\"model_hash\":\"%s\",\"feature_version\":\"%s\"," +
                "\"server_ai_latency_ms\":%d,\"local_ai_latency_ms\":%d,\"total_latency_ms\":%d}",
                modelVersion, modelHash, featureVersion,
                getServerAILatencyMs(), getLocalAILatencyMs(), getTotalAILatencyMs()
            );
        }
    }
}
