package com.tbot.execution.service;

import com.tbot.execution.domain.AIDecisionType;
import com.tbot.execution.dto.LocalAIDecisionRequest;
import com.tbot.execution.dto.LocalAIDecisionResponse;
import com.tbot.execution.dto.ServerAIAnalysisResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
public class LocalDecisionService {

    public LocalAIDecisionResponse makeDecision(LocalAIDecisionRequest request) {
        log.debug("Making local AI decision for asset: {}", request.asset());

        try {
            Map<String, Object> modelOutputs = new HashMap<>();
            
            // Simple heuristic model for now (placeholder for trained model)
            BigDecimal buyScore = calculateBuyScore(request);
            BigDecimal sellScore = calculateSellScore(request);
            BigDecimal holdScore = calculateHoldScore(request);

            modelOutputs.put("buyScore", buyScore);
            modelOutputs.put("sellScore", sellScore);
            modelOutputs.put("holdScore", holdScore);
            modelOutputs.put("serverAnalysis", request.serverAnalysis());

            AIDecisionType decision = determineDecision(buyScore, sellScore, holdScore);
            BigDecimal confidence = calculateConfidence(buyScore, sellScore, holdScore);
            String rationale = generateRationale(decision, confidence, request.serverAnalysis());

            log.debug("Local AI decision: {} with confidence {}", decision, confidence);

            return new LocalAIDecisionResponse(
                decision,
                confidence,
                modelOutputs,
                rationale
            );
        } catch (Exception e) {
            log.error("Error in local AI decision making", e);
            // Default to HOLD on error
            return new LocalAIDecisionResponse(
                AIDecisionType.HOLD,
                BigDecimal.ZERO,
                Map.of("error", e.getMessage()),
                "Error during decision making: " + e.getMessage()
            );
        }
    }

    private BigDecimal calculateBuyScore(LocalAIDecisionRequest request) {
        ServerAIAnalysisResponse analysis = request.serverAnalysis();
        BigDecimal score = BigDecimal.ZERO;

        // Trend component
        score = score.add(analysis.trendStrength());

        // Sentiment component (positive sentiment favors buy)
        BigDecimal sentimentFactor = request.currentSentiment().max(BigDecimal.ZERO);
        score = score.add(sentimentFactor.multiply(new BigDecimal("0.3")));

        // Volatility penalty (high volatility reduces buy signal)
        BigDecimal volatilityPenalty = new BigDecimal(analysis.volatilityScore().doubleValue());
        score = score.subtract(volatilityPenalty.multiply(new BigDecimal("0.1")));

        // Position context
        if (request.positionContext() != null) {
            Object currentPosition = request.positionContext().get("currentPosition");
            if (currentPosition != null && ((Number) currentPosition).doubleValue() > 0) {
                score = score.subtract(new BigDecimal("0.2")); // Reduce buy signal if already long
            }
        }

        return score.max(BigDecimal.ZERO).min(BigDecimal.ONE);
    }

    private BigDecimal calculateSellScore(LocalAIDecisionRequest request) {
        ServerAIAnalysisResponse analysis = request.serverAnalysis();
        BigDecimal score = BigDecimal.ZERO;

        // Reverse trend component
        score = score.add(BigDecimal.ONE.subtract(analysis.trendStrength()));

        // Negative sentiment favors sell
        BigDecimal sentimentFactor = request.currentSentiment().negate().max(BigDecimal.ZERO);
        score = score.add(sentimentFactor.multiply(new BigDecimal("0.3")));

        // High volatility increases sell signal
        score = score.add(analysis.volatilityScore().multiply(new BigDecimal("0.2")));

        // Position context
        if (request.positionContext() != null) {
            Object currentPosition = request.positionContext().get("currentPosition");
            if (currentPosition == null || ((Number) currentPosition).doubleValue() <= 0) {
                score = score.subtract(new BigDecimal("0.2")); // Reduce sell signal if not long
            }
        }

        return score.max(BigDecimal.ZERO).min(BigDecimal.ONE);
    }

    private BigDecimal calculateHoldScore(LocalAIDecisionRequest request) {
        ServerAIAnalysisResponse analysis = request.serverAnalysis();
        
        // Hold is likely in choppy market or neutral conditions
        if ("CHOPPY".equals(analysis.regime().toString())) {
            return new BigDecimal("0.7");
        }

        BigDecimal uncertainty = BigDecimal.ONE.subtract(analysis.regimeConfidence());
        return BigDecimal.ONE.subtract(uncertainty);
    }

    private AIDecisionType determineDecision(BigDecimal buyScore, BigDecimal sellScore, BigDecimal holdScore) {
        if (buyScore.compareTo(sellScore) > 0 && buyScore.compareTo(holdScore) > 0) {
            return AIDecisionType.BUY;
        } else if (sellScore.compareTo(buyScore) > 0 && sellScore.compareTo(holdScore) > 0) {
            return AIDecisionType.SELL;
        } else {
            return AIDecisionType.HOLD;
        }
    }

    private BigDecimal calculateConfidence(BigDecimal buyScore, BigDecimal sellScore, BigDecimal holdScore) {
        BigDecimal maxScore = buyScore.max(sellScore).max(holdScore);
        BigDecimal sumScores = buyScore.add(sellScore).add(holdScore);

        if (sumScores.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }

        return maxScore.divide(sumScores, 6, java.math.RoundingMode.HALF_UP);
    }

    private String generateRationale(AIDecisionType decision, BigDecimal confidence, ServerAIAnalysisResponse analysis) {
        return String.format(
            "%s decision with confidence %.2f%% based on trend=%s, volatility=%s, regime=%s",
            decision,
            confidence.multiply(new BigDecimal("100")).doubleValue(),
            analysis.trend(),
            analysis.volatility(),
            analysis.regime()
        );
    }
}
