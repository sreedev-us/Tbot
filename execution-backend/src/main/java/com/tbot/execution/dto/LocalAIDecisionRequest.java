package com.tbot.execution.dto;

import com.tbot.execution.domain.AIDecisionType;
import java.math.BigDecimal;

public record LocalAIDecisionRequest(
    String asset,
    String modelVersion,
    String featureVersion,
    java.util.Map<String, Double> marketFeatures,
    ServerAIAnalysisResponse serverAnalysis,
    BigDecimal currentSentiment,
    java.util.Map<String, Object> positionContext
) {}
