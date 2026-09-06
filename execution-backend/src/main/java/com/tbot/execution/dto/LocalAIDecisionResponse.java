package com.tbot.execution.dto;

import com.tbot.execution.domain.AIDecisionType;
import java.math.BigDecimal;

public record LocalAIDecisionResponse(
    AIDecisionType decision,
    BigDecimal confidence,
    java.util.Map<String, Object> modelOutputs,
    String rationale
) {}
