package com.tbot.execution.dto;

import com.tbot.execution.domain.TrendDirection;
import com.tbot.execution.domain.VolatilityLevel;
import com.tbot.execution.domain.MarketRegime;
import java.math.BigDecimal;

public record ServerAIAnalysisResponse(
    TrendDirection trend,
    BigDecimal trendStrength,
    VolatilityLevel volatility,
    BigDecimal volatilityScore,
    MarketRegime regime,
    BigDecimal regimeConfidence,
    String riskEnvironment,
    BigDecimal riskScore
) {}
