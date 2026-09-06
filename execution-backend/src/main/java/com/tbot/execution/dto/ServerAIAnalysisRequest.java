package com.tbot.execution.dto;

import com.tbot.execution.domain.TrendDirection;
import com.tbot.execution.domain.VolatilityLevel;
import com.tbot.execution.domain.MarketRegime;
import java.math.BigDecimal;

public record ServerAIAnalysisRequest(
    String asset,
    BigDecimal currentPrice,
    BigDecimal[] recentPrices,
    long[] timestamps,
    BigDecimal volume,
    BigDecimal sentiment,
    BigDecimal sentimentConfidence
) {}
