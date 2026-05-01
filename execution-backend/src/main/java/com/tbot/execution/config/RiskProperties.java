package com.tbot.execution.config;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "tbot.risk")
public class RiskProperties {

    private BigDecimal maxPortfolioExposure = BigDecimal.valueOf(10_000);
    private BigDecimal maxDailyDrawdown = BigDecimal.valueOf(500);
    private BigDecimal maxConfidenceRejectionThreshold = BigDecimal.valueOf(0.55);
    private BigDecimal negativeSentimentBlockThreshold = BigDecimal.valueOf(-0.80);
    private BigDecimal positiveSentimentBoostThreshold = BigDecimal.valueOf(0.70);
    private BigDecimal sentimentConfidenceFloor = BigDecimal.valueOf(0.60);
    private BigDecimal positiveSentimentPositionMultiplier = BigDecimal.valueOf(1.20);
    private long sentimentMaxAgeSeconds = 900;
    private List<String> illiquidAssets = new ArrayList<>();
    private long artificialLatencyMs = 100;
    private BigDecimal slippageBps = BigDecimal.TEN;
}
