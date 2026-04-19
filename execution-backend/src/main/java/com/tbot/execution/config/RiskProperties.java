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
    private List<String> illiquidAssets = new ArrayList<>();
    private long artificialLatencyMs = 100;
    private BigDecimal slippageBps = BigDecimal.TEN;
}
