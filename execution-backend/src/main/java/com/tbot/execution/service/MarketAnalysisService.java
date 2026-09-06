package com.tbot.execution.service;

import com.tbot.execution.domain.MarketRegime;
import com.tbot.execution.domain.TrendDirection;
import com.tbot.execution.domain.VolatilityLevel;
import com.tbot.execution.dto.ServerAIAnalysisRequest;
import com.tbot.execution.dto.ServerAIAnalysisResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.Arrays;

@Slf4j
@Service
public class MarketAnalysisService {

    private static final int MIN_PRICE_HISTORY_SIZE = 20;

    public ServerAIAnalysisResponse analyzeMarketContext(ServerAIAnalysisRequest request) {
        log.debug("Analyzing market context for asset: {}", request.asset());

        if (request.recentPrices() == null || request.recentPrices().length < MIN_PRICE_HISTORY_SIZE) {
            log.warn("Insufficient price history for {} analysis", request.asset());
            return createNeutralAnalysis();
        }

        TrendDirection trend = analyzeTrend(request.recentPrices());
        BigDecimal trendStrength = calculateTrendStrength(request.recentPrices());
        VolatilityLevel volatility = calculateVolatility(request.recentPrices());
        BigDecimal volatilityScore = calculateVolatilityScore(request.recentPrices());
        MarketRegime regime = determineMarketRegime(request.recentPrices());
        BigDecimal regimeConfidence = calculateRegimeConfidence(request.recentPrices());
        String riskEnvironment = assessRiskEnvironment(volatilityScore, request.sentiment());
        BigDecimal riskScore = calculateRiskScore(volatilityScore, request.sentiment());

        return new ServerAIAnalysisResponse(
            trend,
            trendStrength,
            volatility,
            volatilityScore,
            regime,
            regimeConfidence,
            riskEnvironment,
            riskScore
        );
    }

    private TrendDirection analyzeTrend(BigDecimal[] prices) {
        if (prices.length < 2) {
            return TrendDirection.NEUTRAL;
        }

        BigDecimal sma20 = calculateSMA(prices, 20);
        BigDecimal sma50 = calculateSMA(prices, Math.min(50, prices.length));
        BigDecimal currentPrice = prices[prices.length - 1];

        if (currentPrice.compareTo(sma20) > 0 && sma20.compareTo(sma50) > 0) {
            return TrendDirection.BULLISH;
        } else if (currentPrice.compareTo(sma20) < 0 && sma20.compareTo(sma50) < 0) {
            return TrendDirection.BEARISH;
        } else {
            return TrendDirection.NEUTRAL;
        }
    }

    private BigDecimal calculateTrendStrength(BigDecimal[] prices) {
        if (prices.length < 2) {
            return BigDecimal.ZERO;
        }

        BigDecimal sma20 = calculateSMA(prices, 20);
        BigDecimal sma50 = calculateSMA(prices, Math.min(50, prices.length));
        BigDecimal currentPrice = prices[prices.length - 1];

        BigDecimal priceDiff = currentPrice.subtract(sma20).abs();
        BigDecimal smaDiff = sma20.subtract(sma50).abs();
        BigDecimal totalDiff = priceDiff.add(smaDiff);

        if (totalDiff.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }

        BigDecimal strength = priceDiff.divide(totalDiff, 6, java.math.RoundingMode.HALF_UP);
        return strength.min(BigDecimal.ONE);
    }

    private VolatilityLevel calculateVolatility(BigDecimal[] prices) {
        BigDecimal volatilityScore = calculateVolatilityScore(prices);

        if (volatilityScore.compareTo(new BigDecimal("0.02")) < 0) {
            return VolatilityLevel.LOW;
        } else if (volatilityScore.compareTo(new BigDecimal("0.05")) < 0) {
            return VolatilityLevel.MEDIUM;
        } else if (volatilityScore.compareTo(new BigDecimal("0.10")) < 0) {
            return VolatilityLevel.HIGH;
        } else {
            return VolatilityLevel.EXTREME;
        }
    }

    private BigDecimal calculateVolatilityScore(BigDecimal[] prices) {
        if (prices.length < 2) {
            return BigDecimal.ZERO;
        }

        BigDecimal mean = Arrays.stream(prices)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(prices.length), 8, java.math.RoundingMode.HALF_UP);

        BigDecimal sumSquaredDiff = Arrays.stream(prices)
            .map(price -> price.subtract(mean).pow(2))
            .reduce(BigDecimal.ZERO, BigDecimal::add);

        BigDecimal variance = sumSquaredDiff.divide(new BigDecimal(prices.length), 8, java.math.RoundingMode.HALF_UP);
        double stdDev = Math.sqrt(variance.doubleValue());

        double coefficientOfVariation = stdDev / mean.doubleValue();
        return new BigDecimal(coefficientOfVariation).min(BigDecimal.ONE);
    }

    private MarketRegime determineMarketRegime(BigDecimal[] prices) {
        if (prices.length < 20) {
            return MarketRegime.CHOPPY;
        }

        BigDecimal range = Arrays.stream(prices)
            .reduce(prices[0], (max, price) -> price.compareTo(max) > 0 ? price : max)
            .subtract(Arrays.stream(prices)
                .reduce(prices[0], (min, price) -> price.compareTo(min) < 0 ? price : min));

        BigDecimal mean = Arrays.stream(prices)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(prices.length), 8, java.math.RoundingMode.HALF_UP);

        BigDecimal rangePercent = range.divide(mean, 8, java.math.RoundingMode.HALF_UP);

        if (rangePercent.compareTo(new BigDecimal("0.03")) < 0) {
            return MarketRegime.MEAN_REVERSION;
        } else if (rangePercent.compareTo(new BigDecimal("0.10")) > 0) {
            return MarketRegime.TRENDING;
        } else {
            return MarketRegime.CHOPPY;
        }
    }

    private BigDecimal calculateRegimeConfidence(BigDecimal[] prices) {
        return new BigDecimal("0.65");
    }

    private String assessRiskEnvironment(BigDecimal volatilityScore, BigDecimal sentiment) {
        boolean highVolatility = volatilityScore.compareTo(new BigDecimal("0.08")) > 0;
        boolean negativeSentiment = sentiment.compareTo(BigDecimal.ZERO) < 0;

        if (highVolatility && negativeSentiment) {
            return "ELEVATED";
        } else if (highVolatility || negativeSentiment) {
            return "MODERATE";
        } else {
            return "LOW";
        }
    }

    private BigDecimal calculateRiskScore(BigDecimal volatilityScore, BigDecimal sentiment) {
        BigDecimal volComponent = volatilityScore.multiply(new BigDecimal("0.6"));
        BigDecimal sentimentComponent = sentiment.multiply(new BigDecimal("-0.4"));
        return volComponent.add(sentimentComponent).max(BigDecimal.ZERO).min(BigDecimal.ONE);
    }

    private BigDecimal calculateSMA(BigDecimal[] prices, int period) {
        int actualPeriod = Math.min(period, prices.length);
        return Arrays.stream(prices, prices.length - actualPeriod, prices.length)
            .reduce(BigDecimal.ZERO, BigDecimal::add)
            .divide(new BigDecimal(actualPeriod), 8, java.math.RoundingMode.HALF_UP);
    }

    private ServerAIAnalysisResponse createNeutralAnalysis() {
        return new ServerAIAnalysisResponse(
            TrendDirection.NEUTRAL,
            BigDecimal.ZERO,
            VolatilityLevel.MEDIUM,
            new BigDecimal("0.05"),
            MarketRegime.CHOPPY,
            new BigDecimal("0.50"),
            "MODERATE",
            new BigDecimal("0.50")
        );
    }
}
