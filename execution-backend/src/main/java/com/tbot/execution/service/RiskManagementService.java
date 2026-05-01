package com.tbot.execution.service;

import com.tbot.execution.config.RiskProperties;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.dto.ExecutionSignalRequest;
import com.tbot.execution.entity.MarketSentimentRecord;
import com.tbot.execution.repository.OrderRecordRepository;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.EnumSet;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
public class RiskManagementService {

    private final OrderRecordRepository orderRecordRepository;
    private final RiskProperties riskProperties;
    private final MarketSentimentService marketSentimentService;

    public RiskManagementService(
            OrderRecordRepository orderRecordRepository,
            RiskProperties riskProperties,
            MarketSentimentService marketSentimentService
    ) {
        this.orderRecordRepository = orderRecordRepository;
        this.riskProperties = riskProperties;
        this.marketSentimentService = marketSentimentService;
    }

    public RiskDecision evaluate(ExecutionSignalRequest request) {
        Instant sentimentCutoff = Instant.now().minusSeconds(riskProperties.getSentimentMaxAgeSeconds());
        Optional<MarketSentimentRecord> sentimentRecord = marketSentimentService.getLatestRecord(request.asset())
                .filter(record -> !record.getObservedAt().isBefore(sentimentCutoff));
        BigDecimal sentimentScore = sentimentRecord.map(MarketSentimentRecord::getSentimentScore).orElse(BigDecimal.ZERO);
        BigDecimal sentimentConfidence = sentimentRecord.map(MarketSentimentRecord::getConfidence).orElse(BigDecimal.ZERO);

        if (riskProperties.getIlliquidAssets().contains(request.asset())) {
            return RiskDecision.reject("ASSET_FLAGGED_ILLIQUID", BigDecimal.ZERO, sentimentScore, sentimentConfidence);
        }

        if (request.confidence().compareTo(riskProperties.getMaxConfidenceRejectionThreshold()) < 0) {
            return RiskDecision.reject("CONFIDENCE_BELOW_THRESHOLD", BigDecimal.ZERO, sentimentScore, sentimentConfidence);
        }

        if (request.action() == OrderAction.BUY
                && sentimentConfidence.compareTo(riskProperties.getSentimentConfidenceFloor()) >= 0
                && sentimentScore.compareTo(riskProperties.getNegativeSentimentBlockThreshold()) <= 0) {
            return RiskDecision.reject("SENTIMENT_CIRCUIT_BREAKER", BigDecimal.ZERO, sentimentScore, sentimentConfidence);
        }

        BigDecimal approvedNotional = request.requestedNotional();
        if (request.action() == OrderAction.BUY
                && sentimentConfidence.compareTo(riskProperties.getSentimentConfidenceFloor()) >= 0
                && sentimentScore.compareTo(riskProperties.getPositiveSentimentBoostThreshold()) >= 0) {
            approvedNotional = request.requestedNotional()
                    .multiply(riskProperties.getPositiveSentimentPositionMultiplier())
                    .setScale(8, RoundingMode.HALF_UP);
        }

        BigDecimal activeExposure = orderRecordRepository.sumRequestedNotionalByStatusIn(
                EnumSet.of(OrderStatus.RECEIVED, OrderStatus.ROUTED, OrderStatus.EXECUTED)
        );
        BigDecimal projectedExposure = activeExposure.add(approvedNotional);
        if (projectedExposure.compareTo(riskProperties.getMaxPortfolioExposure()) > 0) {
            return RiskDecision.reject("PORTFOLIO_EXPOSURE_LIMIT", BigDecimal.ZERO, sentimentScore, sentimentConfidence);
        }

        Instant startOfDay = LocalDate.now(ZoneOffset.UTC).atStartOfDay().toInstant(ZoneOffset.UTC);
        BigDecimal rejectedToday = orderRecordRepository.sumRejectedNotionalSince(startOfDay);
        if (rejectedToday.compareTo(riskProperties.getMaxDailyDrawdown()) > 0) {
            return RiskDecision.reject("DAILY_DRAWDOWN_LIMIT", BigDecimal.ZERO, sentimentScore, sentimentConfidence);
        }

        return RiskDecision.approve(approvedNotional, sentimentScore, sentimentConfidence);
    }
}
