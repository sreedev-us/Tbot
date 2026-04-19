package com.tbot.execution.service;

import com.tbot.execution.config.RiskProperties;
import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.dto.ExecutionSignalRequest;
import com.tbot.execution.repository.OrderRecordRepository;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.EnumSet;
import org.springframework.stereotype.Service;

@Service
public class RiskManagementService {

    private final OrderRecordRepository orderRecordRepository;
    private final RiskProperties riskProperties;

    public RiskManagementService(OrderRecordRepository orderRecordRepository, RiskProperties riskProperties) {
        this.orderRecordRepository = orderRecordRepository;
        this.riskProperties = riskProperties;
    }

    public RiskDecision evaluate(ExecutionSignalRequest request) {
        if (riskProperties.getIlliquidAssets().contains(request.asset())) {
            return RiskDecision.reject("ASSET_FLAGGED_ILLIQUID");
        }

        if (request.confidence().compareTo(riskProperties.getMaxConfidenceRejectionThreshold()) < 0) {
            return RiskDecision.reject("CONFIDENCE_BELOW_THRESHOLD");
        }

        BigDecimal activeExposure = orderRecordRepository.sumRequestedNotionalByStatusIn(
                EnumSet.of(OrderStatus.RECEIVED, OrderStatus.ROUTED, OrderStatus.EXECUTED)
        );
        BigDecimal projectedExposure = activeExposure.add(request.requestedNotional());
        if (projectedExposure.compareTo(riskProperties.getMaxPortfolioExposure()) > 0) {
            return RiskDecision.reject("PORTFOLIO_EXPOSURE_LIMIT");
        }

        Instant startOfDay = LocalDate.now(ZoneOffset.UTC).atStartOfDay().toInstant(ZoneOffset.UTC);
        BigDecimal rejectedToday = orderRecordRepository.sumRejectedNotionalSince(startOfDay);
        if (rejectedToday.compareTo(riskProperties.getMaxDailyDrawdown()) > 0) {
            return RiskDecision.reject("DAILY_DRAWDOWN_LIMIT");
        }

        return RiskDecision.approve();
    }
}
