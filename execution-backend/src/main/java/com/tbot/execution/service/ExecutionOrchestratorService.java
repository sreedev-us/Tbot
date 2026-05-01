package com.tbot.execution.service;

import com.tbot.execution.config.RiskProperties;
import com.tbot.execution.domain.CommandType;
import com.tbot.execution.domain.ExecutionStatus;
import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.dto.ExecutionDecisionResponse;
import com.tbot.execution.dto.ExecutionSignalRequest;
import com.tbot.execution.entity.ExecutionRecord;
import com.tbot.execution.entity.OrderRecord;
import com.tbot.execution.repository.ExecutionRecordRepository;
import com.tbot.execution.repository.OrderRecordRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class ExecutionOrchestratorService {

    private final OrderRecordRepository orderRecordRepository;
    private final ExecutionRecordRepository executionRecordRepository;
    private final RiskManagementService riskManagementService;
    private final TelemetryService telemetryService;
    private final RiskProperties riskProperties;
    private final ControlCommandService controlCommandService;
    private final PaperTradeService paperTradeService;

    public ExecutionOrchestratorService(
            OrderRecordRepository orderRecordRepository,
            ExecutionRecordRepository executionRecordRepository,
            RiskManagementService riskManagementService,
            TelemetryService telemetryService,
            RiskProperties riskProperties,
            ControlCommandService controlCommandService,
            PaperTradeService paperTradeService
    ) {
        this.orderRecordRepository = orderRecordRepository;
        this.executionRecordRepository = executionRecordRepository;
        this.riskManagementService = riskManagementService;
        this.telemetryService = telemetryService;
        this.riskProperties = riskProperties;
        this.controlCommandService = controlCommandService;
        this.paperTradeService = paperTradeService;
    }

    @Transactional
    public ExecutionDecisionResponse processSignal(ExecutionSignalRequest request) {
        OrderRecord existing = orderRecordRepository.findBySignalId(request.signalId()).orElse(null);
        if (existing != null) {
            return new ExecutionDecisionResponse(
                    existing.getSignalId(),
                    existing.getStatus(),
                    existing.getRejectionReason(),
                    null,
                    existing.getRequestedNotional(),
                    existing.getRequestedNotional(),
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    Instant.now()
            );
        }

        OrderRecord order = new OrderRecord();
        order.setSignalId(request.signalId());
        order.setCorrelationId(request.correlationId());
        order.setAsset(request.asset());
        order.setExchangeName(request.exchange());
        order.setAction(request.action());
        order.setConfidence(request.confidence());
        order.setRequestedNotional(request.requestedNotional());
        order.setStrategyName(request.strategyName());
        order.setGeneratedAt(request.generatedAt());
        order.setReceivedAt(Instant.now());
        order.setStatus(OrderStatus.RECEIVED);
        orderRecordRepository.save(order);

        if (controlCommandService.getState().engineHalted()) {
            order.setStatus(OrderStatus.REJECTED);
            order.setRejectionReason("ENGINE_HALTED");
            order.setCompletedAt(Instant.now());
            return new ExecutionDecisionResponse(
                    order.getSignalId(),
                    order.getStatus(),
                    order.getRejectionReason(),
                    null,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    Instant.now()
            );
        }

        RiskDecision decision = riskManagementService.evaluate(request);
        if (!decision.approved()) {
            order.setStatus(OrderStatus.REJECTED);
            order.setRejectionReason(decision.reason());
            order.setCompletedAt(Instant.now());
            telemetryService.record(
                    "risk",
                    "signal_rejected",
                    request.requestedNotional(),
                    "usd",
                    "exchange=" + request.exchange() + ",asset=" + request.asset()
            );
            return new ExecutionDecisionResponse(
                    order.getSignalId(),
                    order.getStatus(),
                    decision.reason(),
                    null,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    BigDecimal.ZERO,
                    decision.sentimentScore(),
                    decision.sentimentConfidence(),
                    Instant.now()
            );
        }

        order.setRequestedNotional(decision.approvedNotional());
        order.setStatus(OrderStatus.ROUTED);
        order.setRoutedAt(Instant.now());

        Instant exchangeAckAt = Instant.now().plusMillis(riskProperties.getArtificialLatencyMs());
        BigDecimal slippageFee = decision.approvedNotional()
                .multiply(riskProperties.getSlippageBps())
                .divide(BigDecimal.valueOf(10_000), 8, RoundingMode.HALF_UP);
        BigDecimal executedNotional = decision.approvedNotional().subtract(slippageFee).max(BigDecimal.ZERO);

        ExecutionRecord execution = new ExecutionRecord();
        execution.setOrder(order);
        execution.setVenueOrderId("SIM-" + UUID.randomUUID().toString().substring(0, 12));
        execution.setExchangeName(request.exchange());
        execution.setAction(request.action());
        execution.setStatus(ExecutionStatus.SIMULATED);
        execution.setRequestedNotional(decision.approvedNotional());
        execution.setExecutedNotional(executedNotional);
        execution.setSlippageFee(slippageFee);
        execution.setExchangeAckAt(exchangeAckAt);
        execution.setFillConfirmedAt(exchangeAckAt.plusMillis(25));
        executionRecordRepository.save(execution);

        order.setStatus(OrderStatus.EXECUTED);
        order.setCompletedAt(execution.getFillConfirmedAt());
        paperTradeService.openTrade(
                order,
                request.marketPrice(),
                request.stopLossPrice(),
                request.takeProfitPrice()
        );

        if (controlCommandService.getState().liquidationRequested()) {
            controlCommandService.clearActiveCommand(CommandType.LIQUIDATE_ALL);
        }

        telemetryService.record(
                "execution",
                "signal_to_fill_latency",
                BigDecimal.valueOf(Duration.between(request.generatedAt(), execution.getFillConfirmedAt()).toMillis()),
                "ms",
                "exchange=" + request.exchange() + ",asset=" + request.asset()
        );

        return new ExecutionDecisionResponse(
                order.getSignalId(),
                order.getStatus(),
                "SIMULATED_FILL",
                execution.getVenueOrderId(),
                decision.approvedNotional(),
                executedNotional,
                slippageFee,
                decision.sentimentScore(),
                decision.sentimentConfidence(),
                Instant.now()
        );
    }
}
