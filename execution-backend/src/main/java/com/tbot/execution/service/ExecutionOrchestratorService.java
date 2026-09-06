package com.tbot.execution.service;

import com.tbot.execution.config.RiskProperties;
import com.tbot.execution.domain.CommandType;
import com.tbot.execution.domain.ExecutionStatus;
import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.dto.ExecutionDecisionResponse;
import com.tbot.execution.dto.ExecutionSignalRequest;
import com.tbot.execution.entity.AIDecisionRecord;
import com.tbot.execution.entity.ExecutionRecord;
import com.tbot.execution.entity.OrderRecord;
import com.tbot.execution.repository.AIDecisionRepository;
import com.tbot.execution.repository.ExecutionRecordRepository;
import com.tbot.execution.repository.OrderRecordRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class ExecutionOrchestratorService {

    private final OrderRecordRepository orderRecordRepository;
    private final ExecutionRecordRepository executionRecordRepository;
    private final AIDecisionRepository aiDecisionRepository;
    private final RiskManagementService riskManagementService;
    private final TelemetryService telemetryService;
    private final RiskProperties riskProperties;
    private final ControlCommandService controlCommandService;
    private final PaperTradeService paperTradeService;
    private final AIOrchestrationService aiOrchestrationService;

    @Value("${tbot.ai.model-version:local-v1.0.0}")
    private String aiModelVersion;

    @Value("${tbot.ai.feature-version:v1}")
    private String aiFeatureVersion;

    @Value("${tbot.ai.enable:false}")
    private boolean aiEnabled;

    public ExecutionOrchestratorService(
            OrderRecordRepository orderRecordRepository,
            ExecutionRecordRepository executionRecordRepository,
            AIDecisionRepository aiDecisionRepository,
            RiskManagementService riskManagementService,
            TelemetryService telemetryService,
            RiskProperties riskProperties,
            ControlCommandService controlCommandService,
            PaperTradeService paperTradeService,
            AIOrchestrationService aiOrchestrationService
    ) {
        this.orderRecordRepository = orderRecordRepository;
        this.executionRecordRepository = executionRecordRepository;
        this.aiDecisionRepository = aiDecisionRepository;
        this.riskManagementService = riskManagementService;
        this.telemetryService = telemetryService;
        this.riskProperties = riskProperties;
        this.controlCommandService = controlCommandService;
        this.paperTradeService = paperTradeService;
        this.aiOrchestrationService = aiOrchestrationService;
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

    @Transactional
    public ExecutionDecisionResponse processSignalWithAI(ExecutionSignalRequest request) {
        log.info("Processing signal with AI pipeline: {}", request.signalId());
        Instant t0 = Instant.now();

        OrderRecord existing = orderRecordRepository.findBySignalId(request.signalId()).orElse(null);
        if (existing != null) {
            log.debug("Signal already processed: {}", request.signalId());
            return buildResponse(existing);
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
            orderRecordRepository.save(order);
            return buildResponse(order);
        }

        // Execute AI pipeline if enabled
        AIOrchestrationService.AIDecisionPipeline aiPipeline = null;
        if (aiEnabled) {
            try {
                log.debug("Running AI decision pipeline for {}", request.asset());
                aiPipeline = aiOrchestrationService.executeAIPipeline(
                    request.asset(),
                    request.marketPrice(),
                    new BigDecimal[]{request.marketPrice()}, // TODO: Get full price history
                    new long[]{System.currentTimeMillis()},
                    BigDecimal.ZERO, // TODO: Get volume
                    BigDecimal.ZERO, // TODO: Get sentiment
                    BigDecimal.ZERO, // TODO: Get sentiment confidence
                    aiModelVersion,
                    aiFeatureVersion,
                    Map.of()
                );

                // Store AI decision record
                AIDecisionRecord aiDecision = new AIDecisionRecord();
                aiDecision.setOrder(order);
                aiDecision.setModelVersion(aiPipeline.getModelVersion());
                aiDecision.setModelHash(aiPipeline.getModelHash());
                aiDecision.setFeatureVersion(aiPipeline.getFeatureVersion());
                aiDecision.setInputTimestamp(t0);
                aiDecision.setDecision(aiPipeline.getLocalDecision().decision());
                aiDecision.setConfidence(aiPipeline.getLocalDecision().confidence());
                aiDecision.setFeaturesJson(convertMapToJson(aiPipeline.getMarketFeatures()));
                if (aiPipeline.getServerAnalysis() != null) {
                    aiDecision.setServerAIResultJson(convertServerAIToJson(aiPipeline.getServerAnalysis()));
                }
                aiDecision.setLocalAIResultJson(convertMapToJson(aiPipeline.getLocalDecision().modelOutputs()));
                aiDecision.setDecidedAt(Instant.now());
                aiDecisionRepository.save(aiDecision);

                log.info("AI Decision: {} with confidence {}", 
                    aiPipeline.getLocalDecision().decision(), 
                    aiPipeline.getLocalDecision().confidence());
                log.info("AI Latency: Server={}ms, Local={}ms, Total={}ms",
                    aiPipeline.getServerAILatencyMs(),
                    aiPipeline.getLocalAILatencyMs(),
                    aiPipeline.getTotalAILatencyMs());
            } catch (Exception e) {
                log.error("AI pipeline execution failed", e);
                order.setStatus(OrderStatus.REJECTED);
                order.setRejectionReason("AI_EXECUTION_ERROR");
                order.setCompletedAt(Instant.now());
                orderRecordRepository.save(order);
                return buildResponse(order);
            }
        }

        // Risk management (AI decision or original signal)
        RiskDecision riskDecision = riskManagementService.evaluate(request);
        if (!riskDecision.approved()) {
            order.setStatus(OrderStatus.REJECTED);
            order.setRejectionReason(riskDecision.reason());
            order.setCompletedAt(Instant.now());
            telemetryService.record("risk", "signal_rejected", request.requestedNotional(), "usd",
                "exchange=" + request.exchange() + ",asset=" + request.asset());
            return buildResponse(order);
        }

        // Execution
        order.setRequestedNotional(riskDecision.approvedNotional());
        order.setStatus(OrderStatus.ROUTED);
        order.setRoutedAt(Instant.now());

        Instant exchangeAckAt = Instant.now().plusMillis(riskProperties.getArtificialLatencyMs());
        BigDecimal slippageFee = riskDecision.approvedNotional()
            .multiply(riskProperties.getSlippageBps())
            .divide(BigDecimal.valueOf(10_000), 8, RoundingMode.HALF_UP);
        BigDecimal executedNotional = riskDecision.approvedNotional().subtract(slippageFee).max(BigDecimal.ZERO);

        ExecutionRecord execution = new ExecutionRecord();
        execution.setOrder(order);
        execution.setVenueOrderId("SIM-" + UUID.randomUUID().toString().substring(0, 12));
        execution.setExchangeName(request.exchange());
        execution.setAction(request.action());
        execution.setStatus(ExecutionStatus.SIMULATED);
        execution.setRequestedNotional(riskDecision.approvedNotional());
        execution.setExecutedNotional(executedNotional);
        execution.setSlippageFee(slippageFee);
        execution.setExchangeAckAt(exchangeAckAt);
        execution.setFillConfirmedAt(exchangeAckAt.plusMillis(25));
        executionRecordRepository.save(execution);

        order.setStatus(OrderStatus.EXECUTED);
        order.setCompletedAt(execution.getFillConfirmedAt());
        paperTradeService.openTrade(order, request.marketPrice(), request.stopLossPrice(), request.takeProfitPrice());

        if (controlCommandService.getState().liquidationRequested()) {
            controlCommandService.clearActiveCommand(CommandType.LIQUIDATE_ALL);
        }

        telemetryService.record("execution", "signal_to_fill_latency",
            BigDecimal.valueOf(Duration.between(request.generatedAt(), execution.getFillConfirmedAt()).toMillis()),
            "ms", "exchange=" + request.exchange() + ",asset=" + request.asset());

        if (aiPipeline != null) {
            telemetryService.record("ai", "total_latency_ms",
                BigDecimal.valueOf(aiPipeline.getTotalAILatencyMs()),
                "ms", "model=" + aiPipeline.getModelVersion());
        }

        return buildResponse(order);
    }

    private ExecutionDecisionResponse buildResponse(OrderRecord order) {
        return new ExecutionDecisionResponse(
            order.getSignalId(),
            order.getStatus(),
            order.getRejectionReason(),
            null,
            order.getRequestedNotional(),
            order.getRequestedNotional(),
            BigDecimal.ZERO,
            BigDecimal.ZERO,
            BigDecimal.ZERO,
            Instant.now()
        );
    }

    private String convertMapToJson(Map<String, ?> map) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(map);
        } catch (Exception e) {
            log.error("Error converting map to JSON", e);
            return "{}";
        }
    }

    private String convertServerAIToJson(Object obj) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(obj);
        } catch (Exception e) {
            log.error("Error converting server AI to JSON", e);
            return "{}";
        }
