package com.tbot.execution.service;

import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.dto.ControlStateResponse;
import com.tbot.execution.dto.DashboardSnapshotResponse;
import com.tbot.execution.repository.ExecutionRecordRepository;
import com.tbot.execution.repository.OrderRecordRepository;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.EnumSet;
import org.springframework.stereotype.Service;

@Service
public class DashboardService {

    private final OrderRecordRepository orderRecordRepository;
    private final ExecutionRecordRepository executionRecordRepository;
    private final ControlCommandService controlCommandService;

    public DashboardService(
            OrderRecordRepository orderRecordRepository,
            ExecutionRecordRepository executionRecordRepository,
            ControlCommandService controlCommandService
    ) {
        this.orderRecordRepository = orderRecordRepository;
        this.executionRecordRepository = executionRecordRepository;
        this.controlCommandService = controlCommandService;
    }

    public DashboardSnapshotResponse snapshot() {
        ControlStateResponse state = controlCommandService.getState();
        BigDecimal averageLatency = executionRecordRepository.findAll()
                .stream()
                .map(execution -> BigDecimal.valueOf(
                        java.time.Duration.between(
                                execution.getOrder().getGeneratedAt(),
                                execution.getFillConfirmedAt()
                        ).toMillis()
                ))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        long executionCount = executionRecordRepository.count();
        if (executionCount > 0) {
            averageLatency = averageLatency.divide(BigDecimal.valueOf(executionCount), 2, RoundingMode.HALF_UP);
        }

        return new DashboardSnapshotResponse(
                orderRecordRepository.count(),
                orderRecordRepository.countByStatus(OrderStatus.EXECUTED),
                orderRecordRepository.countByStatus(OrderStatus.REJECTED),
                orderRecordRepository.sumRequestedNotionalByStatusIn(
                        EnumSet.of(OrderStatus.RECEIVED, OrderStatus.ROUTED, OrderStatus.EXECUTED)
                ),
                orderRecordRepository.sumExecutedNotional(),
                averageLatency,
                state.engineHalted(),
                state.liquidationRequested(),
                Instant.now()
        );
    }
}
