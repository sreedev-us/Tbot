package com.tbot.execution.service;

import com.tbot.execution.domain.CommandStatus;
import com.tbot.execution.domain.CommandType;
import com.tbot.execution.dto.ControlCommandRequest;
import com.tbot.execution.dto.ControlCommandResponse;
import com.tbot.execution.dto.ControlStateResponse;
import com.tbot.execution.entity.ControlCommand;
import com.tbot.execution.repository.ControlCommandRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class ControlCommandService {

    private final ControlCommandRepository controlCommandRepository;
    private final TelemetryService telemetryService;

    public ControlCommandService(
            ControlCommandRepository controlCommandRepository,
            TelemetryService telemetryService
    ) {
        this.controlCommandRepository = controlCommandRepository;
        this.telemetryService = telemetryService;
    }

    @Transactional
    public ControlCommandResponse issue(ControlCommandRequest request) {
        if (request.commandType() == CommandType.RESUME_ENGINE) {
            clearActiveCommand(CommandType.HALT_ENGINE);
        }

        if (request.commandType() == CommandType.LIQUIDATE_ALL) {
            clearActiveCommand(CommandType.LIQUIDATE_ALL);
        }

        ControlCommand command = new ControlCommand();
        command.setCommandType(request.commandType());
        command.setStatus(CommandStatus.ACTIVE);
        command.setInitiatedBy(request.initiatedBy());
        command.setReason(request.reason());
        command.setCreatedAt(Instant.now());
        ControlCommand saved = controlCommandRepository.save(command);

        telemetryService.record(
                "control",
                request.commandType().name().toLowerCase(),
                BigDecimal.ONE,
                "count",
                "initiatedBy=" + request.initiatedBy()
        );

        return toResponse(saved);
    }

    public ControlStateResponse getState() {
        boolean halted = controlCommandRepository
                .findTopByCommandTypeAndStatusOrderByCreatedAtDesc(CommandType.HALT_ENGINE, CommandStatus.ACTIVE)
                .isPresent();
        boolean liquidation = controlCommandRepository
                .findTopByCommandTypeAndStatusOrderByCreatedAtDesc(CommandType.LIQUIDATE_ALL, CommandStatus.ACTIVE)
                .isPresent();
        return new ControlStateResponse(halted, liquidation, Instant.now());
    }

    public List<ControlCommandResponse> recentCommands() {
        return controlCommandRepository.findTop20ByOrderByCreatedAtDesc()
                .stream()
                .map(this::toResponse)
                .toList();
    }

    @Transactional
    public void clearActiveCommand(CommandType type) {
        controlCommandRepository.findTopByCommandTypeAndStatusOrderByCreatedAtDesc(type, CommandStatus.ACTIVE)
                .ifPresent(command -> {
                    command.setStatus(CommandStatus.CLEARED);
                    command.setResolvedAt(Instant.now());
                });
    }

    private ControlCommandResponse toResponse(ControlCommand command) {
        return new ControlCommandResponse(
                command.getId(),
                command.getCommandType(),
                command.getStatus(),
                command.getInitiatedBy(),
                command.getReason(),
                command.getCreatedAt(),
                command.getResolvedAt()
        );
    }
}
