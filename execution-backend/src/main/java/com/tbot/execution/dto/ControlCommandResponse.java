package com.tbot.execution.dto;

import com.tbot.execution.domain.CommandStatus;
import com.tbot.execution.domain.CommandType;
import java.time.Instant;

public record ControlCommandResponse(
        Long id,
        CommandType commandType,
        CommandStatus status,
        String initiatedBy,
        String reason,
        Instant createdAt,
        Instant resolvedAt
) {
}
