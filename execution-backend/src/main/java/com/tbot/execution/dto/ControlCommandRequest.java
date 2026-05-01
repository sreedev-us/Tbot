package com.tbot.execution.dto;

import com.tbot.execution.domain.CommandType;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public record ControlCommandRequest(
        @NotNull CommandType commandType,
        @NotBlank String reason
) {
}
