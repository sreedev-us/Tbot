package com.tbot.execution.dto;

import java.time.Instant;

public record ControlStateResponse(
        boolean engineHalted,
        boolean liquidationRequested,
        Instant asOf
) {
}
