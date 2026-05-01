package com.tbot.execution.service;

import com.tbot.execution.domain.CommandType;
import jakarta.annotation.PreDestroy;
import java.math.BigDecimal;
import org.springframework.stereotype.Component;

@Component
public class ShutdownGuardService {

    private final ControlCommandService controlCommandService;
    private final TelemetryService telemetryService;

    public ShutdownGuardService(
            ControlCommandService controlCommandService,
            TelemetryService telemetryService
    ) {
        this.controlCommandService = controlCommandService;
        this.telemetryService = telemetryService;
    }

    @PreDestroy
    public void onShutdown() {
        telemetryService.record(
                "system",
                "backend_shutdown",
                BigDecimal.ONE,
                "count",
                "source=spring-boot"
        );
        controlCommandService.issueSystemCommand(
                CommandType.HALT_ENGINE,
                "Automatic halt during JVM shutdown."
        );
        controlCommandService.issueSystemCommand(
                CommandType.LIQUIDATE_ALL,
                "Automatic liquidation request during JVM shutdown."
        );
    }
}
