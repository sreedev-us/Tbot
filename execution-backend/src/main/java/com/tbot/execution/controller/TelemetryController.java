package com.tbot.execution.controller;

import com.tbot.execution.dto.TelemetryIngestRequest;
import com.tbot.execution.service.TelemetryService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/telemetry")
public class TelemetryController {

    private final TelemetryService telemetryService;

    public TelemetryController(TelemetryService telemetryService) {
        this.telemetryService = telemetryService;
    }

    @PostMapping
    public ResponseEntity<Void> ingest(@Valid @RequestBody TelemetryIngestRequest request) {
        telemetryService.record(
                request.metricGroup(),
                request.metricName(),
                request.metricValue(),
                request.metricUnit(),
                request.tags()
        );
        return ResponseEntity.accepted().build();
    }
}
