package com.tbot.execution.service;

import com.tbot.execution.entity.SystemTelemetry;
import com.tbot.execution.repository.SystemTelemetryRepository;
import java.math.BigDecimal;
import java.time.Instant;
import org.springframework.stereotype.Service;

@Service
public class TelemetryService {

    private final SystemTelemetryRepository telemetryRepository;

    public TelemetryService(SystemTelemetryRepository telemetryRepository) {
        this.telemetryRepository = telemetryRepository;
    }

    public void record(String group, String name, BigDecimal value, String unit, String tags) {
        SystemTelemetry telemetry = new SystemTelemetry();
        telemetry.setMetricGroup(group);
        telemetry.setMetricName(name);
        telemetry.setMetricValue(value);
        telemetry.setMetricUnit(unit);
        telemetry.setTags(tags);
        telemetry.setRecordedAt(Instant.now());
        telemetryRepository.save(telemetry);
    }
}
