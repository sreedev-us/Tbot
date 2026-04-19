package com.tbot.execution.repository;

import com.tbot.execution.entity.SystemTelemetry;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SystemTelemetryRepository extends JpaRepository<SystemTelemetry, Long> {
}
