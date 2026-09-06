package com.tbot.execution.controller;

import com.tbot.execution.service.ModelMonitoringService;
import com.tbot.execution.service.ModelMonitoringService.ModelDriftMetrics;
import com.tbot.execution.service.ModelMonitoringService.ModelHealthMetrics;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/v1/model-monitoring")
@RequiredArgsConstructor
public class ModelMonitoringController {

    private final ModelMonitoringService modelMonitoringService;

    @GetMapping("/health/{modelVersion}")
    public ResponseEntity<ModelHealthMetrics> getModelHealth(
            @PathVariable String modelVersion,
            @RequestParam(defaultValue = "60") int windowMinutes) {
        
        log.info("Checking health for model: {} (window: {}m)", modelVersion, windowMinutes);
        
        try {
            ModelHealthMetrics metrics = modelMonitoringService.getModelHealth(modelVersion, windowMinutes);
            return ResponseEntity.ok(metrics);
        } catch (Exception e) {
            log.error("Error checking model health", e);
            return ResponseEntity.status(500).build();
        }
    }

    @GetMapping("/drift/{modelVersion}")
    public ResponseEntity<ModelDriftMetrics> checkDrift(
            @PathVariable String modelVersion,
            @RequestParam(defaultValue = "60") int windowMinutes) {
        
        log.info("Checking drift for model: {} (window: {}m)", modelVersion, windowMinutes);
        
        try {
            ModelDriftMetrics metrics = modelMonitoringService.detectDrift(modelVersion, windowMinutes);
            return ResponseEntity.ok(metrics);
        } catch (Exception e) {
            log.error("Error detecting drift", e);
            return ResponseEntity.status(500).build();
        }
    }

    @GetMapping("/metrics/{modelVersion}")
    public ResponseEntity<Map<String, Object>> getMetrics(@PathVariable String modelVersion) {
        log.info("Getting metrics for model: {}", modelVersion);
        
        try {
            Map<String, Object> metrics = modelMonitoringService.getModelMetrics(modelVersion);
            return ResponseEntity.ok(metrics);
        } catch (Exception e) {
            log.error("Error getting metrics", e);
            return ResponseEntity.status(500).build();
        }
    }
}
