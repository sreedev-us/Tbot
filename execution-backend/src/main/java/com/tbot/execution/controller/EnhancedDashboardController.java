package com.tbot.execution.controller;

import com.tbot.execution.dto.EnhancedDashboardResponse;
import com.tbot.execution.service.DashboardService;
import com.tbot.execution.service.ModelMonitoringService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/v1/dashboard")
@RequiredArgsConstructor
public class EnhancedDashboardController {

    private final DashboardService dashboardService;
    private final ModelMonitoringService modelMonitoringService;

    @GetMapping("/enhanced-snapshot")
    public ResponseEntity<EnhancedDashboardResponse> getEnhancedSnapshot() {
        log.info("Fetching enhanced dashboard snapshot");
        
        try {
            // This will be implemented by wrapping existing DashboardService
            // and adding AI + latency metrics
            return ResponseEntity.ok(new EnhancedDashboardResponse(
                null, null, null, null, null, null
            ));
        } catch (Exception e) {
            log.error("Error fetching enhanced dashboard", e);
            return ResponseEntity.status(500).build();
        }
    }
}
