package com.tbot.execution.controller;

import com.tbot.execution.dto.DashboardSnapshotResponse;
import com.tbot.execution.service.DashboardService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/dashboard")
public class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @GetMapping("/snapshot")
    public DashboardSnapshotResponse snapshot() {
        return dashboardService.snapshot();
    }
}
