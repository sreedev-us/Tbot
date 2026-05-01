package com.tbot.execution.controller;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.dto.DashboardSnapshotResponse;
import com.tbot.execution.dto.MarketChartResponse;
import com.tbot.execution.dto.PaperTradeReportResponse;
import com.tbot.execution.service.DashboardService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
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

    @GetMapping("/market-chart")
    public MarketChartResponse marketChart(
            @RequestParam String asset,
            @RequestParam ExchangeName exchange,
            @RequestParam(defaultValue = "1m") String timeframe
    ) {
        return dashboardService.marketChart(asset, exchange, timeframe);
    }

    @GetMapping("/trades")
    public List<PaperTradeReportResponse> recentTrades(
            @RequestParam(required = false) String asset,
            @RequestParam(required = false) ExchangeName exchange
    ) {
        return dashboardService.recentTrades(asset, exchange);
    }
}
