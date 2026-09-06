package com.tbot.execution.controller;

import com.tbot.execution.dto.ServerAIAnalysisRequest;
import com.tbot.execution.dto.ServerAIAnalysisResponse;
import com.tbot.execution.service.MarketAnalysisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/v1/analysis")
@RequiredArgsConstructor
public class ServerAIController {

    private final MarketAnalysisService marketAnalysisService;

    @PostMapping
    public ResponseEntity<ServerAIAnalysisResponse> analyzeMarket(@RequestBody ServerAIAnalysisRequest request) {
        log.info("Received market analysis request for asset: {}", request.asset());

        try {
            ServerAIAnalysisResponse analysis = marketAnalysisService.analyzeMarketContext(request);
            log.debug("Market analysis complete: trend={}, volatility={}, regime={}", 
                analysis.trend(), analysis.volatility(), analysis.regime());
            return ResponseEntity.ok(analysis);
        } catch (Exception e) {
            log.error("Error analyzing market context", e);
            return ResponseEntity.status(500).build();
        }
    }
}
