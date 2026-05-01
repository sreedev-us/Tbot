package com.tbot.execution.controller;

import com.tbot.execution.dto.MarketCandleSyncRequest;
import com.tbot.execution.service.MarketDataService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/market-data")
public class MarketDataController {

    private final MarketDataService marketDataService;

    public MarketDataController(MarketDataService marketDataService) {
        this.marketDataService = marketDataService;
    }

    @PostMapping("/candles")
    public ResponseEntity<Void> syncCandles(@Valid @RequestBody MarketCandleSyncRequest request) {
        marketDataService.syncCandles(request);
        return ResponseEntity.accepted().build();
    }
}
