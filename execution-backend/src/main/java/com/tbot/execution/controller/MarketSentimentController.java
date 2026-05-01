package com.tbot.execution.controller;

import com.tbot.execution.dto.MarketSentimentIngestRequest;
import com.tbot.execution.dto.MarketSentimentStateResponse;
import com.tbot.execution.service.MarketSentimentService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/sentiment")
public class MarketSentimentController {

    private final MarketSentimentService marketSentimentService;

    public MarketSentimentController(MarketSentimentService marketSentimentService) {
        this.marketSentimentService = marketSentimentService;
    }

    @PostMapping
    public ResponseEntity<MarketSentimentStateResponse> ingest(@Valid @RequestBody MarketSentimentIngestRequest request) {
        return ResponseEntity.accepted().body(marketSentimentService.ingest(request));
    }

    @GetMapping
    public ResponseEntity<MarketSentimentStateResponse> latest(@RequestParam String asset) {
        return marketSentimentService.getLatestState(asset)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.noContent().build());
    }
}
