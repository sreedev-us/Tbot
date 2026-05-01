package com.tbot.execution.controller;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.dto.PaperTradeCloseRequest;
import com.tbot.execution.dto.PaperTradeReportResponse;
import com.tbot.execution.dto.PaperTradeStateResponse;
import com.tbot.execution.service.PaperTradeService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/paper-trades")
public class PaperTradeController {

    private final PaperTradeService paperTradeService;

    public PaperTradeController(PaperTradeService paperTradeService) {
        this.paperTradeService = paperTradeService;
    }

    @GetMapping("/open")
    public ResponseEntity<PaperTradeStateResponse> openTrade(
            @RequestParam String asset,
            @RequestParam ExchangeName exchange
    ) {
        PaperTradeStateResponse trade = paperTradeService.findOpenTrade(asset, exchange);
        return trade == null ? ResponseEntity.noContent().build() : ResponseEntity.ok(trade);
    }

    @PostMapping("/close")
    public ResponseEntity<PaperTradeReportResponse> closeTrade(@Valid @RequestBody PaperTradeCloseRequest request) {
        return ResponseEntity.ok(paperTradeService.closeTrade(request));
    }
}
