package com.tbot.execution.dto;

import java.util.List;

public record MarketChartResponse(
        String asset,
        String exchange,
        String timeframe,
        List<MarketChartPointResponse> candles,
        List<TradeMarkerResponse> trades
) {
}
