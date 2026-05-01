package com.tbot.execution.service;

import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.dto.ControlStateResponse;
import com.tbot.execution.dto.DashboardSnapshotResponse;
import com.tbot.execution.dto.MarketChartPointResponse;
import com.tbot.execution.dto.MarketChartResponse;
import com.tbot.execution.dto.PaperTradeReportResponse;
import com.tbot.execution.dto.TradeMarkerResponse;
import com.tbot.execution.entity.MarketCandleRecord;
import com.tbot.execution.entity.PaperTradeRecord;
import com.tbot.execution.repository.ExecutionRecordRepository;
import com.tbot.execution.repository.OrderRecordRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class DashboardService {

    private final OrderRecordRepository orderRecordRepository;
    private final ExecutionRecordRepository executionRecordRepository;
    private final ControlCommandService controlCommandService;
    private final MarketDataService marketDataService;
    private final PaperTradeService paperTradeService;

    public DashboardService(
            OrderRecordRepository orderRecordRepository,
            ExecutionRecordRepository executionRecordRepository,
            ControlCommandService controlCommandService,
            MarketDataService marketDataService,
            PaperTradeService paperTradeService
    ) {
        this.orderRecordRepository = orderRecordRepository;
        this.executionRecordRepository = executionRecordRepository;
        this.controlCommandService = controlCommandService;
        this.marketDataService = marketDataService;
        this.paperTradeService = paperTradeService;
    }

    @Transactional
    public DashboardSnapshotResponse snapshot() {
        ControlStateResponse state = controlCommandService.getState();
        BigDecimal averageLatency = executionRecordRepository.findAll()
                .stream()
                .map(execution -> BigDecimal.valueOf(
                        java.time.Duration.between(
                                execution.getOrder().getGeneratedAt(),
                                execution.getFillConfirmedAt()
                        ).toMillis()
                ))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        long executionCount = executionRecordRepository.count();
        if (executionCount > 0) {
            averageLatency = averageLatency.divide(BigDecimal.valueOf(executionCount), 2, RoundingMode.HALF_UP);
        }

        return new DashboardSnapshotResponse(
                orderRecordRepository.count(),
                orderRecordRepository.countByStatus(OrderStatus.EXECUTED),
                orderRecordRepository.countByStatus(OrderStatus.REJECTED),
                orderRecordRepository.sumRequestedNotionalByStatusIn(
                        EnumSet.of(OrderStatus.RECEIVED, OrderStatus.ROUTED, OrderStatus.EXECUTED)
                ),
                orderRecordRepository.sumExecutedNotional(),
                averageLatency,
                state.engineHalted(),
                state.liquidationRequested(),
                Instant.now()
        );
    }

    public MarketChartResponse marketChart(String asset, ExchangeName exchange, String timeframe) {
        List<MarketCandleRecord> candles = marketDataService.fetchRecentCandles(asset, exchange, timeframe)
                .stream()
                .sorted(Comparator.comparing(MarketCandleRecord::getCandleTime))
                .toList();
        List<PaperTradeReportResponse> trades = paperTradeService.recentTrades(asset, exchange);

        return new MarketChartResponse(
                asset,
                exchange.name(),
                timeframe,
                candles.stream()
                        .map(candle -> new MarketChartPointResponse(
                                candle.getCandleTime(),
                                candle.getOpenPrice(),
                                candle.getHighPrice(),
                                candle.getLowPrice(),
                                candle.getClosePrice(),
                                candle.getVolume()
                        ))
                        .toList(),
                trades.stream()
                        .map(this::toMarker)
                        .toList()
        );
    }

    public List<PaperTradeReportResponse> recentTrades(String asset, ExchangeName exchange) {
        return paperTradeService.recentTrades(asset, exchange);
    }

    private TradeMarkerResponse toMarker(PaperTradeReportResponse trade) {
        return new TradeMarkerResponse(
                trade.tradeId(),
                trade.signalId(),
                trade.action(),
                trade.status(),
                trade.outcome(),
                trade.openedAt(),
                trade.closedAt(),
                trade.entryPrice(),
                trade.exitPrice(),
                trade.realizedPnl(),
                trade.realizedPnlPct(),
                trade.closeReason()
        );
    }
}
