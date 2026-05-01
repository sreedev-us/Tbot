package com.tbot.execution.service;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import com.tbot.execution.dto.PaperTradeCloseRequest;
import com.tbot.execution.dto.PaperTradeReportResponse;
import com.tbot.execution.dto.PaperTradeStateResponse;
import com.tbot.execution.entity.OrderRecord;
import com.tbot.execution.entity.PaperTradeRecord;
import com.tbot.execution.repository.MarketCandleRecordRepository;
import com.tbot.execution.repository.PaperTradeRecordRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class PaperTradeService {

    private final PaperTradeRecordRepository paperTradeRecordRepository;
    private final MarketCandleRecordRepository marketCandleRecordRepository;

    public PaperTradeService(
            PaperTradeRecordRepository paperTradeRecordRepository,
            MarketCandleRecordRepository marketCandleRecordRepository
    ) {
        this.paperTradeRecordRepository = paperTradeRecordRepository;
        this.marketCandleRecordRepository = marketCandleRecordRepository;
    }

    @Transactional
    public PaperTradeRecord openTrade(
            OrderRecord order,
            BigDecimal entryPrice,
            BigDecimal stopLossPrice,
            BigDecimal takeProfitPrice
    ) {
        BigDecimal quantity = order.getRequestedNotional()
                .divide(entryPrice, 8, RoundingMode.HALF_UP);
        PaperTradeRecord trade = new PaperTradeRecord();
        trade.setOrder(order);
        trade.setAsset(order.getAsset());
        trade.setExchangeName(order.getExchangeName());
        trade.setAction(order.getAction());
        trade.setStrategyName(order.getStrategyName());
        trade.setEntryPrice(entryPrice);
        trade.setQuantity(quantity);
        trade.setRequestedNotional(order.getRequestedNotional());
        trade.setStopLossPrice(stopLossPrice);
        trade.setTakeProfitPrice(takeProfitPrice);
        trade.setConfidence(order.getConfidence());
        trade.setStatus(TradeStatus.OPEN);
        trade.setOutcome(TradeOutcome.OPEN);
        trade.setOpenedAt(order.getCompletedAt() != null ? order.getCompletedAt() : order.getGeneratedAt());
        return paperTradeRecordRepository.save(trade);
    }

    @Transactional
    public PaperTradeStateResponse findOpenTrade(String asset, ExchangeName exchange) {
        return paperTradeRecordRepository
                .findFirstByAssetAndExchangeNameAndStatusOrderByOpenedAtDesc(asset, exchange, TradeStatus.OPEN)
                .map(this::toState)
                .orElse(null);
    }

    @Transactional
    public PaperTradeReportResponse closeTrade(PaperTradeCloseRequest request) {
        PaperTradeRecord trade = paperTradeRecordRepository
                .findFirstByAssetAndExchangeNameAndStatusOrderByOpenedAtDesc(
                        request.asset(),
                        request.exchange(),
                        TradeStatus.OPEN
                )
                .orElseThrow(() -> new IllegalArgumentException("No open paper trade found for asset/exchange."));
        return closeTrade(trade, request.exitPrice(), request.closedAt(), request.closeReason());
    }

    @Transactional
    public List<PaperTradeReportResponse> recentTrades(String asset, ExchangeName exchange) {
        List<PaperTradeRecord> trades = asset != null && exchange != null
                ? paperTradeRecordRepository.findTop50ByAssetAndExchangeNameOrderByOpenedAtDesc(asset, exchange)
                : paperTradeRecordRepository.findTop50ByOrderByOpenedAtDesc();
        return trades.stream().map(this::toReport).toList();
    }

    @Transactional
    public List<PaperTradeReportResponse> liquidateAllOpenTrades(String closeReason) {
        List<PaperTradeRecord> openTrades = paperTradeRecordRepository.findByStatusOrderByOpenedAtAsc(TradeStatus.OPEN);
        List<PaperTradeReportResponse> closedTrades = new ArrayList<>();

        for (PaperTradeRecord trade : openTrades) {
            BigDecimal exitPrice = marketCandleRecordRepository
                    .findTopByAssetAndExchangeNameOrderByCandleTimeDesc(trade.getAsset(), trade.getExchangeName())
                    .map(candle -> candle.getClosePrice())
                    .orElse(trade.getEntryPrice());
            closedTrades.add(closeTrade(trade, exitPrice, Instant.now(), closeReason));
        }

        return closedTrades;
    }

    public PaperTradeStateResponse toState(PaperTradeRecord trade) {
        return new PaperTradeStateResponse(
                trade.getId(),
                trade.getOrder().getSignalId(),
                trade.getAsset(),
                trade.getExchangeName(),
                trade.getAction(),
                trade.getStatus(),
                trade.getOutcome(),
                trade.getEntryPrice(),
                trade.getStopLossPrice(),
                trade.getTakeProfitPrice(),
                trade.getQuantity(),
                trade.getOpenedAt()
        );
    }

    private PaperTradeReportResponse closeTrade(
            PaperTradeRecord trade,
            BigDecimal exitPrice,
            Instant closedAt,
            String closeReason
    ) {
        trade.setExitPrice(exitPrice);
        trade.setClosedAt(closedAt);
        trade.setCloseReason(closeReason);
        trade.setStatus(TradeStatus.CLOSED);

        BigDecimal priceDelta = exitPrice.subtract(trade.getEntryPrice());
        if (trade.getAction() == OrderAction.SELL) {
            priceDelta = trade.getEntryPrice().subtract(exitPrice);
        }

        BigDecimal realizedPnl = priceDelta.multiply(trade.getQuantity()).setScale(8, RoundingMode.HALF_UP);
        BigDecimal realizedPnlPct = BigDecimal.ZERO;
        if (trade.getRequestedNotional().compareTo(BigDecimal.ZERO) > 0) {
            realizedPnlPct = realizedPnl
                    .divide(trade.getRequestedNotional(), 8, RoundingMode.HALF_UP)
                    .multiply(BigDecimal.valueOf(100))
                    .setScale(4, RoundingMode.HALF_UP);
        }

        trade.setRealizedPnl(realizedPnl);
        trade.setRealizedPnlPct(realizedPnlPct);
        trade.setOutcome(realizedPnl.compareTo(BigDecimal.ZERO) > 0
                ? TradeOutcome.PROFIT
                : realizedPnl.compareTo(BigDecimal.ZERO) < 0
                ? TradeOutcome.LOSS
                : TradeOutcome.BREAKEVEN);

        return toReport(paperTradeRecordRepository.save(trade));
    }

    public PaperTradeReportResponse toReport(PaperTradeRecord trade) {
        return new PaperTradeReportResponse(
                trade.getId(),
                trade.getOrder().getSignalId(),
                trade.getAsset(),
                trade.getExchangeName(),
                trade.getAction(),
                trade.getStrategyName(),
                trade.getConfidence(),
                trade.getRequestedNotional(),
                trade.getQuantity(),
                trade.getEntryPrice(),
                trade.getExitPrice(),
                trade.getStopLossPrice(),
                trade.getTakeProfitPrice(),
                trade.getOpenedAt(),
                trade.getClosedAt(),
                trade.getStatus(),
                trade.getOutcome(),
                trade.getRealizedPnl(),
                trade.getRealizedPnlPct(),
                trade.getCloseReason()
        );
    }
}
