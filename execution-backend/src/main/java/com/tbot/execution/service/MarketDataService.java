package com.tbot.execution.service;

import com.tbot.execution.dto.MarketCandleSyncRequest;
import com.tbot.execution.entity.MarketCandleRecord;
import com.tbot.execution.repository.MarketCandleRecordRepository;
import jakarta.transaction.Transactional;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class MarketDataService {

    private final MarketCandleRecordRepository marketCandleRecordRepository;

    public MarketDataService(MarketCandleRecordRepository marketCandleRecordRepository) {
        this.marketCandleRecordRepository = marketCandleRecordRepository;
    }

    @Transactional
    public void syncCandles(MarketCandleSyncRequest request) {
        request.candles().forEach(candle -> {
            MarketCandleRecord record = marketCandleRecordRepository
                    .findByAssetAndExchangeNameAndTimeframeAndCandleTime(
                            request.asset(),
                            request.exchange(),
                            request.timeframe(),
                            candle.timestamp()
                    )
                    .orElseGet(MarketCandleRecord::new);
            record.setAsset(request.asset());
            record.setExchangeName(request.exchange());
            record.setTimeframe(request.timeframe());
            record.setCandleTime(candle.timestamp());
            record.setOpenPrice(candle.open());
            record.setHighPrice(candle.high());
            record.setLowPrice(candle.low());
            record.setClosePrice(candle.close());
            record.setVolume(candle.volume());
            marketCandleRecordRepository.save(record);
        });
    }

    public List<MarketCandleRecord> fetchRecentCandles(String asset, com.tbot.execution.domain.ExchangeName exchange, String timeframe) {
        return marketCandleRecordRepository.findTop200ByAssetAndExchangeNameAndTimeframeOrderByCandleTimeDesc(
                asset,
                exchange,
                timeframe
        );
    }
}
