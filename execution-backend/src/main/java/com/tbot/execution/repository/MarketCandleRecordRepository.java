package com.tbot.execution.repository;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.entity.MarketCandleRecord;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MarketCandleRecordRepository extends JpaRepository<MarketCandleRecord, Long> {

    Optional<MarketCandleRecord> findByAssetAndExchangeNameAndTimeframeAndCandleTime(
            String asset,
            ExchangeName exchangeName,
            String timeframe,
            Instant candleTime
    );

    List<MarketCandleRecord> findTop200ByAssetAndExchangeNameAndTimeframeOrderByCandleTimeDesc(
            String asset,
            ExchangeName exchangeName,
            String timeframe
    );

    Optional<MarketCandleRecord> findTopByAssetAndExchangeNameOrderByCandleTimeDesc(
            String asset,
            ExchangeName exchangeName
    );
}
