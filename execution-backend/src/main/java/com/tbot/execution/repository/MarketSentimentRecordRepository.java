package com.tbot.execution.repository;

import com.tbot.execution.entity.MarketSentimentRecord;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MarketSentimentRecordRepository extends JpaRepository<MarketSentimentRecord, Long> {

    Optional<MarketSentimentRecord> findTopByAssetOrderByObservedAtDesc(String asset);
}
