package com.tbot.execution.service;

import com.tbot.execution.dto.MarketSentimentIngestRequest;
import com.tbot.execution.dto.MarketSentimentStateResponse;
import com.tbot.execution.entity.MarketSentimentRecord;
import com.tbot.execution.repository.MarketSentimentRecordRepository;
import java.time.Instant;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
public class MarketSentimentService {

    private final MarketSentimentRecordRepository marketSentimentRecordRepository;
    private final TelemetryService telemetryService;

    public MarketSentimentService(
            MarketSentimentRecordRepository marketSentimentRecordRepository,
            TelemetryService telemetryService
    ) {
        this.marketSentimentRecordRepository = marketSentimentRecordRepository;
        this.telemetryService = telemetryService;
    }

    public MarketSentimentStateResponse ingest(MarketSentimentIngestRequest request) {
        MarketSentimentRecord record = new MarketSentimentRecord();
        record.setAsset(request.asset());
        record.setSource(request.source());
        record.setHeadline(request.headline());
        record.setSentimentScore(request.sentimentScore());
        record.setConfidence(request.confidence());
        record.setTags(request.tags());
        record.setObservedAt(request.observedAt());
        record.setRecordedAt(Instant.now());
        MarketSentimentRecord saved = marketSentimentRecordRepository.save(record);
        telemetryService.record(
                "sentiment",
                "latest_score",
                request.sentimentScore(),
                "score",
                "asset=" + request.asset() + ",source=" + request.source()
        );
        return toResponse(saved);
    }

    public Optional<MarketSentimentRecord> getLatestRecord(String asset) {
        return marketSentimentRecordRepository.findTopByAssetOrderByObservedAtDesc(asset);
    }

    public Optional<MarketSentimentStateResponse> getLatestState(String asset) {
        return getLatestRecord(asset).map(this::toResponse);
    }

    private MarketSentimentStateResponse toResponse(MarketSentimentRecord record) {
        return new MarketSentimentStateResponse(
                record.getAsset(),
                record.getSentimentScore(),
                record.getConfidence(),
                record.getSource(),
                record.getHeadline(),
                record.getTags(),
                record.getObservedAt()
        );
    }
}
