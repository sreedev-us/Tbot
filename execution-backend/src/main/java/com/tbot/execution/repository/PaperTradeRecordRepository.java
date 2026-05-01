package com.tbot.execution.repository;

import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.TradeStatus;
import com.tbot.execution.entity.PaperTradeRecord;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PaperTradeRecordRepository extends JpaRepository<PaperTradeRecord, Long> {

    Optional<PaperTradeRecord> findFirstByAssetAndExchangeNameAndStatusOrderByOpenedAtDesc(
            String asset,
            ExchangeName exchangeName,
            TradeStatus status
    );

    List<PaperTradeRecord> findTop50ByOrderByOpenedAtDesc();

    List<PaperTradeRecord> findTop50ByAssetAndExchangeNameOrderByOpenedAtDesc(
            String asset,
            ExchangeName exchangeName
    );

    List<PaperTradeRecord> findByStatusOrderByOpenedAtAsc(TradeStatus status);
}
