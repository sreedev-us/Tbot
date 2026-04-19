package com.tbot.execution.repository;

import com.tbot.execution.domain.OrderStatus;
import com.tbot.execution.entity.OrderRecord;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.Collection;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

public interface OrderRecordRepository extends JpaRepository<OrderRecord, Long> {

    Optional<OrderRecord> findBySignalId(String signalId);

    @Query("""
            select coalesce(sum(o.requestedNotional), 0)
            from OrderRecord o
            where o.status in :activeStatuses
            """)
    BigDecimal sumRequestedNotionalByStatusIn(Collection<OrderStatus> activeStatuses);

    @Query("""
            select coalesce(sum(o.requestedNotional), 0)
            from OrderRecord o
            where o.status = com.tbot.execution.domain.OrderStatus.REJECTED
              and o.receivedAt >= :startOfDay
            """)
    BigDecimal sumRejectedNotionalSince(Instant startOfDay);

    long countByStatus(OrderStatus status);

    @Query("""
            select coalesce(sum(o.requestedNotional), 0)
            from OrderRecord o
            where o.status = com.tbot.execution.domain.OrderStatus.EXECUTED
            """)
    BigDecimal sumExecutedNotional();
}
