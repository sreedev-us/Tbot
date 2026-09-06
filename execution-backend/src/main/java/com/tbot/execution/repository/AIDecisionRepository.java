package com.tbot.execution.repository;

import com.tbot.execution.entity.AIDecisionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Repository
public interface AIDecisionRepository extends JpaRepository<AIDecisionRecord, Long> {

    Optional<AIDecisionRecord> findByOrderId(Long orderId);

    @Query("SELECT a FROM AIDecisionRecord a WHERE a.modelVersion = ?1 ORDER BY a.decidedAt DESC LIMIT 100")
    List<AIDecisionRecord> findRecentByModelVersion(String modelVersion);

    @Query("SELECT a FROM AIDecisionRecord a WHERE a.decidedAt >= ?1 AND a.decidedAt < ?2 ORDER BY a.decidedAt DESC")
    List<AIDecisionRecord> findByDecisionDateRange(Instant startInclusive, Instant endExclusive);

    @Query("SELECT a FROM AIDecisionRecord a WHERE a.modelHash = ?1 ORDER BY a.decidedAt DESC LIMIT 1000")
    List<AIDecisionRecord> findByModelHash(String modelHash);

    @Query("SELECT COUNT(a) FROM AIDecisionRecord a WHERE a.rejectReason IS NOT NULL AND a.decidedAt >= ?1")
    long countRejectionsAfter(Instant since);

    @Query("SELECT COUNT(a) FROM AIDecisionRecord a WHERE a.modelVersion = ?1")
    long countByModelVersion(String modelVersion);
}
