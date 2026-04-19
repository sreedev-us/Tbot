package com.tbot.execution.repository;

import com.tbot.execution.entity.ExecutionRecord;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ExecutionRecordRepository extends JpaRepository<ExecutionRecord, Long> {
}
