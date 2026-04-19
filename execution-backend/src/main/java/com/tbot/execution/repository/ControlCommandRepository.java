package com.tbot.execution.repository;

import com.tbot.execution.domain.CommandStatus;
import com.tbot.execution.domain.CommandType;
import com.tbot.execution.entity.ControlCommand;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ControlCommandRepository extends JpaRepository<ControlCommand, Long> {

    Optional<ControlCommand> findTopByCommandTypeAndStatusOrderByCreatedAtDesc(CommandType commandType, CommandStatus status);

    List<ControlCommand> findTop20ByOrderByCreatedAtDesc();
}
