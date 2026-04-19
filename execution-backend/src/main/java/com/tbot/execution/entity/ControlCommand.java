package com.tbot.execution.entity;

import com.tbot.execution.domain.CommandStatus;
import com.tbot.execution.domain.CommandType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.Instant;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "control_commands")
public class ControlCommand {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private CommandType commandType;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private CommandStatus status;

    @Column(nullable = false, length = 64)
    private String initiatedBy;

    @Column(nullable = false, length = 256)
    private String reason;

    @Column(nullable = false)
    private Instant createdAt;

    private Instant resolvedAt;
}
