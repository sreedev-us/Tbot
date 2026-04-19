package com.tbot.execution.controller;

import com.tbot.execution.dto.ExecutionDecisionResponse;
import com.tbot.execution.dto.ExecutionSignalRequest;
import com.tbot.execution.service.ExecutionOrchestratorService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class ExecutionController {

    private final ExecutionOrchestratorService executionOrchestratorService;

    public ExecutionController(ExecutionOrchestratorService executionOrchestratorService) {
        this.executionOrchestratorService = executionOrchestratorService;
    }

    @PostMapping("/execute")
    public ResponseEntity<ExecutionDecisionResponse> execute(@Valid @RequestBody ExecutionSignalRequest request) {
        ExecutionDecisionResponse response = executionOrchestratorService.processSignal(request);
        HttpStatus status = response.status().name().equals("REJECTED") ? HttpStatus.UNPROCESSABLE_ENTITY : HttpStatus.ACCEPTED;
        return ResponseEntity.status(status).body(response);
    }
}
