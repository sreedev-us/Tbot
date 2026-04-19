package com.tbot.execution.controller;

import com.tbot.execution.dto.ControlCommandRequest;
import com.tbot.execution.dto.ControlCommandResponse;
import com.tbot.execution.dto.ControlStateResponse;
import com.tbot.execution.service.ControlCommandService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/control")
public class ControlController {

    private final ControlCommandService controlCommandService;

    public ControlController(ControlCommandService controlCommandService) {
        this.controlCommandService = controlCommandService;
    }

    @GetMapping("/state")
    public ControlStateResponse state() {
        return controlCommandService.getState();
    }

    @GetMapping("/commands")
    public List<ControlCommandResponse> commands() {
        return controlCommandService.recentCommands();
    }

    @PostMapping("/commands")
    public ResponseEntity<ControlCommandResponse> issue(@Valid @RequestBody ControlCommandRequest request) {
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(controlCommandService.issue(request));
    }
}
