package com.tbot.execution;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class ExecutionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @DynamicPropertySource
    static void overrideProps(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", () -> "jdbc:h2:mem:tbot-" + java.util.UUID.randomUUID() + ";MODE=PostgreSQL");
        registry.add("spring.datasource.username", () -> "sa");
        registry.add("spring.datasource.password", () -> "");
    }

    @Test
    void shouldAcceptQualifiedSignal() throws Exception {
        String payload = """
                {
                  "signalId": "sig-001",
                  "correlationId": "corr-001",
                  "asset": "BTC/USDT",
                  "exchange": "BYBIT",
                  "action": "BUY",
                  "confidence": 0.92,
                  "requestedNotional": 100,
                  "strategyName": "mean-reversion-v1",
                  "generatedAt": "2026-04-19T10:15:30Z"
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(payload))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("EXECUTED"))
                .andExpect(jsonPath("$.reason").value("SIMULATED_FILL"));
    }

    @Test
    void shouldRejectLowConfidenceSignal() throws Exception {
        String payload = """
                {
                  "signalId": "sig-002",
                  "correlationId": "corr-002",
                  "asset": "BTC/USDT",
                  "exchange": "BINANCE",
                  "action": "BUY",
                  "confidence": 0.20,
                  "requestedNotional": 100,
                  "strategyName": "mean-reversion-v1",
                  "generatedAt": "2026-04-19T10:15:30Z"
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(payload))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.status").value("REJECTED"))
                .andExpect(jsonPath("$.reason").value("CONFIDENCE_BELOW_THRESHOLD"));
    }

    @Test
    void shouldRejectSignalsWhenEngineIsHalted() throws Exception {
        String haltPayload = """
                {
                  "commandType": "HALT_ENGINE",
                  "initiatedBy": "ops@test",
                  "reason": "manual kill switch"
                }
                """;

        mockMvc.perform(post("/api/v1/control/commands")
                        .contentType("application/json")
                        .content(haltPayload))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.commandType").value("HALT_ENGINE"));

        String signalPayload = """
                {
                  "signalId": "sig-003",
                  "correlationId": "corr-003",
                  "asset": "BTC/USDT",
                  "exchange": "KRAKEN",
                  "action": "SELL",
                  "confidence": 0.90,
                  "requestedNotional": 100,
                  "strategyName": "mean-reversion-v1",
                  "generatedAt": "2026-04-19T10:15:30Z"
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(signalPayload))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.reason").value("ENGINE_HALTED"));
    }
}
