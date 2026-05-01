package com.tbot.execution;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.tbot.execution.dto.MarketSentimentIngestRequest;
import com.tbot.execution.domain.ExchangeName;
import com.tbot.execution.domain.OrderAction;
import com.tbot.execution.domain.TradeOutcome;
import com.tbot.execution.domain.TradeStatus;
import com.tbot.execution.entity.MarketCandleRecord;
import com.tbot.execution.entity.OrderRecord;
import com.tbot.execution.entity.PaperTradeRecord;
import com.tbot.execution.repository.MarketCandleRecordRepository;
import com.tbot.execution.repository.OrderRecordRepository;
import com.tbot.execution.repository.PaperTradeRecordRepository;
import com.tbot.execution.service.MarketSentimentService;
import java.math.BigDecimal;
import java.time.Instant;
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

    private static final String TEST_BEARER = "Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiI3N2M4NzFlMy05MjdhLTQ2NGUtOWE5YS1jb250cm9sLXVzZXIifQ.";

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private OrderRecordRepository orderRecordRepository;

    @Autowired
    private PaperTradeRecordRepository paperTradeRecordRepository;

    @Autowired
    private MarketCandleRecordRepository marketCandleRecordRepository;

    @Autowired
    private MarketSentimentService marketSentimentService;

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
                  "generatedAt": "2026-04-19T10:15:30Z",
                  "marketPrice": 64000,
                  "stopLossPrice": 63680,
                  "takeProfitPrice": 64640
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
                  "generatedAt": "2026-04-19T10:15:30Z",
                  "marketPrice": 64000,
                  "stopLossPrice": 63680,
                  "takeProfitPrice": 64640
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
                  "reason": "manual kill switch"
                }
                """;

        mockMvc.perform(post("/api/v1/control/commands")
                        .header("Authorization", TEST_BEARER)
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
                  "generatedAt": "2026-04-19T10:15:30Z",
                  "marketPrice": 64000,
                  "stopLossPrice": 64320,
                  "takeProfitPrice": 63360
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(signalPayload))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.reason").value("ENGINE_HALTED"));
    }

    @Test
    void shouldLiquidateOpenDemoTrades() throws Exception {
        OrderRecord order = new OrderRecord();
        order.setSignalId("sig-liquidate-001");
        order.setCorrelationId("corr-liquidate-001");
        order.setAsset("BTC/USDT");
        order.setExchangeName(ExchangeName.BYBIT);
        order.setAction(OrderAction.BUY);
        order.setConfidence(new BigDecimal("0.95"));
        order.setRequestedNotional(new BigDecimal("100"));
        order.setStrategyName("mean-reversion-v1");
        order.setStatus(com.tbot.execution.domain.OrderStatus.EXECUTED);
        order.setGeneratedAt(Instant.parse("2026-04-19T10:15:30Z"));
        order.setReceivedAt(Instant.parse("2026-04-19T10:15:31Z"));
        order.setCompletedAt(Instant.parse("2026-04-19T10:15:35Z"));
        order = orderRecordRepository.save(order);

        PaperTradeRecord trade = new PaperTradeRecord();
        trade.setOrder(order);
        trade.setAsset("BTC/USDT");
        trade.setExchangeName(ExchangeName.BYBIT);
        trade.setAction(OrderAction.BUY);
        trade.setStrategyName("mean-reversion-v1");
        trade.setEntryPrice(new BigDecimal("64000"));
        trade.setQuantity(new BigDecimal("0.00156250"));
        trade.setRequestedNotional(new BigDecimal("100"));
        trade.setStopLossPrice(new BigDecimal("63680"));
        trade.setTakeProfitPrice(new BigDecimal("64640"));
        trade.setConfidence(new BigDecimal("0.95"));
        trade.setStatus(TradeStatus.OPEN);
        trade.setOutcome(TradeOutcome.OPEN);
        trade.setOpenedAt(Instant.parse("2026-04-19T10:15:35Z"));
        paperTradeRecordRepository.save(trade);

        MarketCandleRecord candle = new MarketCandleRecord();
        candle.setAsset("BTC/USDT");
        candle.setExchangeName(ExchangeName.BYBIT);
        candle.setTimeframe("1m");
        candle.setCandleTime(Instant.parse("2026-04-19T10:16:00Z"));
        candle.setOpenPrice(new BigDecimal("64010"));
        candle.setHighPrice(new BigDecimal("64150"));
        candle.setLowPrice(new BigDecimal("63990"));
        candle.setClosePrice(new BigDecimal("64100"));
        candle.setVolume(new BigDecimal("12.5"));
        marketCandleRecordRepository.save(candle);

        String liquidationPayload = """
                {
                  "commandType": "LIQUIDATE_ALL",
                  "reason": "flatten demo book"
                }
                """;

        mockMvc.perform(post("/api/v1/control/commands")
                        .header("Authorization", TEST_BEARER)
                        .contentType("application/json")
                        .content(liquidationPayload))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("CLEARED"));

        PaperTradeRecord liquidated = paperTradeRecordRepository.findByStatusOrderByOpenedAtAsc(TradeStatus.CLOSED)
                .stream()
                .findFirst()
                .orElseThrow();

        org.junit.jupiter.api.Assertions.assertEquals(TradeStatus.CLOSED, liquidated.getStatus());
        org.junit.jupiter.api.Assertions.assertEquals("LIQUIDATE_ALL", liquidated.getCloseReason());
        org.junit.jupiter.api.Assertions.assertEquals(new BigDecimal("64100.00000000"), liquidated.getExitPrice());
    }

    @Test
    void shouldRejectBuySignalsOnSevereNegativeSentiment() throws Exception {
        Instant freshObservedAt = Instant.now();
        marketSentimentService.ingest(new MarketSentimentIngestRequest(
                "BTC/USDT",
                "oracle-test",
                "SEC escalation triggers risk-off move",
                new BigDecimal("-0.85"),
                new BigDecimal("0.91"),
                "BTC,SEC",
                freshObservedAt
        ));

        String payload = """
                {
                  "signalId": "sig-sentiment-reject-001",
                  "correlationId": "corr-sentiment-reject-001",
                  "asset": "BTC/USDT",
                  "exchange": "BYBIT",
                  "action": "BUY",
                  "confidence": 0.92,
                  "requestedNotional": 100,
                  "strategyName": "mean-reversion-v1",
                  "generatedAt": "2026-04-19T10:15:30Z",
                  "marketPrice": 64000,
                  "stopLossPrice": 63680,
                  "takeProfitPrice": 64640
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(payload))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.reason").value("SENTIMENT_CIRCUIT_BREAKER"));
    }

    @Test
    void shouldBoostNotionalOnPositiveAlignedSentiment() throws Exception {
        Instant freshObservedAt = Instant.now();
        marketSentimentService.ingest(new MarketSentimentIngestRequest(
                "BTC/USDT",
                "oracle-test",
                "Fed easing signals support risk assets",
                new BigDecimal("0.75"),
                new BigDecimal("0.88"),
                "BTC,Federal Reserve,Inflation",
                freshObservedAt
        ));

        String payload = """
                {
                  "signalId": "sig-sentiment-boost-001",
                  "correlationId": "corr-sentiment-boost-001",
                  "asset": "BTC/USDT",
                  "exchange": "BYBIT",
                  "action": "BUY",
                  "confidence": 0.92,
                  "requestedNotional": 100,
                  "strategyName": "mean-reversion-v1",
                  "generatedAt": "2026-04-19T10:15:30Z",
                  "marketPrice": 64000,
                  "stopLossPrice": 63680,
                  "takeProfitPrice": 64640
                }
                """;

        mockMvc.perform(post("/api/v1/execute")
                        .contentType("application/json")
                        .content(payload))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.approvedNotional").value(120.0))
                .andExpect(jsonPath("$.sentimentScore").value(0.75))
                .andExpect(jsonPath("$.sentimentConfidence").value(0.88));
    }
}
