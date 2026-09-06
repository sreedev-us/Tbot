-- Create AI Decisions table for storing model decisions with full trace
CREATE TABLE IF NOT EXISTS ai_decisions (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(id) ON DELETE RESTRICT,
    model_version VARCHAR(64) NOT NULL,
    model_hash VARCHAR(64) NOT NULL,
    feature_version VARCHAR(32) NOT NULL,
    input_timestamp TIMESTAMPTZ NOT NULL,
    decision VARCHAR(8) NOT NULL CHECK (decision IN ('BUY', 'SELL', 'HOLD')),
    confidence NUMERIC(10, 6) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    features_json JSONB NOT NULL,
    server_ai_result_json JSONB,
    local_ai_result_json JSONB NOT NULL,
    risk_decision_json JSONB,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMPTZ,
    reject_reason VARCHAR(256),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ai_decisions_order_id ON ai_decisions(order_id);
CREATE INDEX idx_ai_decisions_model_version ON ai_decisions(model_version);
CREATE INDEX idx_ai_decisions_decided_at ON ai_decisions(decided_at DESC);
CREATE INDEX idx_ai_decisions_model_hash ON ai_decisions(model_hash);

COMMENT ON TABLE ai_decisions IS 'Stores all AI model decisions for trading signals with full feature/decision trace for debugging and monitoring';
COMMENT ON COLUMN ai_decisions.model_version IS 'Semantic version of the AI model (e.g., local-v1.0.0)';
COMMENT ON COLUMN ai_decisions.model_hash IS 'SHA256 hash of model weights for exact reproducibility';
COMMENT ON COLUMN ai_decisions.feature_version IS 'Version of feature engineering pipeline used';
COMMENT ON COLUMN ai_decisions.features_json IS 'All input features used for the decision as JSON';
COMMENT ON COLUMN ai_decisions.server_ai_result_json IS 'Result from server-side analysis (trend, volatility, regime)';
COMMENT ON COLUMN ai_decisions.local_ai_result_json IS 'Raw output from local decision model including probabilities';
COMMENT ON COLUMN ai_decisions.risk_decision_json IS 'Risk manager decision: approved/rejected with rationale';
