create table orders (
    id bigserial primary key,
    signal_id varchar(64) not null unique,
    correlation_id varchar(32) not null,
    asset varchar(24) not null,
    exchange_name varchar(16) not null,
    action varchar(8) not null,
    confidence numeric(10, 6) not null,
    requested_notional numeric(18, 8) not null,
    strategy_name varchar(64) not null,
    status varchar(16) not null,
    generated_at timestamp with time zone not null,
    received_at timestamp with time zone not null,
    routed_at timestamp with time zone,
    completed_at timestamp with time zone,
    rejection_reason varchar(256)
);

create table executions (
    id bigserial primary key,
    order_id bigint not null references orders(id),
    venue_order_id varchar(32) not null,
    exchange_name varchar(16) not null,
    action varchar(8) not null,
    status varchar(16) not null,
    requested_notional numeric(18, 8) not null,
    executed_notional numeric(18, 8) not null,
    slippage_fee numeric(18, 8) not null,
    exchange_ack_at timestamp with time zone not null,
    fill_confirmed_at timestamp with time zone not null
);

create table system_telemetry (
    id bigserial primary key,
    metric_group varchar(32) not null,
    metric_name varchar(64) not null,
    metric_value numeric(18, 8) not null,
    metric_unit varchar(16) not null,
    tags varchar(256) not null,
    recorded_at timestamp with time zone not null
);
