create table paper_trades (
    id bigserial primary key,
    order_id bigint not null unique references orders(id),
    asset varchar(24) not null,
    exchange_name varchar(16) not null,
    action varchar(8) not null,
    strategy_name varchar(64) not null,
    entry_price numeric(18, 8) not null,
    exit_price numeric(18, 8),
    quantity numeric(18, 8) not null,
    requested_notional numeric(18, 8) not null,
    stop_loss_price numeric(18, 8) not null,
    take_profit_price numeric(18, 8) not null,
    confidence numeric(10, 6) not null,
    status varchar(16) not null,
    outcome varchar(16) not null,
    opened_at timestamp with time zone not null,
    closed_at timestamp with time zone,
    realized_pnl numeric(18, 8),
    realized_pnl_pct numeric(10, 4),
    close_reason varchar(32)
);

create table market_candles (
    id bigserial primary key,
    asset varchar(24) not null,
    exchange_name varchar(16) not null,
    timeframe varchar(8) not null,
    candle_time timestamp with time zone not null,
    open_price numeric(18, 8) not null,
    high_price numeric(18, 8) not null,
    low_price numeric(18, 8) not null,
    close_price numeric(18, 8) not null,
    volume numeric(18, 8) not null,
    constraint uk_market_candle unique (asset, exchange_name, timeframe, candle_time)
);
