create table market_sentiment_records (
    id bigserial primary key,
    asset varchar(24) not null,
    sentiment_score numeric(6, 4) not null,
    confidence numeric(6, 4) not null,
    source varchar(64) not null,
    headline varchar(512) not null,
    tags varchar(256) not null,
    observed_at timestamp with time zone not null,
    recorded_at timestamp with time zone not null
);

create index idx_market_sentiment_asset_observed_at
    on market_sentiment_records (asset, observed_at desc);
