create table control_commands (
    id bigserial primary key,
    command_type varchar(32) not null,
    status varchar(16) not null,
    initiated_by varchar(64) not null,
    reason varchar(256) not null,
    created_at timestamp with time zone not null,
    resolved_at timestamp with time zone
);
