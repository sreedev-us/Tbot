-- Replace this UUID before executing the script in Supabase SQL editor.
-- It must match the single authenticated admin user allowed to stream data.
--
-- Example:
--   auth.uid() = '11111111-2222-3333-4444-555555555555'

do $$
begin
    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'orders'
    ) then
        alter publication supabase_realtime add table public.orders;
    end if;

    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'executions'
    ) then
        alter publication supabase_realtime add table public.executions;
    end if;

    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'system_telemetry'
    ) then
        alter publication supabase_realtime add table public.system_telemetry;
    end if;

    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'paper_trades'
    ) then
        alter publication supabase_realtime add table public.paper_trades;
    end if;

    if not exists (
        select 1
        from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'public'
          and tablename = 'control_commands'
    ) then
        alter publication supabase_realtime add table public.control_commands;
    end if;
end
$$;

alter table public.orders enable row level security;
alter table public.executions enable row level security;
alter table public.system_telemetry enable row level security;
alter table public.control_commands enable row level security;
alter table public.paper_trades enable row level security;

drop policy if exists "anon read orders" on public.orders;
drop policy if exists "anon read executions" on public.executions;
drop policy if exists "anon read telemetry" on public.system_telemetry;
drop policy if exists "anon read control commands" on public.control_commands;
drop policy if exists "anon read paper trades" on public.paper_trades;

drop policy if exists "admin read orders" on public.orders;
create policy "admin read orders"
on public.orders for select
to authenticated
using (auth.uid() = '37c5946c-7cb9-417b-b6f4-a81d81a482dd');

drop policy if exists "admin read executions" on public.executions;
create policy "admin read executions"
on public.executions for select
to authenticated
using (auth.uid() = '37c5946c-7cb9-417b-b6f4-a81d81a482dd');

drop policy if exists "admin read telemetry" on public.system_telemetry;
create policy "admin read telemetry"
on public.system_telemetry for select
to authenticated
using (auth.uid() = '37c5946c-7cb9-417b-b6f4-a81d81a482dd');

drop policy if exists "admin read control commands" on public.control_commands;
create policy "admin read control commands"
on public.control_commands for select
to authenticated
using (auth.uid() = '37c5946c-7cb9-417b-b6f4-a81d81a482dd');

drop policy if exists "admin read paper trades" on public.paper_trades;
create policy "admin read paper trades"
on public.paper_trades for select
to authenticated
using (auth.uid() = '37c5946c-7cb9-417b-b6f4-a81d81a482dd');
