class AppConfig {
  static const supabaseUrl = String.fromEnvironment(
    'SUPABASE_URL',
  );

  static const supabaseAnonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
  );

  static const backendBaseUrl = String.fromEnvironment(
    'TBOT_BACKEND_URL',
    defaultValue: 'http://localhost:8080',
  );

  static const enableSupabaseStreams = bool.fromEnvironment(
    'TBOT_ENABLE_SUPABASE_STREAMS',
    defaultValue: false,
  );

  static const defaultAsset = String.fromEnvironment(
    'TBOT_DEFAULT_ASSET',
    defaultValue: 'BTC/USDT',
  );

  static const defaultExchange = String.fromEnvironment(
    'TBOT_DEFAULT_EXCHANGE',
    defaultValue: 'BYBIT',
  );

  static const defaultTimeframe = String.fromEnvironment(
    'TBOT_DEFAULT_TIMEFRAME',
    defaultValue: '1m',
  );
}
