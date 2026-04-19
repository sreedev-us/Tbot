class AppConfig {
  static const supabaseUrl = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://wmpkqdbftotydbmgagxw.supabase.co',
  );

  static const supabaseAnonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
    defaultValue: '',
  );

  static const backendBaseUrl = String.fromEnvironment(
    'TBOT_BACKEND_URL',
    defaultValue: 'http://localhost:8080',
  );
}
