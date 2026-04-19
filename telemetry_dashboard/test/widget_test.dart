import 'package:flutter_test/flutter_test.dart';
import 'package:telemetry_dashboard/config/app_config.dart';

void main() {
  test('ships with configured Supabase project and backend defaults', () {
    expect(AppConfig.supabaseUrl, 'https://wmpkqdbftotydbmgagxw.supabase.co');
    expect(AppConfig.supabaseAnonKey.isNotEmpty, isTrue);
    expect(AppConfig.backendBaseUrl, 'http://localhost:8080');
  });
}
