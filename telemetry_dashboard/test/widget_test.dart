import 'package:flutter_test/flutter_test.dart';
import 'package:telemetry_dashboard/config/app_config.dart';

void main() {
  test('ships with secure runtime config requirements and backend defaults', () {
    expect(AppConfig.supabaseUrl, isEmpty);
    expect(AppConfig.supabaseAnonKey, isEmpty);
    expect(AppConfig.backendBaseUrl, 'http://localhost:8080');
    expect(AppConfig.defaultExchange, 'BYBIT');
  });
}
