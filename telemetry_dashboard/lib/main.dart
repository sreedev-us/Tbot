import 'package:flutter/widgets.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'app.dart';
import 'config/app_config.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  if (AppConfig.supabaseAnonKey.isEmpty) {
    throw StateError(
      'SUPABASE_ANON_KEY is required. Pass it with --dart-define before launching the dashboard.',
    );
  }
  await Supabase.initialize(
    url: AppConfig.supabaseUrl,
    anonKey: AppConfig.supabaseAnonKey,
  );
  runApp(const TbotTelemetryApp());
}
