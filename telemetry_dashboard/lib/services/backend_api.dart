import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/app_config.dart';
import '../models/dashboard_snapshot.dart';

class BackendApi {
  final Uri _snapshotUri = Uri.parse(
    '${AppConfig.backendBaseUrl}/api/v1/dashboard/snapshot',
  );
  final Uri _controlCommandsUri = Uri.parse(
    '${AppConfig.backendBaseUrl}/api/v1/control/commands',
  );

  Future<DashboardSnapshot> fetchSnapshot() async {
    final response = await http.get(_snapshotUri);
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch dashboard snapshot: ${response.body}');
    }
    return DashboardSnapshot.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> sendCommand({
    required String commandType,
    required String initiatedBy,
    required String reason,
  }) async {
    final response = await http.post(
      _controlCommandsUri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'commandType': commandType,
        'initiatedBy': initiatedBy,
        'reason': reason,
      }),
    );
    if (response.statusCode >= 400) {
      throw Exception('Command rejected: ${response.body}');
    }
  }
}
