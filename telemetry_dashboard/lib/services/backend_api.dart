import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

import '../config/app_config.dart';
import '../models/dashboard_snapshot.dart';
import '../models/paper_trade_report.dart';

class BackendApi {
  static const _requestTimeout = Duration(seconds: 3);

  final Uri _snapshotUri = Uri.parse(
    '${AppConfig.backendBaseUrl}/api/v1/dashboard/snapshot',
  );
  final Uri _controlCommandsUri = Uri.parse(
    '${AppConfig.backendBaseUrl}/api/v1/control/commands',
  );
  final Uri _controlStateUri = Uri.parse(
    '${AppConfig.backendBaseUrl}/api/v1/control/state',
  );
  final Uri _tradesUri = Uri.parse('${AppConfig.backendBaseUrl}/api/v1/dashboard/trades');

  Future<DashboardSnapshot> fetchSnapshot() async {
    final response = await http
        .get(_snapshotUri, headers: _authorizedHeaders())
        .timeout(_requestTimeout);
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch dashboard snapshot: ${response.body}');
    }
    return DashboardSnapshot.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<void> sendCommand({
    required String commandType,
    required String reason,
  }) async {
    final response = await http
        .post(
          _controlCommandsUri,
          headers: {
            'Content-Type': 'application/json',
            ..._authorizedHeaders(),
          },
          body: jsonEncode({
            'commandType': commandType,
            'reason': reason,
          }),
        )
        .timeout(_requestTimeout);
    if (response.statusCode >= 400) {
      throw Exception('Command rejected: ${response.body}');
    }
  }

  Future<List<PaperTradeReport>> fetchTradeReports({
    String? asset,
    String? exchange,
  }) async {
    final params = <String, String>{};
    if (asset != null && asset.isNotEmpty) {
      params['asset'] = asset;
    }
    if (exchange != null && exchange.isNotEmpty) {
      params['exchange'] = exchange;
    }
    final uri = _tradesUri.replace(queryParameters: params.isEmpty ? null : params);
    final response = await http
        .get(uri, headers: _authorizedHeaders())
        .timeout(_requestTimeout);
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch trade reports: ${response.body}');
    }
    return (jsonDecode(response.body) as List<dynamic>)
        .map((item) => PaperTradeReport.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<Map<String, dynamic>> fetchControlState() async {
    final response = await http
        .get(_controlStateUri, headers: _authorizedHeaders())
        .timeout(_requestTimeout);
    if (response.statusCode >= 400) {
      throw Exception('Failed to fetch control state: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Map<String, String> _authorizedHeaders() {
    final session = Supabase.instance.client.auth.currentSession;
    final accessToken = session?.accessToken;
    if (accessToken == null || accessToken.isEmpty) {
      return const {};
    }
    return {'Authorization': 'Bearer $accessToken'};
  }
}
