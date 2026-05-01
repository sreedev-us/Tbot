import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/paper_trade_report.dart';

class RealtimeLedgerService {
  RealtimeLedgerService(this._client);

  final SupabaseClient _client;

  Stream<List<Map<String, dynamic>>> ordersStream() {
    return _client
        .from('orders')
        .stream(primaryKey: ['id'])
        .order('received_at', ascending: false);
  }

  Stream<List<Map<String, dynamic>>> telemetryStream() {
    return _client
        .from('system_telemetry')
        .stream(primaryKey: ['id'])
        .order('recorded_at', ascending: false);
  }

  Stream<List<Map<String, dynamic>>> controlCommandsStream() {
    return _client
        .from('control_commands')
        .stream(primaryKey: ['id'])
        .order('created_at', ascending: false);
  }

  Stream<List<PaperTradeReport>> paperTradesStream() {
    return _client
        .from('paper_trades')
        .stream(primaryKey: ['id'])
        .order('opened_at', ascending: false)
        .asyncMap((tradeRows) async {
          final orderRows = await _client
              .from('orders')
              .select('id,signal_id')
              .order('received_at', ascending: false);
          final ordersById = {
            for (final row in orderRows)
              row['id'].toString(): row,
          };
          return tradeRows
              .map(
                (row) => PaperTradeReport.fromRealtime(
                  row,
                  orderRow: ordersById[row['order_id']?.toString()],
                ),
              )
              .toList();
        });
  }
}
