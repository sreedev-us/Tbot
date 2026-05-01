class PaperTradeReport {
  PaperTradeReport({
    required this.tradeId,
    required this.orderId,
    required this.signalId,
    required this.asset,
    required this.exchange,
    required this.action,
    required this.strategyName,
    required this.confidence,
    required this.requestedNotional,
    required this.quantity,
    required this.entryPrice,
    required this.exitPrice,
    required this.stopLossPrice,
    required this.takeProfitPrice,
    required this.openedAt,
    required this.closedAt,
    required this.status,
    required this.outcome,
    required this.realizedPnl,
    required this.realizedPnlPct,
    required this.closeReason,
  });

  final String tradeId;
  final String? orderId;
  final String signalId;
  final String asset;
  final String exchange;
  final String action;
  final String strategyName;
  final num confidence;
  final num requestedNotional;
  final num quantity;
  final num entryPrice;
  final num? exitPrice;
  final num stopLossPrice;
  final num takeProfitPrice;
  final DateTime openedAt;
  final DateTime? closedAt;
  final String status;
  final String outcome;
  final num? realizedPnl;
  final num? realizedPnlPct;
  final String? closeReason;

  factory PaperTradeReport.fromJson(Map<String, dynamic> json) {
    final orderId = _stringOf(json['orderId'] ?? json['order_id']);
    return PaperTradeReport(
      tradeId: _stringOf(json['tradeId'] ?? json['id']) ?? '',
      orderId: orderId,
      signalId: _stringOf(json['signalId'] ?? json['signal_id']) ?? '',
      asset: _stringOf(json['asset']) ?? '',
      exchange: _stringOf(json['exchange'] ?? json['exchangeName'] ?? json['exchange_name']) ?? '',
      action: _stringOf(json['action']) ?? '',
      strategyName: _stringOf(json['strategyName'] ?? json['strategy_name']) ?? '',
      confidence: _numOf(json['confidence']),
      requestedNotional: _numOf(json['requestedNotional'] ?? json['requested_notional']),
      quantity: _numOf(json['quantity']),
      entryPrice: _numOf(json['entryPrice'] ?? json['entry_price']),
      exitPrice: _nullableNumOf(json['exitPrice'] ?? json['exit_price']),
      stopLossPrice: _numOf(json['stopLossPrice'] ?? json['stop_loss_price']),
      takeProfitPrice: _numOf(json['takeProfitPrice'] ?? json['take_profit_price']),
      openedAt:
          DateTime.tryParse(_stringOf(json['openedAt'] ?? json['opened_at']) ?? '') ??
          DateTime.now().toUtc(),
      closedAt: DateTime.tryParse(_stringOf(json['closedAt'] ?? json['closed_at']) ?? ''),
      status: _stringOf(json['status']) ?? 'OPEN',
      outcome: _stringOf(json['outcome']) ?? 'OPEN',
      realizedPnl: _nullableNumOf(json['realizedPnl'] ?? json['realized_pnl']),
      realizedPnlPct: _nullableNumOf(json['realizedPnlPct'] ?? json['realized_pnl_pct']),
      closeReason: _stringOf(json['closeReason'] ?? json['close_reason']),
    );
  }

  static PaperTradeReport fromRealtime(
    Map<String, dynamic> tradeRow, {
    Map<String, dynamic>? orderRow,
  }) {
    return PaperTradeReport.fromJson({
      ...tradeRow,
      if (orderRow != null) ...orderRow,
      'orderId': tradeRow['order_id'] ?? orderRow?['id'],
      'signalId': orderRow?['signal_id'] ?? orderRow?['signalId'],
      'exchange':
          tradeRow['exchange_name'] ??
          orderRow?['exchange_name'] ??
          tradeRow['exchange'] ??
          orderRow?['exchange'],
    });
  }
}

String? _stringOf(Object? value) {
  if (value == null) {
    return null;
  }
  return value.toString();
}

num _numOf(Object? value) {
  return _nullableNumOf(value) ?? 0;
}

num? _nullableNumOf(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value;
  }
  return num.tryParse(value.toString());
}
