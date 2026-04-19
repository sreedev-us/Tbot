class DashboardSnapshot {
  DashboardSnapshot({
    required this.totalOrders,
    required this.executedOrders,
    required this.rejectedOrders,
    required this.currentExposure,
    required this.totalExecutedNotional,
    required this.averageExecutionLatencyMs,
    required this.engineHalted,
    required this.liquidationRequested,
    required this.generatedAt,
  });

  final int totalOrders;
  final int executedOrders;
  final int rejectedOrders;
  final num currentExposure;
  final num totalExecutedNotional;
  final num averageExecutionLatencyMs;
  final bool engineHalted;
  final bool liquidationRequested;
  final DateTime generatedAt;

  factory DashboardSnapshot.fromJson(Map<String, dynamic> json) {
    return DashboardSnapshot(
      totalOrders: json['totalOrders'] as int? ?? 0,
      executedOrders: json['executedOrders'] as int? ?? 0,
      rejectedOrders: json['rejectedOrders'] as int? ?? 0,
      currentExposure: json['currentExposure'] as num? ?? 0,
      totalExecutedNotional: json['totalExecutedNotional'] as num? ?? 0,
      averageExecutionLatencyMs: json['averageExecutionLatencyMs'] as num? ?? 0,
      engineHalted: json['engineHalted'] as bool? ?? false,
      liquidationRequested: json['liquidationRequested'] as bool? ?? false,
      generatedAt:
          DateTime.tryParse(json['generatedAt'] as String? ?? '') ??
          DateTime.now().toUtc(),
    );
  }
}
