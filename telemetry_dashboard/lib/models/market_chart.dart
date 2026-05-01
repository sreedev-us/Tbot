class MarketChart {
  MarketChart({
    required this.asset,
    required this.exchange,
    required this.timeframe,
    required this.source,
    required this.candles,
  });

  final String asset;
  final String exchange;
  final String timeframe;
  final String source;
  final List<MarketChartCandle> candles;
}

class MarketChartCandle {
  MarketChartCandle({
    required this.timestamp,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
  });

  final DateTime timestamp;
  final num open;
  final num high;
  final num low;
  final num close;
  final num volume;
}
