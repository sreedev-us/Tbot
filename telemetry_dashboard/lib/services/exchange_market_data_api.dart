import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/market_chart.dart';

class ExchangeMarketDataApi {
  ExchangeMarketDataApi({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<MarketChart> fetchCandles({
    required String asset,
    required String exchange,
    required String timeframe,
    int limit = 200,
  }) async {
    final normalizedExchange = exchange.toUpperCase();
    return switch (normalizedExchange) {
      'BYBIT' => _fetchBybitCandles(
        asset: asset,
        timeframe: timeframe,
        limit: limit,
      ),
      'BINANCE' => _fetchBinanceCandles(
        asset: asset,
        timeframe: timeframe,
        limit: limit,
      ),
      'KRAKEN' => _fetchKrakenCandles(
        asset: asset,
        timeframe: timeframe,
        limit: limit,
      ),
      _ => throw Exception('Unsupported exchange for candle feed: $exchange'),
    };
  }

  Stream<MarketChart> watchCandles({
    required String asset,
    required String exchange,
    required String timeframe,
    Duration refreshInterval = const Duration(seconds: 3),
  }) async* {
    yield await fetchCandles(
      asset: asset,
      exchange: exchange,
      timeframe: timeframe,
    );
    yield* Stream.periodic(refreshInterval).asyncMap(
      (_) => fetchCandles(
        asset: asset,
        exchange: exchange,
        timeframe: timeframe,
      ),
    );
  }

  Future<MarketChart> _fetchBybitCandles({
    required String asset,
    required String timeframe,
    required int limit,
  }) async {
    final symbol = _compactSymbol(asset);
    final interval = switch (timeframe) {
      '1m' => '1',
      '3m' => '3',
      '5m' => '5',
      '15m' => '15',
      '30m' => '30',
      '1h' => '60',
      '4h' => '240',
      '1d' => 'D',
      _ => '1',
    };
    final uri = Uri.https('api.bybit.com', '/v5/market/kline', {
      'category': 'linear',
      'symbol': symbol,
      'interval': interval,
      'limit': '$limit',
    });
    final response = await _client.get(uri);
    if (response.statusCode >= 400) {
      throw Exception('Bybit candles failed: ${response.body}');
    }
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    final rows = (payload['result']?['list'] as List<dynamic>? ?? const [])
        .cast<List<dynamic>>()
        .reversed
        .toList();
    return MarketChart(
      asset: asset,
      exchange: 'BYBIT',
      timeframe: timeframe,
      source: 'exchange-rest',
      candles: rows.map((row) {
        return MarketChartCandle(
          timestamp: DateTime.fromMillisecondsSinceEpoch(
            int.parse(row[0].toString()),
            isUtc: true,
          ),
          open: num.parse(row[1].toString()),
          high: num.parse(row[2].toString()),
          low: num.parse(row[3].toString()),
          close: num.parse(row[4].toString()),
          volume: num.parse(row[5].toString()),
        );
      }).toList(),
    );
  }

  Future<MarketChart> _fetchBinanceCandles({
    required String asset,
    required String timeframe,
    required int limit,
  }) async {
    final symbol = _compactSymbol(asset);
    final uri = Uri.https('api.binance.com', '/api/v3/klines', {
      'symbol': symbol,
      'interval': timeframe,
      'limit': '$limit',
    });
    final response = await _client.get(uri);
    if (response.statusCode >= 400) {
      throw Exception('Binance candles failed: ${response.body}');
    }
    final rows = (jsonDecode(response.body) as List<dynamic>).cast<List<dynamic>>();
    return MarketChart(
      asset: asset,
      exchange: 'BINANCE',
      timeframe: timeframe,
      source: 'exchange-rest',
      candles: rows.map((row) {
        return MarketChartCandle(
          timestamp: DateTime.fromMillisecondsSinceEpoch(
            row[0] as int,
            isUtc: true,
          ),
          open: num.parse(row[1].toString()),
          high: num.parse(row[2].toString()),
          low: num.parse(row[3].toString()),
          close: num.parse(row[4].toString()),
          volume: num.parse(row[5].toString()),
        );
      }).toList(),
    );
  }

  Future<MarketChart> _fetchKrakenCandles({
    required String asset,
    required String timeframe,
    required int limit,
  }) async {
    final pair = _krakenPair(asset);
    final intervalMinutes = switch (timeframe) {
      '1m' => '1',
      '5m' => '5',
      '15m' => '15',
      '30m' => '30',
      '1h' => '60',
      '4h' => '240',
      '1d' => '1440',
      _ => '1',
    };
    final uri = Uri.https('api.kraken.com', '/0/public/OHLC', {
      'pair': pair,
      'interval': intervalMinutes,
    });
    final response = await _client.get(uri);
    if (response.statusCode >= 400) {
      throw Exception('Kraken candles failed: ${response.body}');
    }
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    final result = payload['result'] as Map<String, dynamic>? ?? const {};
    final rows = (result[pair] as List<dynamic>? ?? const [])
        .cast<List<dynamic>>()
        .take(limit)
        .toList();
    return MarketChart(
      asset: asset,
      exchange: 'KRAKEN',
      timeframe: timeframe,
      source: 'exchange-rest',
      candles: rows.map((row) {
        return MarketChartCandle(
          timestamp: DateTime.fromMillisecondsSinceEpoch(
            (num.parse(row[0].toString()) * 1000).toInt(),
            isUtc: true,
          ),
          open: num.parse(row[1].toString()),
          high: num.parse(row[2].toString()),
          low: num.parse(row[3].toString()),
          close: num.parse(row[4].toString()),
          volume: num.parse(row[6].toString()),
        );
      }).toList(),
    );
  }

  String _compactSymbol(String asset) => asset.replaceAll('/', '').toUpperCase();

  String _krakenPair(String asset) {
    return switch (asset.toUpperCase()) {
      'BTC/USDT' => 'XBTUSDT',
      'BTC/USD' => 'XBTUSD',
      'ETH/USDT' => 'ETHUSDT',
      'ETH/USD' => 'ETHUSD',
      _ => _compactSymbol(asset),
    };
  }
}
