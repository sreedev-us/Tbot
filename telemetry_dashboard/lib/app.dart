import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'config/app_config.dart';
import 'models/dashboard_snapshot.dart';
import 'models/market_chart.dart';
import 'models/paper_trade_report.dart';
import 'services/backend_api.dart';
import 'services/exchange_market_data_api.dart';
import 'services/realtime_ledger_service.dart';
import 'web_url_helper.dart';

class TbotTelemetryApp extends StatelessWidget {
  const TbotTelemetryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'tbot telemetry',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF2EDE2),
        textTheme: GoogleFonts.spaceGroteskTextTheme(),
        colorScheme:
            ColorScheme.fromSeed(
              seedColor: const Color(0xFF005F73),
              brightness: Brightness.light,
            ).copyWith(
              surface: const Color(0xFFFFFCF6),
              primary: const Color(0xFF005F73),
              secondary: const Color(0xFFEE9B00),
              tertiary: const Color(0xFF9B2226),
            ),
      ),
      home: const _AuthGate(),
    );
  }
}

class _AuthGate extends StatefulWidget {
  const _AuthGate();

  @override
  State<_AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<_AuthGate> {
  bool _processingRedirect = false;
  bool _localBypass = false;
  String? _authError;

  @override
  void initState() {
    super.initState();
    _handleAuthRedirect();
  }

  Future<void> _handleAuthRedirect() async {
    if (!hasAuthRedirectParams()) {
      return;
    }

    setState(() {
      _processingRedirect = true;
      _authError = null;
      _localBypass = false;
    });

    try {
      await Supabase.instance.client.auth.getSessionFromUrl(Uri.base);
      clearAuthRedirectFromUrl();
    } on AuthException catch (error) {
      setState(() {
        _authError = error.message;
      });
    } catch (error) {
      setState(() {
        _authError = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _processingRedirect = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_processingRedirect) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final auth = Supabase.instance.client.auth;
    return StreamBuilder<AuthState>(
      stream: auth.onAuthStateChange,
      initialData: AuthState(
        AuthChangeEvent.initialSession,
        auth.currentSession,
      ),
      builder: (context, snapshot) {
        final session = snapshot.data?.session ?? auth.currentSession;
        if (session != null) {
          return DashboardPage(
            userEmail: session.user.email ?? session.user.id,
            adminSignedIn: true,
            onConnectAdmin: null,
          );
        }

        if (!_localBypass) {
          return _AdminSignInPage(
            errorMessage: _authError,
            allowLocalMode: !AppConfig.enableSupabaseStreams,
            onContinueLocal: () {
              setState(() {
                _localBypass = true;
                _authError = null;
              });
            },
          );
        }

        return DashboardPage(
          userEmail: 'local-observer',
          adminSignedIn: false,
          onConnectAdmin: () {
            setState(() {
              _localBypass = false;
              _authError = null;
            });
          },
        );
      },
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({
    required this.userEmail,
    required this.adminSignedIn,
    required this.onConnectAdmin,
    super.key,
  });

  final String userEmail;
  final bool adminSignedIn;
  final VoidCallback? onConnectAdmin;

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final BackendApi _backendApi = BackendApi();
  final ExchangeMarketDataApi _marketDataApi = ExchangeMarketDataApi();
  final TextEditingController _reasonController = TextEditingController(
    text: 'manual override',
  );
  final ValueNotifier<int> _refreshNonce = ValueNotifier<int>(0);
  final ValueNotifier<DateTime?> _lastSyncAt = ValueNotifier<DateTime?>(null);
  final ValueNotifier<String?> _staleReason = ValueNotifier<String?>(null);
  final DateFormat _timeFormat = DateFormat('MMM d, HH:mm:ss');
  final NumberFormat _priceFormat = NumberFormat.currency(
    symbol: '\$',
    decimalDigits: 2,
  );
  final NumberFormat _pnlFormat = NumberFormat.currency(
    symbol: '\$',
    decimalDigits: 4,
  );

  bool _sendingCommand = false;
  late final String _asset;
  late final String _exchange;
  late final String _timeframe;
  late final RealtimeLedgerService _ledgerService;

  @override
  void initState() {
    super.initState();
    _asset = AppConfig.defaultAsset;
    _exchange = AppConfig.defaultExchange;
    _timeframe = AppConfig.defaultTimeframe;
    _ledgerService = RealtimeLedgerService(Supabase.instance.client);
  }

  @override
  void dispose() {
    _reasonController.dispose();
    _refreshNonce.dispose();
    _lastSyncAt.dispose();
    _staleReason.dispose();
    super.dispose();
  }

  Future<void> _sendCommand(String commandType) async {
    setState(() {
      _sendingCommand = true;
    });

    try {
      await _backendApi.sendCommand(
        commandType: commandType,
        reason: _reasonController.text.trim(),
      );
      _refreshNonce.value += 1;
      _lastSyncAt.value = DateTime.now();
      _staleReason.value = null;
    } finally {
      if (mounted) {
        setState(() {
          _sendingCommand = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final body = AppConfig.enableSupabaseStreams
        ? _RealtimeDashboardView(
            asset: _asset,
            exchange: _exchange,
            timeframe: _timeframe,
            backendApi: _backendApi,
            marketDataApi: _marketDataApi,
            ledgerService: _ledgerService,
            reasonController: _reasonController,
            sendingCommand: _sendingCommand,
            onCommand: _sendCommand,
            userEmail: widget.userEmail,
            adminSignedIn: widget.adminSignedIn,
            onConnectAdmin: widget.onConnectAdmin,
            timeFormat: _timeFormat,
            priceFormat: _priceFormat,
            pnlFormat: _pnlFormat,
            lastSyncAt: _lastSyncAt,
            staleReason: _staleReason,
          )
        : _LocalObserverView(
            asset: _asset,
            exchange: _exchange,
            timeframe: _timeframe,
            backendApi: _backendApi,
            marketDataApi: _marketDataApi,
            reasonController: _reasonController,
            sendingCommand: _sendingCommand,
            onCommand: _sendCommand,
            userEmail: widget.userEmail,
            adminSignedIn: widget.adminSignedIn,
            onConnectAdmin: widget.onConnectAdmin,
            timeFormat: _timeFormat,
            priceFormat: _priceFormat,
            pnlFormat: _pnlFormat,
            refreshNonce: _refreshNonce,
            lastSyncAt: _lastSyncAt,
            staleReason: _staleReason,
          );

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFF2EDE2), Color(0xFFE3D5CA), Color(0xFFD8F3DC)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(child: body),
      ),
    );
  }
}

class _RealtimeDashboardView extends StatelessWidget {
  const _RealtimeDashboardView({
    required this.asset,
    required this.exchange,
    required this.timeframe,
    required this.backendApi,
    required this.marketDataApi,
    required this.ledgerService,
    required this.reasonController,
    required this.sendingCommand,
    required this.onCommand,
    required this.userEmail,
    required this.adminSignedIn,
    required this.onConnectAdmin,
    required this.timeFormat,
    required this.priceFormat,
    required this.pnlFormat,
    required this.lastSyncAt,
    required this.staleReason,
  });

  final String asset;
  final String exchange;
  final String timeframe;
  final BackendApi backendApi;
  final ExchangeMarketDataApi marketDataApi;
  final RealtimeLedgerService ledgerService;
  final TextEditingController reasonController;
  final bool sendingCommand;
  final Future<void> Function(String commandType) onCommand;
  final String userEmail;
  final bool adminSignedIn;
  final VoidCallback? onConnectAdmin;
  final DateFormat timeFormat;
  final NumberFormat priceFormat;
  final NumberFormat pnlFormat;
  final ValueNotifier<DateTime?> lastSyncAt;
  final ValueNotifier<String?> staleReason;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<List<PaperTradeReport>>(
      stream: _trackStream(
        ledgerService.paperTradesStream(),
        lastSyncAt: lastSyncAt,
        staleReason: staleReason,
      ),
      builder: (context, tradesSnapshot) {
        final streamedTrades = tradesSnapshot.data ?? const <PaperTradeReport>[];
        final needsFallback =
            streamedTrades.isEmpty || tradesSnapshot.hasError;
        return FutureBuilder<List<PaperTradeReport>>(
          future: needsFallback
              ? backendApi.fetchTradeReports(asset: asset, exchange: exchange)
              : Future.value(streamedTrades),
          builder: (context, tradeFallbackSnapshot) {
            final trades =
                tradeFallbackSnapshot.data?.isNotEmpty == true
                ? tradeFallbackSnapshot.data!
                : streamedTrades;
            return StreamBuilder<List<Map<String, dynamic>>>(
              stream: _trackStream(
                ledgerService.ordersStream(),
                lastSyncAt: lastSyncAt,
                staleReason: staleReason,
              ),
              builder: (context, ordersSnapshot) {
                final orders =
                    ordersSnapshot.data ?? const <Map<String, dynamic>>[];
                return StreamBuilder<List<Map<String, dynamic>>>(
                  stream: _trackStream(
                    ledgerService.telemetryStream(),
                    lastSyncAt: lastSyncAt,
                    staleReason: staleReason,
                  ),
                  builder: (context, telemetrySnapshot) {
                    final telemetry =
                        telemetrySnapshot.data ?? const <Map<String, dynamic>>[];
                    return StreamBuilder<List<Map<String, dynamic>>>(
                      stream: _trackStream(
                        ledgerService.controlCommandsStream(),
                        lastSyncAt: lastSyncAt,
                        staleReason: staleReason,
                      ),
                      builder: (context, controlSnapshot) {
                        final controlRows =
                            controlSnapshot.data ??
                            const <Map<String, dynamic>>[];
                        final state = _deriveControlState(controlRows);
                        final metrics = _deriveMetrics(
                          orders: orders,
                          trades: trades,
                          telemetry: telemetry,
                        );
                        final error = [
                          tradesSnapshot.error,
                          tradeFallbackSnapshot.error,
                          ordersSnapshot.error,
                          telemetrySnapshot.error,
                          controlSnapshot.error,
                        ].whereType<Object>().map((e) => e.toString()).join('\n');

                        return _DashboardLayout(
                          asset: asset,
                          exchange: exchange,
                          timeframe: timeframe,
                          userEmail: userEmail,
                          adminSignedIn: adminSignedIn,
                          onConnectAdmin: onConnectAdmin,
                          state: state,
                          metrics: metrics,
                          trades: trades,
                          lastSyncAt: lastSyncAt,
                          staleReason: staleReason,
                          marketChart: _ExchangeMarketChartCard(
                            asset: asset,
                            exchange: exchange,
                            timeframe: timeframe,
                            trades: trades,
                            marketDataApi: marketDataApi,
                            priceFormat: priceFormat,
                            timeFormat: timeFormat,
                            lastSyncAt: lastSyncAt,
                            staleReason: staleReason,
                          ),
                          reasonController: reasonController,
                          sendingCommand: sendingCommand,
                          onCommand: onCommand,
                          timeFormat: timeFormat,
                          priceFormat: priceFormat,
                          pnlFormat: pnlFormat,
                          errorMessage: error.isEmpty ? null : error,
                        );
                      },
                    );
                  },
                );
              },
            );
          },
        );
      },
    );
  }
}

  Stream<T> _trackStream<T>(
    Stream<T> source, {
    required ValueNotifier<DateTime?> lastSyncAt,
    required ValueNotifier<String?> staleReason,
  }) {
    return source.map((event) {
      lastSyncAt.value = DateTime.now();
      staleReason.value = null;
      return event;
    }).handleError((error) {
      staleReason.value = error.toString();
    });
  }

class _LocalObserverView extends StatelessWidget {
  const _LocalObserverView({
    required this.asset,
    required this.exchange,
    required this.timeframe,
    required this.backendApi,
    required this.marketDataApi,
    required this.reasonController,
    required this.sendingCommand,
    required this.onCommand,
    required this.userEmail,
    required this.adminSignedIn,
    required this.onConnectAdmin,
    required this.timeFormat,
    required this.priceFormat,
    required this.pnlFormat,
    required this.refreshNonce,
    required this.lastSyncAt,
    required this.staleReason,
  });

  final String asset;
  final String exchange;
  final String timeframe;
  final BackendApi backendApi;
  final ExchangeMarketDataApi marketDataApi;
  final TextEditingController reasonController;
  final bool sendingCommand;
  final Future<void> Function(String commandType) onCommand;
  final String userEmail;
  final bool adminSignedIn;
  final VoidCallback? onConnectAdmin;
  final DateFormat timeFormat;
  final NumberFormat priceFormat;
  final NumberFormat pnlFormat;
  final ValueNotifier<int> refreshNonce;
  final ValueNotifier<DateTime?> lastSyncAt;
  final ValueNotifier<String?> staleReason;

  Future<_LocalDashboardData> _load() async {
    try {
      final results = await Future.wait([
        backendApi.fetchSnapshot(),
        backendApi.fetchTradeReports(asset: asset, exchange: exchange),
        backendApi.fetchControlState(),
      ]);
      lastSyncAt.value = DateTime.now();
      staleReason.value = null;
      return _LocalDashboardData(
        snapshot: results[0] as DashboardSnapshot,
        trades: results[1] as List<PaperTradeReport>,
        controlState: results[2] as Map<String, dynamic>,
      );
    } catch (error) {
      staleReason.value = error.toString();
      rethrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<int>(
      valueListenable: refreshNonce,
      builder: (context, _, __) {
        return FutureBuilder<_LocalDashboardData>(
          future: _load(),
          builder: (context, snapshot) {
            final data = snapshot.data;
            final localState = _DerivedControlState(
              engineHalted: data?.controlState['engineHalted'] as bool? ?? false,
              liquidationRequested:
                  data?.controlState['liquidationRequested'] as bool? ?? false,
            );
            final localMetrics = _DerivedMetrics.fromSnapshot(data?.snapshot);
            return _DashboardLayout(
              asset: asset,
              exchange: exchange,
              timeframe: timeframe,
              userEmail: userEmail,
              adminSignedIn: adminSignedIn,
              onConnectAdmin: onConnectAdmin,
              state: localState,
              metrics: localMetrics,
              trades: data?.trades ?? const [],
              lastSyncAt: lastSyncAt,
              staleReason: staleReason,
              marketChart: Column(
                children: [
                  _StreamsRequiredCard(),
                  const SizedBox(height: 16),
                  _ExchangeMarketChartCard(
                    asset: asset,
                    exchange: exchange,
                    timeframe: timeframe,
                    trades: data?.trades ?? const [],
                    marketDataApi: marketDataApi,
                    priceFormat: priceFormat,
                    timeFormat: timeFormat,
                    lastSyncAt: lastSyncAt,
                    staleReason: staleReason,
                  ),
                ],
              ),
              reasonController: reasonController,
              sendingCommand: sendingCommand,
              onCommand: onCommand,
              timeFormat: timeFormat,
              priceFormat: priceFormat,
              pnlFormat: pnlFormat,
              errorMessage: snapshot.hasError ? snapshot.error.toString() : null,
            );
          },
        );
      },
    );
  }
}

class _DashboardLayout extends StatelessWidget {
  const _DashboardLayout({
    required this.asset,
    required this.exchange,
    required this.timeframe,
    required this.userEmail,
    required this.adminSignedIn,
    required this.onConnectAdmin,
    required this.state,
    required this.metrics,
    required this.trades,
    required this.lastSyncAt,
    required this.staleReason,
    required this.marketChart,
    required this.reasonController,
    required this.sendingCommand,
    required this.onCommand,
    required this.timeFormat,
    required this.priceFormat,
    required this.pnlFormat,
    required this.errorMessage,
  });

  final String asset;
  final String exchange;
  final String timeframe;
  final String userEmail;
  final bool adminSignedIn;
  final VoidCallback? onConnectAdmin;
  final _DerivedControlState state;
  final _DerivedMetrics metrics;
  final List<PaperTradeReport> trades;
  final ValueNotifier<DateTime?> lastSyncAt;
  final ValueNotifier<String?> staleReason;
  final Widget marketChart;
  final TextEditingController reasonController;
  final bool sendingCommand;
  final Future<void> Function(String commandType) onCommand;
  final DateFormat timeFormat;
  final NumberFormat priceFormat;
  final NumberFormat pnlFormat;
  final String? errorMessage;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    final wide = width > 1120;

    return Stack(
      children: [
        ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Wrap(
              spacing: 16,
              runSpacing: 16,
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 760),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'tbot telemetry',
                        style: Theme.of(context).textTheme.displaySmall?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Trade state is stream-driven from Supabase. Visual candles are sourced directly from the exchange public API so the execution backend never owns chart data.',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Watching $exchange $asset on $timeframe candles.',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ],
                  ),
                ),
                _StatusBanner(
                  state: state,
                  userEmail: userEmail,
                  adminSignedIn: adminSignedIn,
                  onConnectAdmin: onConnectAdmin,
                ),
              ],
            ),
            const SizedBox(height: 24),
            if (errorMessage != null) ...[
              _AlertBar(message: errorMessage!),
              const SizedBox(height: 16),
            ],
            Wrap(
              spacing: 16,
              runSpacing: 16,
              children: [
                _MetricCard(
                  title: 'Orders',
                  value: '${metrics.totalOrders}',
                  detail:
                      '${metrics.executedOrders} executed / ${metrics.rejectedOrders} rejected',
                  accent: const Color(0xFF005F73),
                ),
                _MetricCard(
                  title: 'Open trades',
                  value: '${metrics.openTrades}',
                  detail: 'Realtime paper-trade state',
                  accent: const Color(0xFF0A9396),
                ),
                _MetricCard(
                  title: 'Net demo P/L',
                  value: pnlFormat.format(metrics.netPnl),
                  detail:
                      '${metrics.profitTrades} profits / ${metrics.lossTrades} losses',
                  accent: metrics.netPnl >= 0
                      ? const Color(0xFF2A9D8F)
                      : const Color(0xFF9B2226),
                ),
                _MetricCard(
                  title: 'Latency',
                  value: '${metrics.averageLatencyMs.toStringAsFixed(1)} ms',
                  detail: 'Derived from streamed telemetry',
                  accent: const Color(0xFFEE9B00),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Flex(
              direction: wide ? Axis.horizontal : Axis.vertical,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 5,
                  child: Column(
                    children: [
                      marketChart,
                      const SizedBox(height: 16),
                      _TradeHistoryCard(
                        trades: trades,
                        priceFormat: priceFormat,
                        pnlFormat: pnlFormat,
                        timeFormat: timeFormat,
                      ),
                    ],
                  ),
                ),
                SizedBox(width: wide ? 16 : 0, height: wide ? 0 : 16),
                Expanded(
                  flex: 3,
                  child: Column(
                    children: [
                      _ControlPanel(
                        state: state,
                        reasonController: reasonController,
                        sending: sendingCommand,
                        onCommand: onCommand,
                      ),
                      const SizedBox(height: 16),
                      _DemoReportCard(
                        trades: trades,
                        pnlFormat: pnlFormat,
                        priceFormat: priceFormat,
                        timeFormat: timeFormat,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
        _StaleConnectionOverlay(
          lastSyncAt: lastSyncAt,
          staleReason: staleReason,
        ),
      ],
    );
  }
}

class _ExchangeMarketChartCard extends StatelessWidget {
  const _ExchangeMarketChartCard({
    required this.asset,
    required this.exchange,
    required this.timeframe,
    required this.trades,
    required this.marketDataApi,
    required this.priceFormat,
    required this.timeFormat,
    required this.lastSyncAt,
    required this.staleReason,
  });

  final String asset;
  final String exchange;
  final String timeframe;
  final List<PaperTradeReport> trades;
  final ExchangeMarketDataApi marketDataApi;
  final NumberFormat priceFormat;
  final DateFormat timeFormat;
  final ValueNotifier<DateTime?> lastSyncAt;
  final ValueNotifier<String?> staleReason;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<MarketChart>(
      stream: marketDataApi.watchCandles(
        asset: asset,
        exchange: exchange,
        timeframe: timeframe,
      ),
      builder: (context, snapshot) {
        final chart = snapshot.data;
        if (snapshot.hasData) {
          lastSyncAt.value = DateTime.now();
          staleReason.value = null;
        }
        if (snapshot.hasError) {
          staleReason.value = snapshot.error.toString();
        }
        return DecoratedBox(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.86),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: const Color(0xFF005F73).withValues(alpha: 0.14),
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Demo market graph',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 8),
                Text(
                  'The chart repaints independently inside a boundary, so latency and trade cards can update without forcing a full canvas rebuild.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 18),
                RepaintBoundary(
                  child: SizedBox(
                    height: 320,
                    child: chart == null
                        ? Center(
                            child: snapshot.hasError
                                ? Text('Chart feed error: ${snapshot.error}')
                                : const CircularProgressIndicator(),
                          )
                        : _MarketLineChart(
                            chart: chart,
                            trades: trades,
                          ),
                  ),
                ),
                if (chart != null && chart.candles.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Text(
                    'Current Price: ${_formatAssetPrice(chart.candles.last.close)}',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF1D4ED8),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 12,
                    runSpacing: 12,
                    children: [
                      _TagChip(
                        label:
                            'Live ${_formatAssetPrice(chart.candles.last.close)}',
                        color: const Color(0xFF1D4ED8),
                      ),
                      _TagChip(
                        label:
                            'Start ${timeFormat.format(chart.candles.first.timestamp.toLocal())}',
                        color: const Color(0xFF005F73),
                      ),
                      _TagChip(
                        label:
                            'End ${timeFormat.format(chart.candles.last.timestamp.toLocal())}',
                        color: const Color(0xFF0A9396),
                      ),
                      _TagChip(
                        label: chart.source,
                        color: const Color(0xFFEE9B00),
                      ),
                    ],
                  ),
                ],
                if (trades.isNotEmpty) ...[
                  const SizedBox(height: 14),
                  ...trades.take(3).map(
                    (trade) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(
                        '#${trade.tradeId} ${trade.action} '
                        '${priceFormat.format(trade.entryPrice)}'
                        '${trade.exitPrice != null ? ' -> ${priceFormat.format(trade.exitPrice)}' : ' -> open'}'
                        ' (${trade.status == 'OPEN' ? 'Open' : trade.outcome})',
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

class _StreamsRequiredCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFFEE9B00).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: const Color(0xFFEE9B00).withValues(alpha: 0.28),
        ),
      ),
      child: const Padding(
        padding: EdgeInsets.all(18),
        child: Text(
          'Supabase realtime is disabled for this build. The dashboard is running in a limited observer mode without the stream-driven ledger path.',
        ),
      ),
    );
  }
}

class _AdminSignInPage extends StatefulWidget {
  const _AdminSignInPage({
    required this.allowLocalMode,
    required this.onContinueLocal,
    this.errorMessage,
  });

  final bool allowLocalMode;
  final VoidCallback onContinueLocal;
  final String? errorMessage;

  @override
  State<_AdminSignInPage> createState() => _AdminSignInPageState();
}

class _AdminSignInPageState extends State<_AdminSignInPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _otpController = TextEditingController();
  bool _submitting = false;
  bool _otpSent = false;
  String? _info;
  String? _error;

  @override
  void initState() {
    super.initState();
    _error = widget.errorMessage;
  }

  @override
  void dispose() {
    _emailController.dispose();
    _otpController.dispose();
    super.dispose();
  }

  @override
  void didUpdateWidget(covariant _AdminSignInPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.errorMessage != oldWidget.errorMessage) {
      _error = widget.errorMessage;
    }
  }

  String _redirectUrl() {
    final current = Uri.base;
    return current.replace(queryParameters: const {}, fragment: '').toString();
  }

  Future<void> _sendMagicLink() async {
    final email = _emailController.text.trim();
    if (email.isEmpty) {
      setState(() {
        _error = 'Enter the admin email first.';
        _info = null;
      });
      return;
    }

    setState(() {
      _submitting = true;
      _error = null;
      _info = null;
    });

    try {
      await Supabase.instance.client.auth.signInWithOtp(
        email: email,
        shouldCreateUser: false,
        emailRedirectTo: kIsWeb ? _redirectUrl() : AppConfig.backendBaseUrl,
      );
      setState(() {
        _otpSent = true;
        _info =
            'Check $email for a sign-in link or one-time code, then come back here.';
      });
    } on AuthException catch (error) {
      setState(() {
        _error = error.message;
      });
    } catch (error) {
      setState(() {
        _error = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _submitting = false;
        });
      }
    }
  }

  Future<void> _verifyOtp() async {
    final email = _emailController.text.trim();
    final token = _otpController.text.trim();

    if (email.isEmpty || token.isEmpty) {
      setState(() {
        _error = 'Enter both the admin email and the one-time code.';
        _info = null;
      });
      return;
    }

    setState(() {
      _submitting = true;
      _error = null;
      _info = null;
    });

    try {
      await Supabase.instance.client.auth.verifyOTP(
        email: email,
        token: token,
        type: OtpType.email,
      );
    } on AuthException catch (error) {
      setState(() {
        _error = error.message;
      });
    } catch (error) {
      setState(() {
        _error = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _submitting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFF2EDE2), Color(0xFFE3D5CA), Color(0xFFD8F3DC)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 460),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.88),
                    borderRadius: BorderRadius.circular(28),
                    border: Border.all(
                      color: const Color(0xFF005F73).withValues(alpha: 0.15),
                    ),
                    boxShadow: const [
                      BoxShadow(
                        blurRadius: 28,
                        color: Color(0x14001219),
                        offset: Offset(0, 18),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Admin sign-in required',
                          style: Theme.of(context).textTheme.headlineSmall
                              ?.copyWith(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'Realtime ledger subscriptions are restricted to your authenticated admin account.',
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                        const SizedBox(height: 20),
                        TextField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          autofillHints: const [AutofillHints.username],
                          decoration: _signInDecoration('Admin email'),
                        ),
                        if (_otpSent) ...[
                          const SizedBox(height: 12),
                          TextField(
                            controller: _otpController,
                            keyboardType: TextInputType.number,
                            decoration: _signInDecoration(
                              'One-time code (optional)',
                            ),
                            onSubmitted: (_) =>
                                _submitting ? null : _verifyOtp(),
                          ),
                        ],
                        if (_info != null) ...[
                          const SizedBox(height: 14),
                          _InfoBar(message: _info!),
                        ],
                        if (_error != null) ...[
                          const SizedBox(height: 14),
                          _AlertBar(message: _error!),
                        ],
                        const SizedBox(height: 18),
                        FilledButton(
                          onPressed: _submitting ? null : _sendMagicLink,
                          style: FilledButton.styleFrom(
                            minimumSize: const Size.fromHeight(50),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          child: Text(
                            _submitting
                                ? 'Sending sign-in email...'
                                : 'Email sign-in link',
                          ),
                        ),
                        if (_otpSent) ...[
                          const SizedBox(height: 12),
                          OutlinedButton(
                            onPressed: _submitting ? null : _verifyOtp,
                            style: OutlinedButton.styleFrom(
                              minimumSize: const Size.fromHeight(50),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            child: const Text('Verify one-time code'),
                          ),
                        ],
                        if (widget.allowLocalMode) ...[
                          const SizedBox(height: 12),
                          TextButton(
                            onPressed: _submitting
                                ? null
                                : widget.onContinueLocal,
                            child: const Text('Continue in local mode'),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  InputDecoration _signInDecoration(String label) {
    return InputDecoration(
      labelText: label,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16)),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({
    required this.state,
    required this.userEmail,
    required this.adminSignedIn,
    required this.onConnectAdmin,
  });

  final _DerivedControlState state;
  final String userEmail;
  final bool adminSignedIn;
  final VoidCallback? onConnectAdmin;

  @override
  Widget build(BuildContext context) {
    final color = state.engineHalted
        ? const Color(0xFF9B2226)
        : state.liquidationRequested
        ? const Color(0xFFEE9B00)
        : const Color(0xFF2A9D8F);

    return DecoratedBox(
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: color.withValues(alpha: 0.42)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              state.engineHalted
                  ? 'Engine halted'
                  : state.liquidationRequested
                  ? 'Liquidation requested'
                  : 'System nominal',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
            const SizedBox(height: 6),
            Text(userEmail, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 10),
            if (adminSignedIn)
              OutlinedButton(
                onPressed: () async {
                  await Supabase.instance.client.auth.signOut(
                    scope: SignOutScope.local,
                  );
                  clearAuthRedirectFromUrl();
                },
                child: const Text('Sign out'),
              )
            else if (onConnectAdmin != null)
              OutlinedButton(
                onPressed: onConnectAdmin,
                child: const Text('Sign in as admin'),
              ),
          ],
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({
    required this.title,
    required this.value,
    required this.detail,
    required this.accent,
  });

  final String title;
  final String value;
  final String detail;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 260,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.76),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: accent.withValues(alpha: 0.24)),
          boxShadow: [
            BoxShadow(
              blurRadius: 24,
              color: accent.withValues(alpha: 0.08),
              offset: const Offset(0, 12),
            ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: accent,
                ),
              ),
              const SizedBox(height: 14),
              Text(
                value,
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(detail),
            ],
          ),
        ),
      ),
    );
  }
}

class _TradeHistoryCard extends StatelessWidget {
  const _TradeHistoryCard({
    required this.trades,
    required this.priceFormat,
    required this.pnlFormat,
    required this.timeFormat,
  });

  final List<PaperTradeReport> trades;
  final NumberFormat priceFormat;
  final NumberFormat pnlFormat;
  final DateFormat timeFormat;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.86),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: const Color(0xFF005F73).withValues(alpha: 0.14),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Trade history',
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(
              'Trades are streamed from the ledger, not inferred from a polling snapshot.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 18),
            if (trades.isEmpty)
              const Text('No paper trades recorded yet.')
            else
              ...trades.map(
                (trade) => Padding(
                  padding: const EdgeInsets.only(bottom: 14),
                  child: _TradeReportTile(
                    trade: trade,
                    priceFormat: priceFormat,
                    pnlFormat: pnlFormat,
                    timeFormat: timeFormat,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _DemoReportCard extends StatelessWidget {
  const _DemoReportCard({
    required this.trades,
    required this.pnlFormat,
    required this.priceFormat,
    required this.timeFormat,
  });

  final List<PaperTradeReport> trades;
  final NumberFormat pnlFormat;
  final NumberFormat priceFormat;
  final DateFormat timeFormat;

  @override
  Widget build(BuildContext context) {
    final latest = trades.isEmpty ? null : trades.first;

    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFFFFFCF6),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: const Color(0xFF005F73).withValues(alpha: 0.14),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detailed report',
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 10),
            if (latest == null)
              const Text(
                'A full trade report will appear here after the first streamed demo fill.',
              )
            else ...[
              _ReportLine(label: 'Trade', value: latest.tradeId),
              _ReportLine(label: 'Signal', value: latest.signalId),
              _ReportLine(
                label: 'Direction',
                value: '${latest.action} on ${latest.exchange}',
              ),
              _ReportLine(
                label: 'Opened',
                value: timeFormat.format(latest.openedAt.toLocal()),
              ),
              _ReportLine(
                label: 'Entry',
                value: priceFormat.format(latest.entryPrice),
              ),
              _ReportLine(
                label: 'Stop loss',
                value: priceFormat.format(latest.stopLossPrice),
              ),
              _ReportLine(
                label: 'Take profit',
                value: priceFormat.format(latest.takeProfitPrice),
              ),
              _ReportLine(
                label: 'Exit',
                value: latest.exitPrice == null
                    ? 'Still open'
                    : priceFormat.format(latest.exitPrice),
              ),
              _ReportLine(
                label: 'Outcome',
                value: latest.status == 'OPEN' ? 'Open trade' : latest.outcome,
              ),
              _ReportLine(
                label: 'P/L',
                value: latest.realizedPnl == null
                    ? 'Pending'
                    : '${pnlFormat.format(latest.realizedPnl)} (${latest.realizedPnlPct?.toStringAsFixed(2) ?? '0.00'}%)',
              ),
              _ReportLine(
                label: 'Close reason',
                value: latest.closeReason ?? 'Pending',
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _TradeReportTile extends StatelessWidget {
  const _TradeReportTile({
    required this.trade,
    required this.priceFormat,
    required this.pnlFormat,
    required this.timeFormat,
  });

  final PaperTradeReport trade;
  final NumberFormat priceFormat;
  final NumberFormat pnlFormat;
  final DateFormat timeFormat;

  @override
  Widget build(BuildContext context) {
    final pnl = trade.realizedPnl ?? 0;
    final accent = trade.status == 'OPEN'
        ? const Color(0xFFEE9B00)
        : pnl >= 0
        ? const Color(0xFF2A9D8F)
        : const Color(0xFF9B2226);
    final subtitle = trade.exitPrice == null
        ? 'Start ${priceFormat.format(trade.entryPrice)} | waiting for exit'
        : 'Start ${priceFormat.format(trade.entryPrice)} | End ${priceFormat.format(trade.exitPrice)}';

    return DecoratedBox(
      decoration: BoxDecoration(
        color: accent.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: accent.withValues(alpha: 0.26)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    '#${trade.tradeId} ${trade.action} ${trade.asset}',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                _TagChip(
                  label: trade.status == 'OPEN' ? 'Open' : trade.outcome,
                  color: accent,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(subtitle),
            const SizedBox(height: 6),
            Text(
              'Opened ${timeFormat.format(trade.openedAt.toLocal())}'
              '${trade.closedAt != null ? ' | Closed ${timeFormat.format(trade.closedAt!.toLocal())}' : ''}',
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                _TagChip(
                  label: 'Stop ${priceFormat.format(trade.stopLossPrice)}',
                  color: const Color(0xFF9B2226),
                ),
                _TagChip(
                  label: 'Take ${priceFormat.format(trade.takeProfitPrice)}',
                  color: const Color(0xFF2A9D8F),
                ),
                _TagChip(
                  label: trade.realizedPnl == null
                      ? 'P/L pending'
                      : 'P/L ${pnlFormat.format(trade.realizedPnl)}',
                  color: accent,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ControlPanel extends StatelessWidget {
  const _ControlPanel({
    required this.state,
    required this.reasonController,
    required this.sending,
    required this.onCommand,
  });

  final _DerivedControlState state;
  final TextEditingController reasonController;
  final bool sending;
  final Future<void> Function(String commandType) onCommand;

  @override
  Widget build(BuildContext context) {
    final buttonStyle = FilledButton.styleFrom(
      minimumSize: const Size.fromHeight(48),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    );

    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF001219),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Manual override terminal',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Buttons still route back through Spring Boot. The UI remains observer-only.',
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: Colors.white70),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: reasonController,
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration('Reason'),
            ),
            const SizedBox(height: 16),
            FilledButton(
              style: buttonStyle.copyWith(
                backgroundColor: WidgetStateProperty.all(
                  const Color(0xFF9B2226),
                ),
              ),
              onPressed: sending ? null : () => onCommand('HALT_ENGINE'),
              child: Text(state.engineHalted ? 'Halt active' : 'Halt engine'),
            ),
            const SizedBox(height: 10),
            FilledButton(
              style: buttonStyle.copyWith(
                backgroundColor: WidgetStateProperty.all(
                  const Color(0xFF2A9D8F),
                ),
              ),
              onPressed: sending ? null : () => onCommand('RESUME_ENGINE'),
              child: const Text('Resume engine'),
            ),
            const SizedBox(height: 10),
            FilledButton(
              style: buttonStyle.copyWith(
                backgroundColor: WidgetStateProperty.all(
                  const Color(0xFFEE9B00),
                ),
              ),
              onPressed: sending ? null : () => onCommand('LIQUIDATE_ALL'),
              child: Text(
                state.liquidationRequested
                    ? 'Liquidation queued'
                    : 'Liquidate all',
              ),
            ),
          ],
        ),
      ),
    );
  }

  InputDecoration _inputDecoration(String label) {
    return InputDecoration(
      labelText: label,
      labelStyle: const TextStyle(color: Colors.white70),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: Colors.white24),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: Colors.white70),
      ),
    );
  }
}

class _TagChip extends StatelessWidget {
  const _TagChip({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Text(label),
      ),
    );
  }
}

class _ReportLine extends StatelessWidget {
  const _ReportLine({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }
}

class _InfoBar extends StatelessWidget {
  const _InfoBar({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF2A9D8F).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFF2A9D8F).withValues(alpha: 0.35),
        ),
      ),
      child: Padding(padding: const EdgeInsets.all(14), child: Text(message)),
    );
  }
}

class _AlertBar extends StatelessWidget {
  const _AlertBar({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF9B2226).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFF9B2226).withValues(alpha: 0.35),
        ),
      ),
      child: Padding(padding: const EdgeInsets.all(14), child: Text(message)),
    );
  }
}

class _StaleConnectionOverlay extends StatelessWidget {
  const _StaleConnectionOverlay({
    required this.lastSyncAt,
    required this.staleReason,
  });

  final ValueNotifier<DateTime?> lastSyncAt;
  final ValueNotifier<String?> staleReason;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<int>(
      stream: Stream.periodic(const Duration(seconds: 1), (tick) => tick),
      builder: (context, snapshot) {
        return ValueListenableBuilder<DateTime?>(
          valueListenable: lastSyncAt,
          builder: (context, lastSync, _) {
            return ValueListenableBuilder<String?>(
              valueListenable: staleReason,
              builder: (context, reason, __) {
                final now = DateTime.now();
                final isStale =
                    reason != null ||
                    lastSync == null ||
                    now.difference(lastSync) > const Duration(seconds: 5);
                if (!isStale) {
                  return const SizedBox.shrink();
                }

                final detail = reason ??
                    (lastSync == null
                        ? 'No successful sync has completed yet.'
                        : 'Last sync ${now.difference(lastSync).inSeconds}s ago.');
                return Positioned(
                  top: 18,
                  left: 18,
                  right: 18,
                  child: IgnorePointer(
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: const Color(0xCC9B2226),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: const Color(0xFFFFD7D9)),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(18),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'CONNECTION LOST - DATA STALE',
                              style: Theme.of(context).textTheme.titleLarge
                                  ?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w800,
                                  ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              detail,
                              style: Theme.of(context).textTheme.bodyLarge
                                  ?.copyWith(color: Colors.white),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              },
            );
          },
        );
      },
    );
  }
}

class _MarketLineChart extends StatelessWidget {
  const _MarketLineChart({
    required this.chart,
    required this.trades,
  });

  final MarketChart chart;
  final List<PaperTradeReport> trades;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _MarketLineChartPainter(chart: chart, trades: trades),
      child: Container(),
    );
  }
}

class _MarketLineChartPainter extends CustomPainter {
  _MarketLineChartPainter({
    required this.chart,
    required this.trades,
  });

  final MarketChart chart;
  final List<PaperTradeReport> trades;

  @override
  void paint(Canvas canvas, Size size) {
    final candles = chart.candles;
    if (candles.length < 2) {
      return;
    }

    const padding = 18.0;
    const rightAxisWidth = 124.0;
    final tradePrices = trades.expand((trade) => [
          trade.entryPrice.toDouble(),
          trade.stopLossPrice.toDouble(),
          trade.takeProfitPrice.toDouble(),
          if (trade.exitPrice != null) trade.exitPrice!.toDouble(),
        ]);
    final minPrice = [
      candles.map((candle) => candle.low.toDouble()).reduce(math.min),
      if (trades.isNotEmpty) tradePrices.reduce(math.min),
    ].reduce(math.min);
    final maxPrice = [
      candles.map((candle) => candle.high.toDouble()).reduce(math.max),
      if (trades.isNotEmpty) trades
          .expand((trade) => [
                trade.entryPrice.toDouble(),
                trade.stopLossPrice.toDouble(),
                trade.takeProfitPrice.toDouble(),
                if (trade.exitPrice != null) trade.exitPrice!.toDouble(),
              ])
          .reduce(math.max),
    ].reduce(math.max);
    final range = maxPrice - minPrice;
    final safeRange = range == 0 ? 1.0 : range;
    final plotRight = size.width - padding - rightAxisWidth;
    final chartWidth = plotRight - padding;
    final chartHeight = size.height - padding * 2;
    final axisStep = _niceAxisStep(safeRange / 4);
    final axisTop = (maxPrice / axisStep).ceil() * axisStep;
    final axisBottom = (minPrice / axisStep).floor() * axisStep;
    final axisRange = math.max(axisTop - axisBottom, axisStep);
    final axisDecimals = _axisDecimals(axisStep);

    double xForIndex(int index) =>
        padding + chartWidth * index / (candles.length - 1);

    double yForPrice(double price) {
      final y =
          padding + chartHeight * (1 - ((price - axisBottom) / axisRange));
      return y.clamp(padding, size.height - padding);
    }

    void drawPriceTag({
      required String text,
      required double y,
      required Color color,
      double? x,
      bool boxed = true,
    }) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: text,
          style: TextStyle(
            color: boxed ? Colors.white : const Color(0xFF334155),
            fontSize: 11,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: rightAxisWidth - 12);
      final dx = (x ?? (size.width - padding - textPainter.width)).clamp(
        padding,
        size.width - textPainter.width - 4,
      );
      final dy = (y - textPainter.height / 2).clamp(
        padding,
        size.height - padding - textPainter.height,
      );
      if (boxed) {
        final boxRect = RRect.fromRectAndRadius(
          Rect.fromLTWH(
            dx - 4,
            dy - 2,
            textPainter.width + 8,
            textPainter.height + 4,
          ),
          const Radius.circular(6),
        );
        canvas.drawRRect(
          boxRect,
          Paint()..color = color,
        );
      }
      textPainter.paint(canvas, Offset(dx, dy));
    }

    final gridPaint = Paint()
      ..color = const Color(0x22005F73)
      ..strokeWidth = 1;
    for (var i = 0; i < 5; i += 1) {
      final priceAtTick = axisTop - (axisStep * i);
      final dy = yForPrice(priceAtTick);
      canvas.drawLine(
        Offset(padding, dy),
        Offset(plotRight, dy),
        gridPaint,
      );
      drawPriceTag(
        text: priceAtTick.toStringAsFixed(axisDecimals),
        y: dy,
        color: const Color(0xFF334155),
        x: plotRight + 8,
        boxed: false,
      );
    }

    canvas.drawLine(
      Offset(plotRight, padding),
      Offset(plotRight, size.height - padding),
      Paint()
        ..color = const Color(0x33005F73)
        ..strokeWidth = 1,
    );

    final linePaint = Paint()
      ..color = const Color(0xFF005F73)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    final fillPaint = Paint()
      ..shader = const LinearGradient(
        colors: [Color(0x55005F73), Color(0x00005F73)],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    final linePath = Path();
    final fillPath = Path();
    for (var index = 0; index < candles.length; index += 1) {
      final candle = candles[index];
      final x = xForIndex(index);
      final y = yForPrice(candle.close.toDouble());
      if (index == 0) {
        linePath.moveTo(x, y);
        fillPath.moveTo(x, size.height - padding);
        fillPath.lineTo(x, y);
      } else {
        linePath.lineTo(x, y);
        fillPath.lineTo(x, y);
      }
    }
    fillPath
      ..lineTo(plotRight, size.height - padding)
      ..close();
    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(linePath, linePaint);

    final latestPrice = candles.last.close.toDouble();
    final latestX = xForIndex(candles.length - 1);
    final latestY = yForPrice(latestPrice);
    final livePricePaint = Paint()
      ..color = const Color(0xFF1D4ED8)
      ..strokeWidth = 1.2;
    canvas.drawLine(
      Offset(padding, latestY),
      Offset(plotRight, latestY),
      livePricePaint,
    );
    canvas.drawCircle(
      Offset(latestX, latestY),
      5,
      Paint()..color = const Color(0xFF1D4ED8),
    );
    drawPriceTag(
      text: 'Live ${latestPrice.toStringAsFixed(axisDecimals)}',
      y: latestY,
      color: const Color(0xFF1D4ED8),
      x: plotRight + 8,
    );

    final highlightTrade = trades.isEmpty ? null : trades.first;
    if (highlightTrade != null) {
      final openIndex = _nearestIndex(candles, highlightTrade.openedAt);
      final openX = xForIndex(openIndex);
      final closeX = highlightTrade.closedAt != null
          ? xForIndex(_nearestIndex(candles, highlightTrade.closedAt!))
          : latestX;
      final left = math.min(openX, closeX);
      final right = math.max(openX, closeX);
      final entryY = yForPrice(highlightTrade.entryPrice.toDouble());
      final stopY = yForPrice(highlightTrade.stopLossPrice.toDouble());
      final takeY = yForPrice(highlightTrade.takeProfitPrice.toDouble());

      final redBand = Rect.fromLTRB(
        padding,
        math.min(entryY, stopY),
        plotRight,
        math.max(entryY, stopY),
      );
      final greenBand = Rect.fromLTRB(
        padding,
        math.min(entryY, takeY),
        plotRight,
        math.max(entryY, takeY),
      );
      canvas.drawRect(
        redBand,
        Paint()..color = const Color(0x339B2226),
      );
      canvas.drawRect(
        greenBand,
        Paint()..color = const Color(0x332A9D8F),
      );

      final openGuidePaint = Paint()
        ..color = const Color(0xFF1D4ED8)
      ..strokeWidth = 2;
      final stopGuidePaint = Paint()
        ..color = const Color(0xFF9B2226)
        ..strokeWidth = 2;
      final takeGuidePaint = Paint()
        ..color = const Color(0xFF2A9D8F)
        ..strokeWidth = 2;

      canvas.drawLine(
        Offset(padding, entryY),
        Offset(plotRight, entryY),
        openGuidePaint,
      );
      canvas.drawLine(
        Offset(padding, stopY),
        Offset(plotRight, stopY),
        stopGuidePaint,
      );
      canvas.drawLine(
        Offset(padding, takeY),
        Offset(plotRight, takeY),
        takeGuidePaint,
      );

      drawPriceTag(
        text: 'OPEN ${highlightTrade.entryPrice.toStringAsFixed(axisDecimals)}',
        y: entryY,
        color: const Color(0xFF1D4ED8),
        x: plotRight + 8,
      );
      drawPriceTag(
        text: 'SL ${highlightTrade.stopLossPrice.toStringAsFixed(axisDecimals)}',
        y: stopY,
        color: const Color(0xFF9B2226),
        x: plotRight + 8,
      );
      drawPriceTag(
        text: 'TP ${highlightTrade.takeProfitPrice.toStringAsFixed(axisDecimals)}',
        y: takeY,
        color: const Color(0xFF2A9D8F),
        x: plotRight + 8,
      );
    }

    for (final trade in trades) {
      final openIndex = _nearestIndex(candles, trade.openedAt);
      final openOffset = Offset(
        xForIndex(openIndex),
        yForPrice(trade.entryPrice.toDouble()),
      );
      final entryPaint = Paint()
        ..color = trade.action == 'BUY'
            ? const Color(0xFF2A9D8F)
            : const Color(0xFFEE9B00);
      canvas.drawCircle(openOffset, 6, entryPaint);
      canvas.drawCircle(
        openOffset,
        10,
        Paint()
          ..color = entryPaint.color.withValues(alpha: 0.12)
          ..style = PaintingStyle.fill,
      );

      if (trade.closedAt != null && trade.exitPrice != null) {
        final closeIndex = _nearestIndex(candles, trade.closedAt!);
        final closeOffset = Offset(
          xForIndex(closeIndex),
          yForPrice(trade.exitPrice!.toDouble()),
        );
        final exitPaint = Paint()
          ..color = trade.outcome == 'PROFIT'
              ? const Color(0xFF2A9D8F)
              : trade.outcome == 'LOSS'
              ? const Color(0xFF9B2226)
              : const Color(0xFF6C757D);
        canvas.drawCircle(closeOffset, 6, exitPaint);
        canvas.drawLine(openOffset, closeOffset, exitPaint..strokeWidth = 2);
      }
    }
  }

  int _nearestIndex(List<MarketChartCandle> candles, DateTime timestamp) {
    var nearestIndex = 0;
    var nearestDelta = (candles.first.timestamp.difference(timestamp)).abs();
    for (var index = 1; index < candles.length; index += 1) {
      final delta = (candles[index].timestamp.difference(timestamp)).abs();
      if (delta < nearestDelta) {
        nearestIndex = index;
        nearestDelta = delta;
      }
    }
    return nearestIndex;
  }

  @override
  bool shouldRepaint(covariant _MarketLineChartPainter oldDelegate) {
    return oldDelegate.chart != chart || oldDelegate.trades != trades;
  }

  double _niceAxisStep(double rawStep) {
    if (rawStep <= 0) {
      return 1;
    }

    final exponent = math.pow(10, (math.log(rawStep) / math.ln10).floor())
        .toDouble();
    final fraction = rawStep / exponent;

    final niceFraction = switch (fraction) {
      <= 1 => 1.0,
      <= 2 => 2.0,
      <= 2.5 => 2.5,
      <= 5 => 5.0,
      _ => 10.0,
    };

    return niceFraction * exponent;
  }

  int _axisDecimals(double step) {
    if (step >= 1000) {
      return 0;
    }
    if (step >= 1) {
      return 2;
    }
    final decimals = (-math.log(step) / math.ln10).ceil() + 1;
    return decimals.clamp(2, 8);
  }
}

class _DerivedControlState {
  const _DerivedControlState({
    required this.engineHalted,
    required this.liquidationRequested,
  });

  final bool engineHalted;
  final bool liquidationRequested;
}

class _DerivedMetrics {
  const _DerivedMetrics({
    required this.totalOrders,
    required this.executedOrders,
    required this.rejectedOrders,
    required this.openTrades,
    required this.profitTrades,
    required this.lossTrades,
    required this.netPnl,
    required this.averageLatencyMs,
  });

  final int totalOrders;
  final int executedOrders;
  final int rejectedOrders;
  final int openTrades;
  final int profitTrades;
  final int lossTrades;
  final num netPnl;
  final num averageLatencyMs;

  factory _DerivedMetrics.fromSnapshot(DashboardSnapshot? snapshot) {
    return _DerivedMetrics(
      totalOrders: snapshot?.totalOrders ?? 0,
      executedOrders: snapshot?.executedOrders ?? 0,
      rejectedOrders: snapshot?.rejectedOrders ?? 0,
      openTrades: 0,
      profitTrades: 0,
      lossTrades: 0,
      netPnl: 0,
      averageLatencyMs: snapshot?.averageExecutionLatencyMs ?? 0,
    );
  }
}

class _LocalDashboardData {
  const _LocalDashboardData({
    required this.snapshot,
    required this.trades,
    required this.controlState,
  });

  final DashboardSnapshot snapshot;
  final List<PaperTradeReport> trades;
  final Map<String, dynamic> controlState;
}

_DerivedControlState _deriveControlState(List<Map<String, dynamic>> rows) {
  DateTime? latestHaltTimestamp;
  bool engineHalted = false;
  DateTime? latestLiquidationTimestamp;
  bool liquidationRequested = false;

  for (final row in rows) {
    final status = row['status']?.toString();
    if (status != null && status != 'ACCEPTED' && status != 'ACTIVE' && status != 'CLEARED') {
      continue;
    }
    final type = row['command_type']?.toString();
    final createdAt =
        DateTime.tryParse(row['created_at']?.toString() ?? '') ??
        DateTime.fromMillisecondsSinceEpoch(0, isUtc: true);

    if ((type == 'HALT_ENGINE' || type == 'RESUME_ENGINE') &&
        (latestHaltTimestamp == null || createdAt.isAfter(latestHaltTimestamp))) {
      latestHaltTimestamp = createdAt;
      engineHalted = type == 'HALT_ENGINE';
    }
    if ((type == 'LIQUIDATE_ALL' || type == 'CLEAR_LIQUIDATION') &&
        (latestLiquidationTimestamp == null ||
            createdAt.isAfter(latestLiquidationTimestamp))) {
      latestLiquidationTimestamp = createdAt;
      liquidationRequested = type == 'LIQUIDATE_ALL';
    }
  }

  return _DerivedControlState(
    engineHalted: engineHalted,
    liquidationRequested: liquidationRequested,
  );
}

_DerivedMetrics _deriveMetrics({
  required List<Map<String, dynamic>> orders,
  required List<PaperTradeReport> trades,
  required List<Map<String, dynamic>> telemetry,
}) {
  final totalOrders = orders.length;
  final executedOrders = orders
      .where((row) => row['status']?.toString() == 'EXECUTED')
      .length;
  final rejectedOrders = orders
      .where((row) => row['status']?.toString() == 'REJECTED')
      .length;
  final openTrades = trades.where((trade) => trade.status == 'OPEN').length;
  final closedTrades = trades.where((trade) => trade.status == 'CLOSED');
  final profitTrades =
      closedTrades.where((trade) => (trade.realizedPnl ?? 0) > 0).length;
  final lossTrades =
      closedTrades.where((trade) => (trade.realizedPnl ?? 0) < 0).length;
  final netPnl = closedTrades.fold<num>(
    0,
    (total, trade) => total + (trade.realizedPnl ?? 0),
  );

  final latencyValues = telemetry
      .where((row) => row['metric_name']?.toString() == 'signal_to_fill_latency')
      .map((row) => num.tryParse(row['metric_value']?.toString() ?? ''))
      .whereType<num>()
      .toList();
  final averageLatencyMs = latencyValues.isEmpty
      ? 0
      : latencyValues.reduce((a, b) => a + b) / latencyValues.length;

  return _DerivedMetrics(
    totalOrders: totalOrders,
    executedOrders: executedOrders,
    rejectedOrders: rejectedOrders,
    openTrades: openTrades,
    profitTrades: profitTrades,
    lossTrades: lossTrades,
    netPnl: netPnl,
    averageLatencyMs: averageLatencyMs,
  );
}

String _formatAssetPrice(num value) {
  final absolute = value.abs();
  if (absolute >= 1000) {
    return value.toStringAsFixed(2);
  }
  if (absolute >= 1) {
    return value.toStringAsFixed(4);
  }
  if (absolute >= 0.01) {
    return value.toStringAsFixed(6);
  }
  return value.toStringAsFixed(8);
}
