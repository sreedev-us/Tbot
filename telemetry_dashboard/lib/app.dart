import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'models/dashboard_snapshot.dart';
import 'services/backend_api.dart';

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

class _AuthGate extends StatelessWidget {
  const _AuthGate();

  @override
  Widget build(BuildContext context) {
    final auth = Supabase.instance.client.auth;

    return StreamBuilder<AuthState>(
      stream: auth.onAuthStateChange,
      initialData: AuthState(
        AuthChangeEvent.initialSession,
        auth.currentSession,
      ),
      builder: (context, snapshot) {
        final session = snapshot.data?.session ?? auth.currentSession;
        if (session == null) {
          return const _AdminSignInPage();
        }

        return DashboardPage(userEmail: session.user.email ?? session.user.id);
      },
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({required this.userEmail, super.key});

  final String userEmail;

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final BackendApi _backendApi = BackendApi();
  final TextEditingController _operatorController = TextEditingController(
    text: 'ops@tbot',
  );
  final TextEditingController _reasonController = TextEditingController(
    text: 'manual override',
  );
  final DateFormat _timeFormat = DateFormat('MMM d, HH:mm:ss');

  Timer? _refreshTimer;
  DashboardSnapshot? _snapshot;
  bool _loadingSnapshot = true;
  bool _sendingCommand = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadSnapshot();
    _refreshTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _loadSnapshot(silent: true),
    );
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _operatorController.dispose();
    _reasonController.dispose();
    super.dispose();
  }

  Future<void> _loadSnapshot({bool silent = false}) async {
    if (!silent) {
      setState(() {
        _loadingSnapshot = true;
        _error = null;
      });
    }

    try {
      final snapshot = await _backendApi.fetchSnapshot();
      if (!mounted) {
        return;
      }
      setState(() {
        _snapshot = snapshot;
        _loadingSnapshot = false;
        _error = null;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _loadingSnapshot = false;
        _error = error.toString();
      });
    }
  }

  Future<void> _sendCommand(String commandType) async {
    setState(() {
      _sendingCommand = true;
      _error = null;
    });

    try {
      await _backendApi.sendCommand(
        commandType: commandType,
        initiatedBy: _operatorController.text.trim(),
        reason: _reasonController.text.trim(),
      );
      await _loadSnapshot(silent: true);
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = error.toString();
      });
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
    final snapshot = _snapshot;
    final width = MediaQuery.sizeOf(context).width;
    final wide = width > 1100;

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
          child: RefreshIndicator(
            onRefresh: _loadSnapshot,
            child: ListView(
              padding: const EdgeInsets.all(24),
              children: [
                Wrap(
                  spacing: 16,
                  runSpacing: 16,
                  alignment: WrapAlignment.spaceBetween,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 680),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'tbot telemetry',
                            style: Theme.of(context).textTheme.displaySmall
                                ?.copyWith(fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Flutter observes. Spring Boot decides. Supabase streams the state ledger in real time.',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Realtime access is restricted to the authenticated admin account.',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        ],
                      ),
                    ),
                    _StatusBanner(
                      snapshot: snapshot,
                      userEmail: widget.userEmail,
                    ),
                  ],
                ),
                const SizedBox(height: 24),
                if (_error != null) ...[
                  _AlertBar(message: _error!),
                  const SizedBox(height: 16),
                ],
                if (_loadingSnapshot && snapshot == null)
                  const Padding(
                    padding: EdgeInsets.all(40),
                    child: Center(child: CircularProgressIndicator()),
                  )
                else
                  Wrap(
                    spacing: 16,
                    runSpacing: 16,
                    children: [
                      _MetricCard(
                        title: 'Orders',
                        value: '${snapshot?.totalOrders ?? 0}',
                        detail:
                            '${snapshot?.executedOrders ?? 0} executed / ${snapshot?.rejectedOrders ?? 0} rejected',
                        accent: const Color(0xFF005F73),
                      ),
                      _MetricCard(
                        title: 'Exposure',
                        value: '\$${snapshot?.currentExposure ?? 0}',
                        detail: 'Tracked by backend risk service',
                        accent: const Color(0xFF0A9396),
                      ),
                      _MetricCard(
                        title: 'Executed',
                        value: '\$${snapshot?.totalExecutedNotional ?? 0}',
                        detail: 'Filled notional in ledger',
                        accent: const Color(0xFFEE9B00),
                      ),
                      _MetricCard(
                        title: 'Latency',
                        value:
                            '${(snapshot?.averageExecutionLatencyMs ?? 0).toStringAsFixed(1)} ms',
                        detail: snapshot == null
                            ? 'Waiting for fills'
                            : 'Snapshot ${_timeFormat.format(snapshot.generatedAt.toLocal())}',
                        accent: const Color(0xFF9B2226),
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
                        children: const [
                          _RealtimeTableCard(
                            title: 'Orders stream',
                            tableName: 'orders',
                            timestampField: 'received_at',
                            fields: [
                              'signal_id',
                              'asset',
                              'status',
                              'exchange_name',
                            ],
                          ),
                          SizedBox(height: 16),
                          _RealtimeTableCard(
                            title: 'Executions stream',
                            tableName: 'executions',
                            timestampField: 'fill_confirmed_at',
                            fields: [
                              'venue_order_id',
                              'status',
                              'executed_notional',
                              'slippage_fee',
                            ],
                          ),
                          SizedBox(height: 16),
                          _RealtimeTableCard(
                            title: 'Telemetry stream',
                            tableName: 'system_telemetry',
                            timestampField: 'recorded_at',
                            fields: [
                              'metric_group',
                              'metric_name',
                              'metric_value',
                              'metric_unit',
                            ],
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
                            snapshot: snapshot,
                            operatorController: _operatorController,
                            reasonController: _reasonController,
                            sending: _sendingCommand,
                            onCommand: _sendCommand,
                          ),
                          const SizedBox(height: 16),
                          const _RealtimeTableCard(
                            title: 'Control command log',
                            tableName: 'control_commands',
                            timestampField: 'created_at',
                            fields: [
                              'command_type',
                              'status',
                              'initiated_by',
                              'reason',
                            ],
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AdminSignInPage extends StatefulWidget {
  const _AdminSignInPage();

  @override
  State<_AdminSignInPage> createState() => _AdminSignInPageState();
}

class _AdminSignInPageState extends State<_AdminSignInPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _submitting = false;
  String? _error;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _signIn() async {
    setState(() {
      _submitting = true;
      _error = null;
    });

    try {
      await Supabase.instance.client.auth.signInWithPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text,
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
                          'Supabase realtime is locked to your authenticated admin account. Sign in before loading any live trading data.',
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                        const SizedBox(height: 20),
                        TextField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          autofillHints: const [AutofillHints.username],
                          decoration: _signInDecoration('Admin email'),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _passwordController,
                          obscureText: true,
                          autofillHints: const [AutofillHints.password],
                          decoration: _signInDecoration('Password'),
                          onSubmitted: (_) => _submitting ? null : _signIn(),
                        ),
                        if (_error != null) ...[
                          const SizedBox(height: 14),
                          _AlertBar(message: _error!),
                        ],
                        const SizedBox(height: 18),
                        FilledButton(
                          onPressed: _submitting ? null : _signIn,
                          style: FilledButton.styleFrom(
                            minimumSize: const Size.fromHeight(50),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          child: Text(
                            _submitting ? 'Signing in...' : 'Sign in',
                          ),
                        ),
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
  const _StatusBanner({required this.snapshot, required this.userEmail});

  final DashboardSnapshot? snapshot;
  final String userEmail;

  @override
  Widget build(BuildContext context) {
    final halted = snapshot?.engineHalted ?? false;
    final liquidation = snapshot?.liquidationRequested ?? false;
    final color = halted
        ? const Color(0xFF9B2226)
        : liquidation
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
              halted
                  ? 'Engine halted'
                  : liquidation
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
            OutlinedButton(
              onPressed: () => Supabase.instance.client.auth.signOut(),
              child: const Text('Sign out'),
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

class _RealtimeTableCard extends StatelessWidget {
  const _RealtimeTableCard({
    required this.title,
    required this.tableName,
    required this.timestampField,
    required this.fields,
  });

  final String title;
  final String tableName;
  final String timestampField;
  final List<String> fields;

  @override
  Widget build(BuildContext context) {
    final stream = Supabase.instance.client
        .from(tableName)
        .stream(primaryKey: ['id'])
        .order(timestampField, ascending: false);

    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.84),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: const Color(0xFF005F73).withValues(alpha: 0.12),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 12),
            StreamBuilder<List<Map<String, dynamic>>>(
              stream: stream,
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  return Text('Stream error: ${snapshot.error}');
                }
                if (!snapshot.hasData) {
                  return const Padding(
                    padding: EdgeInsets.all(24),
                    child: Center(child: CircularProgressIndicator()),
                  );
                }

                final rows = snapshot.data!.take(12).toList();
                if (rows.isEmpty) {
                  return const Padding(
                    padding: EdgeInsets.symmetric(vertical: 24),
                    child: Text('No rows received yet.'),
                  );
                }

                return SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: DataTable(
                    columns: [
                      const DataColumn(label: Text('Time')),
                      ...fields.map((field) => DataColumn(label: Text(field))),
                    ],
                    rows: rows
                        .map(
                          (row) => DataRow(
                            cells: [
                              DataCell(Text('${row[timestampField] ?? '-'}')),
                              ...fields.map(
                                (field) => DataCell(
                                  ConstrainedBox(
                                    constraints: const BoxConstraints(
                                      maxWidth: 220,
                                    ),
                                    child: Text(
                                      '${row[field] ?? '-'}',
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        )
                        .toList(),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _ControlPanel extends StatelessWidget {
  const _ControlPanel({
    required this.snapshot,
    required this.operatorController,
    required this.reasonController,
    required this.sending,
    required this.onCommand,
  });

  final DashboardSnapshot? snapshot;
  final TextEditingController operatorController;
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
              'Buttons only talk to Spring Boot. Risk and execution authority stays server-side.',
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: Colors.white70),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: operatorController,
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration('Operator identity'),
            ),
            const SizedBox(height: 12),
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
              child: Text(
                snapshot?.engineHalted == true ? 'Halt active' : 'Halt engine',
              ),
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
                snapshot?.liquidationRequested == true
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
