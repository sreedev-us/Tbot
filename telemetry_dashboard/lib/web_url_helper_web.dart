// ignore_for_file: avoid_web_libraries_in_flutter, deprecated_member_use

import 'dart:html' as html;

bool hasAuthRedirectParams() {
  final params = Uri.base.queryParameters;
  return params.containsKey('code') ||
      params.containsKey('access_token') ||
      params.containsKey('refresh_token') ||
      params.containsKey('token_hash');
}

void clearAuthRedirectFromUrl() {
  final current = Uri.base;
  final cleared = current.replace(queryParameters: const {}, fragment: '');
  html.window.history.replaceState(null, '', cleared.toString());
}
