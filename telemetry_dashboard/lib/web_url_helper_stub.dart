bool hasAuthRedirectParams() {
  final params = Uri.base.queryParameters;
  return params.containsKey('code') ||
      params.containsKey('access_token') ||
      params.containsKey('refresh_token') ||
      params.containsKey('token_hash');
}

void clearAuthRedirectFromUrl() {}
