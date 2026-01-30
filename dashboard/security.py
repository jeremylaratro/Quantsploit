#!/usr/bin/env python3
"""
Quantsploit Dashboard Security Module
Centralized security functions for authentication, rate limiting,
CSRF protection, security headers, and thread safety.
"""

import os
import re
import secrets
import threading
from functools import wraps
from typing import Dict, Any, Optional, Callable

from flask import Flask, request, session, redirect, url_for, jsonify, Response


# =============================================================================
# THREAD SAFETY - Locks for global dictionaries
# =============================================================================

class ThreadSafeDict:
    """Thread-safe dictionary wrapper using RLock for nested lock acquisition."""

    def __init__(self):
        self._dict: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._dict[key]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._dict

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._dict.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._dict.pop(key, default)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            self._dict.update(other)

    def clear(self) -> None:
        with self._lock:
            self._dict.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the internal dict."""
        with self._lock:
            return dict(self._dict)


# =============================================================================
# AUTHENTICATION
# =============================================================================

# Authentication configuration
AUTH_ENABLED = True
AUTH_SESSION_KEY = 'authenticated'
AUTH_COOKIE_NAME = 'qs_session'


def get_dashboard_password() -> Optional[str]:
    """Get dashboard password from environment variable."""
    return os.environ.get('DASHBOARD_PASSWORD')


def check_password(password: str) -> bool:
    """Verify password against environment variable."""
    expected = get_dashboard_password()
    if not expected:
        return False
    # Use secrets.compare_digest for timing-safe comparison
    return secrets.compare_digest(password.encode(), expected.encode())


def is_authenticated() -> bool:
    """Check if current session is authenticated."""
    if not AUTH_ENABLED:
        return True
    if not get_dashboard_password():
        # No password configured - auth disabled
        return True
    return session.get(AUTH_SESSION_KEY, False)


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# RATE LIMITING
# =============================================================================

# Rate limit storage (IP -> list of request timestamps)
_rate_limit_storage: Dict[str, list] = {}
_rate_limit_lock = threading.Lock()

# Default limits
RATE_LIMIT_DEFAULT = 60  # requests per minute
RATE_LIMIT_EXPENSIVE = 10  # for expensive operations


def get_client_ip() -> str:
    """Get client IP address, handling proxies."""
    # Check for X-Forwarded-For header (behind proxy)
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def check_rate_limit(limit: int = RATE_LIMIT_DEFAULT, window: int = 60) -> bool:
    """
    Check if request is within rate limit.

    Args:
        limit: Maximum requests allowed in window
        window: Time window in seconds

    Returns:
        True if within limit, False if exceeded
    """
    import time

    ip = get_client_ip()
    current_time = time.time()

    with _rate_limit_lock:
        if ip not in _rate_limit_storage:
            _rate_limit_storage[ip] = []

        # Remove timestamps outside window
        _rate_limit_storage[ip] = [
            ts for ts in _rate_limit_storage[ip]
            if current_time - ts < window
        ]

        # Check if within limit
        if len(_rate_limit_storage[ip]) >= limit:
            return False

        # Add current request timestamp
        _rate_limit_storage[ip].append(current_time)
        return True


def rate_limit(limit: int = RATE_LIMIT_DEFAULT, window: int = 60) -> Callable:
    """
    Decorator to apply rate limiting to a route.

    Args:
        limit: Maximum requests allowed in window
        window: Time window in seconds
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not check_rate_limit(limit, window):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': window
                }), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# =============================================================================
# CSRF PROTECTION
# =============================================================================

CSRF_TOKEN_KEY = 'csrf_token'
CSRF_HEADER_NAME = 'X-CSRF-Token'


def generate_csrf_token() -> str:
    """Generate a new CSRF token and store in session."""
    if CSRF_TOKEN_KEY not in session:
        session[CSRF_TOKEN_KEY] = secrets.token_hex(32)
    return session[CSRF_TOKEN_KEY]


def validate_csrf_token(token: str) -> bool:
    """Validate CSRF token against session."""
    expected = session.get(CSRF_TOKEN_KEY)
    if not expected or not token:
        return False
    return secrets.compare_digest(token, expected)


def csrf_protect(f: Callable) -> Callable:
    """Decorator to require CSRF token for POST/PUT/DELETE requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            # Check header first, then form data
            token = request.headers.get(CSRF_HEADER_NAME)
            if not token:
                token = request.form.get('csrf_token')
            if not token and request.is_json:
                token = request.json.get('csrf_token') if request.json else None

            if not validate_csrf_token(token):
                return jsonify({'error': 'Invalid or missing CSRF token'}), 403
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# INPUT VALIDATION
# =============================================================================

# Timestamp format: YYYYmmdd_HHMMSS or YYYYmmdd_HHMMSS_N
TIMESTAMP_PATTERN = re.compile(r'^[0-9]{8}_[0-9]{6}(_[0-9]+)?$')

# Ticker symbol format: 1-5 uppercase letters or letters with dots (BRK.B)
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')

# Scanner ID format: lowercase letters, numbers, underscores
SCANNER_ID_PATTERN = re.compile(r'^[a-z0-9_]+$')


def validate_timestamp(timestamp: str) -> bool:
    """Validate timestamp parameter format."""
    if not timestamp or not isinstance(timestamp, str):
        return False
    return bool(TIMESTAMP_PATTERN.match(timestamp))


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    if not ticker or not isinstance(ticker, str):
        return False
    return bool(TICKER_PATTERN.match(ticker.upper()))


def validate_scanner_id(scanner_id: str) -> bool:
    """Validate scanner ID format."""
    if not scanner_id or not isinstance(scanner_id, str):
        return False
    return bool(SCANNER_ID_PATTERN.match(scanner_id))


def sanitize_for_json(data: Any) -> Any:
    """
    Recursively sanitize data for safe JSON output.
    Escapes HTML-sensitive characters in strings.
    """
    if isinstance(data, str):
        return (data
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#x27;'))
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    return data


# =============================================================================
# SECURITY HEADERS
# =============================================================================

def add_security_headers(response: Response) -> Response:
    """Add security headers to response."""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'

    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # XSS protection (legacy but still useful)
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Content Security Policy - allow CDN resources used by dashboard
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.jsdelivr.net https://cdn.plot.ly https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'"
    )
    response.headers['Content-Security-Policy'] = csp

    # Permissions policy (replaces Feature-Policy)
    response.headers['Permissions-Policy'] = (
        'geolocation=(), microphone=(), camera=(), payment=()'
    )

    return response


# =============================================================================
# FLASK INITIALIZATION
# =============================================================================

def init_security(app: Flask) -> None:
    """
    Initialize security features for Flask app.

    Args:
        app: Flask application instance
    """
    # Configure secret key
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key:
        secret_key = secrets.token_hex(32)
        print("[SECURITY] Warning: Using auto-generated secret key. Set FLASK_SECRET_KEY for production.")
    app.secret_key = secret_key

    # Session configuration
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('HTTPS_ENABLED', 'false').lower() == 'true'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600 * 8  # 8 hours

    # Add security headers to all responses
    @app.after_request
    def apply_security_headers(response):
        return add_security_headers(response)

    # Make CSRF token available in templates
    @app.context_processor
    def inject_csrf_token():
        return {'csrf_token': generate_csrf_token}

    # Log security status
    password_set = bool(get_dashboard_password())
    print(f"[SECURITY] Authentication: {'ENABLED' if password_set else 'DISABLED (no DASHBOARD_PASSWORD set)'}")
    print(f"[SECURITY] Rate limiting: {RATE_LIMIT_DEFAULT}/min (general), {RATE_LIMIT_EXPENSIVE}/min (expensive)")
    print(f"[SECURITY] CSRF protection: ENABLED")
    print(f"[SECURITY] Security headers: ENABLED")
