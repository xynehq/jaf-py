"""
ADK Security Validation - Authentication and Authorization

This module provides security validation functions for authentication,
authorization, and access control in production environments.
"""

import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(str, Enum):
    """Security levels for validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AdkSecurityConfig:
    """Security configuration for the ADK."""
    api_key_header: str = "X-ADK-API-Key"
    session_token_header: str = "X-ADK-Session-Token"
    rate_limit_window_seconds: int = 3600  # 1 hour
    max_requests_per_window: int = 1000
    max_prompt_length: int = 10000
    max_session_length_hours: int = 24
    require_https: bool = True
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    allowed_origins: Optional[List[str]] = None


@dataclass(frozen=True)
class ValidationResult:
    """Result of a security validation."""
    is_valid: bool
    error_message: Optional[str] = None
    security_level: Optional[SecurityLevel] = None
    metadata: Optional[Dict[str, Any]] = None


class AdkAuthenticator(Protocol):
    """Protocol for authentication implementations."""
    
    async def authenticate(self, api_key: str) -> ValidationResult:
        """Authenticate an API key."""
        ...
    
    async def validate_session(self, session_token: str) -> ValidationResult:
        """Validate a session token."""
        ...


class AdkAuthorizer(Protocol):
    """Protocol for authorization implementations."""
    
    async def authorize(self, user_id: str, resource: str, action: str) -> ValidationResult:
        """Check if user is authorized for an action on a resource."""
        ...
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions."""
        ...


class AdkRateLimiter(Protocol):
    """Protocol for rate limiting implementations."""
    
    async def check_rate_limit(self, identifier: str, window_seconds: int, max_requests: int) -> ValidationResult:
        """Check if rate limit is exceeded."""
        ...
    
    async def increment_request_count(self, identifier: str) -> None:
        """Increment request count for rate limiting."""
        ...


class AdkSecurityValidator:
    """
    Production-ready security validator with multiple validation layers.
    """
    
    def __init__(
        self,
        config: AdkSecurityConfig,
        authenticator: Optional[AdkAuthenticator] = None,
        authorizer: Optional[AdkAuthorizer] = None,
        rate_limiter: Optional[AdkRateLimiter] = None
    ):
        """
        Initialize the security validator.
        
        Args:
            config: Security configuration
            authenticator: Authentication implementation
            authorizer: Authorization implementation  
            rate_limiter: Rate limiting implementation
        """
        self.config = config
        self.authenticator = authenticator
        self.authorizer = authorizer
        self.rate_limiter = rate_limiter
    
    async def validate_request(
        self,
        headers: Dict[str, str],
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a complete request with all security checks.
        
        Args:
            headers: Request headers
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            ValidationResult indicating if request is valid
        """
        # Check API key authentication
        api_key = headers.get(self.config.api_key_header)
        if api_key and self.authenticator:
            auth_result = await self.authenticator.authenticate(api_key)
            if not auth_result.is_valid:
                return auth_result
        
        # Check session token
        session_token = headers.get(self.config.session_token_header)
        if session_token and self.authenticator:
            session_result = await self.authenticator.validate_session(session_token)
            if not session_result.is_valid:
                return session_result
        
        # Check authorization
        if user_id and resource and action and self.authorizer:
            authz_result = await self.authorizer.authorize(user_id, resource, action)
            if not authz_result.is_valid:
                return authz_result
        
        # Check rate limits
        if user_id and self.rate_limiter:
            rate_result = await self.rate_limiter.check_rate_limit(
                user_id,
                self.config.rate_limit_window_seconds,
                self.config.max_requests_per_window
            )
            if not rate_result.is_valid:
                return rate_result
        
        return ValidationResult(is_valid=True)
    
    def validate_https_requirement(self, scheme: str) -> ValidationResult:
        """
        Validate HTTPS requirement.
        
        Args:
            scheme: URL scheme (http or https)
            
        Returns:
            ValidationResult
        """
        if self.config.require_https and scheme.lower() != "https":
            return ValidationResult(
                is_valid=False,
                error_message="HTTPS required for security",
                security_level=SecurityLevel.HIGH
            )
        
        return ValidationResult(is_valid=True)
    
    def validate_origin(self, origin: Optional[str]) -> ValidationResult:
        """
        Validate request origin for CORS.
        
        Args:
            origin: Request origin
            
        Returns:
            ValidationResult
        """
        if not self.config.allowed_origins:
            return ValidationResult(is_valid=True)
        
        if not origin:
            return ValidationResult(
                is_valid=False,
                error_message="Origin header required",
                security_level=SecurityLevel.MEDIUM
            )
        
        if origin not in self.config.allowed_origins:
            return ValidationResult(
                is_valid=False,
                error_message=f"Origin not allowed: {origin}",
                security_level=SecurityLevel.HIGH
            )
        
        return ValidationResult(is_valid=True)


# Utility functions for common validation tasks

def validate_api_key(api_key: str, expected_key: str) -> ValidationResult:
    """
    Validate an API key using secure comparison.
    
    Args:
        api_key: Provided API key
        expected_key: Expected API key
        
    Returns:
        ValidationResult
    """
    if not api_key:
        return ValidationResult(
            is_valid=False,
            error_message="API key required",
            security_level=SecurityLevel.HIGH
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, expected_key):
        return ValidationResult(
            is_valid=False,
            error_message="Invalid API key",
            security_level=SecurityLevel.HIGH
        )
    
    return ValidationResult(is_valid=True)


def validate_session_token(token: str, secret: str, max_age_seconds: int = 86400) -> ValidationResult:
    """
    Validate a session token with expiration.
    
    Args:
        token: Session token to validate
        secret: Secret key for HMAC
        max_age_seconds: Maximum token age in seconds
        
    Returns:
        ValidationResult
    """
    if not token:
        return ValidationResult(
            is_valid=False,
            error_message="Session token required",
            security_level=SecurityLevel.MEDIUM
        )
    
    try:
        # Parse token format: timestamp.hmac
        parts = token.split('.')
        if len(parts) != 2:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid token format",
                security_level=SecurityLevel.MEDIUM
            )
        
        timestamp_str, provided_hmac = parts
        timestamp = int(timestamp_str)
        
        # Check token age
        current_time = int(time.time())
        if current_time - timestamp > max_age_seconds:
            return ValidationResult(
                is_valid=False,
                error_message="Session token expired",
                security_level=SecurityLevel.MEDIUM
            )
        
        # Verify HMAC
        expected_hmac = hmac.new(
            secret.encode(),
            timestamp_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(provided_hmac, expected_hmac):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid session token",
                security_level=SecurityLevel.HIGH
            )
        
        return ValidationResult(
            is_valid=True,
            metadata={"timestamp": timestamp, "age_seconds": current_time - timestamp}
        )
        
    except (ValueError, TypeError) as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Token validation error: {str(e)}",
            security_level=SecurityLevel.MEDIUM
        )


def check_rate_limits(
    identifier: str,
    request_counts: Dict[str, List[float]],
    window_seconds: int,
    max_requests: int
) -> ValidationResult:
    """
    Check rate limits using in-memory tracking.
    
    Args:
        identifier: Unique identifier for rate limiting
        request_counts: Dictionary tracking request timestamps
        window_seconds: Time window for rate limiting
        max_requests: Maximum requests allowed in window
        
    Returns:
        ValidationResult
    """
    current_time = time.time()
    
    # Get existing request timestamps for this identifier
    if identifier not in request_counts:
        request_counts[identifier] = []
    
    timestamps = request_counts[identifier]
    
    # Remove old timestamps outside the window
    cutoff_time = current_time - window_seconds
    timestamps[:] = [ts for ts in timestamps if ts > cutoff_time]
    
    # Check if rate limit would be exceeded
    if len(timestamps) >= max_requests:
        return ValidationResult(
            is_valid=False,
            error_message=f"Rate limit exceeded: {len(timestamps)}/{max_requests} requests in {window_seconds}s",
            security_level=SecurityLevel.MEDIUM,
            metadata={
                "current_requests": len(timestamps),
                "max_requests": max_requests,
                "window_seconds": window_seconds,
                "reset_time": timestamps[0] + window_seconds if timestamps else current_time
            }
        )
    
    # Add current request timestamp
    timestamps.append(current_time)
    
    return ValidationResult(
        is_valid=True,
        metadata={
            "current_requests": len(timestamps),
            "max_requests": max_requests,
            "remaining_requests": max_requests - len(timestamps)
        }
    )


def generate_secure_token(user_id: str, secret: str) -> str:
    """
    Generate a secure session token.
    
    Args:
        user_id: User identifier
        secret: Secret key for HMAC
        
    Returns:
        Secure session token
    """
    timestamp = str(int(time.time()))
    data = f"{user_id}.{timestamp}"
    
    signature = hmac.new(
        secret.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{timestamp}.{signature}"


def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """
    Hash a password with salt using secure methods.
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Dictionary with hash and salt
    """
    import secrets
    
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Use PBKDF2 with SHA-256
    import hashlib
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 100,000 iterations
    ).hex()
    
    return {
        "hash": password_hash,
        "salt": salt
    }


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """
    Verify a password against a stored hash.
    
    Args:
        password: Password to verify
        stored_hash: Stored password hash
        salt: Salt used for hashing
        
    Returns:
        True if password matches
    """
    computed_hash = hash_password(password, salt)["hash"]
    return hmac.compare_digest(computed_hash, stored_hash)


def validate_ip_address(ip_address: str, allowed_ips: Optional[List[str]] = None) -> ValidationResult:
    """
    Validate IP address against allowlist.
    
    Args:
        ip_address: IP address to validate
        allowed_ips: List of allowed IP addresses
        
    Returns:
        ValidationResult
    """
    import ipaddress
    
    try:
        # Validate IP format
        ipaddress.ip_address(ip_address)
    except ValueError:
        return ValidationResult(
            is_valid=False,
            error_message="Invalid IP address format",
            security_level=SecurityLevel.MEDIUM
        )
    
    # Check against allowlist if provided
    if allowed_ips and ip_address not in allowed_ips:
        return ValidationResult(
            is_valid=False,
            error_message=f"IP address not allowed: {ip_address}",
            security_level=SecurityLevel.HIGH
        )
    
    return ValidationResult(is_valid=True)


# Security monitoring utilities

def detect_suspicious_patterns(user_activity: Dict[str, Any]) -> List[str]:
    """
    Detect suspicious patterns in user activity.
    
    Args:
        user_activity: Dictionary of user activity data
        
    Returns:
        List of detected suspicious patterns
    """
    suspicious_patterns = []
    
    # High request frequency
    request_rate = user_activity.get("requests_per_minute", 0)
    if request_rate > 100:
        suspicious_patterns.append(f"High request rate: {request_rate}/min")
    
    # Multiple failed authentications
    failed_auth_count = user_activity.get("failed_authentications", 0)
    if failed_auth_count > 5:
        suspicious_patterns.append(f"Multiple failed authentications: {failed_auth_count}")
    
    # Unusual access times
    access_hours = user_activity.get("access_hours", [])
    unusual_hours = [hour for hour in access_hours if hour < 6 or hour > 22]
    if len(unusual_hours) > 3:
        suspicious_patterns.append(f"Unusual access times: {unusual_hours}")
    
    # Geographic anomalies
    countries = user_activity.get("countries", [])
    if len(countries) > 3:
        suspicious_patterns.append(f"Multiple countries: {countries}")
    
    return suspicious_patterns


def calculate_risk_score(security_events: List[Dict[str, Any]]) -> float:
    """
    Calculate a risk score based on security events.
    
    Args:
        security_events: List of security events
        
    Returns:
        Risk score between 0.0 and 1.0
    """
    if not security_events:
        return 0.0
    
    risk_weights = {
        "failed_authentication": 0.2,
        "rate_limit_exceeded": 0.1,
        "suspicious_ip": 0.3,
        "injection_attempt": 0.5,
        "unauthorized_access": 0.4
    }
    
    total_risk = 0.0
    for event in security_events:
        event_type = event.get("type", "unknown")
        weight = risk_weights.get(event_type, 0.1)
        total_risk += weight
    
    # Normalize to 0-1 range
    return min(total_risk / len(security_events), 1.0)