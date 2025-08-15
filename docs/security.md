# Security Guide

Comprehensive security guidelines and best practices for JAF applications in production environments.

## Overview

Security is paramount when deploying AI agents that process user data and interact with external systems. This guide covers authentication, authorization, input validation, secure communication, and threat mitigation strategies.

## Authentication and Authorization

### JWT Authentication

Implement secure JWT-based authentication for your JAF applications:

```python
import jwt
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class JWTAuthenticator:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_token(self, user_id: str, permissions: list, expires_in: int = 3600) -> str:
        """Create a JWT token for a user."""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
            "iss": "jaf-system"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """FastAPI dependency to get current authenticated user."""
        token = credentials.credentials
        payload = self.verify_token(token)
        return {
            "user_id": payload["user_id"],
            "permissions": payload["permissions"]
        }

# Usage in your JAF application
authenticator = JWTAuthenticator(secret_key=os.getenv("JWT_SECRET_KEY"))

@app.post("/chat")
async def chat_endpoint(
    message: str,
    current_user: dict = Depends(authenticator.get_current_user)
):
    # User is authenticated, process the chat message
    context = create_context(
        user_id=current_user["user_id"],
        permissions=current_user["permissions"]
    )
    
    return await process_agent_message(message, context)
```

### API Key Authentication

For service-to-service communication, implement API key authentication:

```python
import secrets
import hashlib
import hmac
from typing import Dict, Optional

class APIKeyManager:
    def __init__(self):
        self.api_keys: Dict[str, dict] = {}
    
    def generate_api_key(self, client_name: str, permissions: list) -> str:
        """Generate a new API key for a client."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "client_name": client_name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key and return client info."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            key_info = self.api_keys[key_hash]
            key_info["last_used"] = datetime.utcnow()
            key_info["usage_count"] += 1
            return key_info
        
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return self.api_keys.pop(key_hash, None) is not None

# FastAPI dependency for API key authentication
async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    key_info = api_key_manager.validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return key_info

@app.post("/api/agents/process")
async def api_process(
    message: str,
    client_info: dict = Depends(verify_api_key)
):
    # Client authenticated via API key
    context = create_api_context(
        client_name=client_info["client_name"],
        permissions=client_info["permissions"]
    )
    
    return await process_agent_message(message, context)
```

### Role-Based Access Control (RBAC)

Implement fine-grained permissions:

```python
from enum import Enum
from typing import Set, List

class Permission(Enum):
    CHAT_BASIC = "chat:basic"
    CHAT_ADVANCED = "chat:advanced"
    AGENT_MANAGE = "agent:manage"
    SYSTEM_ADMIN = "system:admin"
    DATA_EXPORT = "data:export"
    ANALYTICS_VIEW = "analytics:view"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

# Define roles
ROLES = {
    "user": Role("user", {Permission.CHAT_BASIC}),
    "premium_user": Role("premium_user", {Permission.CHAT_BASIC, Permission.CHAT_ADVANCED}),
    "admin": Role("admin", {Permission.CHAT_BASIC, Permission.CHAT_ADVANCED, Permission.AGENT_MANAGE, Permission.ANALYTICS_VIEW}),
    "system_admin": Role("system_admin", {Permission.CHAT_BASIC, Permission.CHAT_ADVANCED, Permission.AGENT_MANAGE, Permission.SYSTEM_ADMIN, Permission.DATA_EXPORT, Permission.ANALYTICS_VIEW})
}

def check_permission(user_permissions: List[str], required_permission: Permission) -> bool:
    """Check if user has required permission."""
    return required_permission.value in user_permissions

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from context (implementation depends on your auth setup)
            user = kwargs.get('current_user') or kwargs.get('context', {}).get('user')
            if not user or not check_permission(user.get('permissions', []), permission):
                raise HTTPException(status_code=403, detail=f"Permission {permission.value} required")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_permission(Permission.AGENT_MANAGE)
async def manage_agent(agent_config: dict, current_user: dict):
    # Only users with agent:manage permission can access this
    pass
```

## Input Validation and Sanitization

### Secure Input Handling

Always validate and sanitize user inputs to prevent injection attacks:

```python
import re
import html
import bleach
from typing import Any, Dict, List
from pydantic import BaseModel, validator, Field

class SecureMessageInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    message_type: str = Field(..., regex=r'^(text|image|document)$')
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def sanitize_content(cls, v):
        # Remove potentially dangerous HTML/script content
        allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
        sanitized = bleach.clean(v, tags=allowed_tags, strip=True)
        
        # Additional sanitization for SQL injection prevention
        dangerous_patterns = [
            r'(union|select|insert|update|delete|drop|create|alter)\s+',
            r'(script|javascript|vbscript|onload|onerror)',
            r'(<|%3C)(script|iframe|object|embed)'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError("Content contains potentially dangerous patterns")
        
        return sanitized
    
    @validator('metadata')
    def validate_metadata(cls, v):
        # Limit metadata size and validate structure
        if len(str(v)) > 1000:
            raise ValueError("Metadata too large")
        
        # Ensure no executable content in metadata
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if isinstance(k, str) and len(k) < 100}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d[:10]]  # Limit list size
            elif isinstance(d, str):
                return html.escape(d[:500])  # Limit string length and escape
            elif isinstance(d, (int, float, bool)):
                return d
            else:
                return str(d)[:500]
        
        return clean_dict(v)

class InputSanitizer:
    def __init__(self):
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+[\"'].*[\"']\s*=\s*[\"'].*[\"'])"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"expression\s*\(",
            r"@import",
            r"<!--.*?-->"
        ]
    
    def sanitize_sql_input(self, input_str: str) -> str:
        """Sanitize input to prevent SQL injection."""
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                raise ValueError("Input contains potential SQL injection patterns")
        
        # Additional escaping
        escaped = input_str.replace("'", "''").replace('"', '""')
        return escaped
    
    def sanitize_xss_input(self, input_str: str) -> str:
        """Sanitize input to prevent XSS attacks."""
        sanitized = html.escape(input_str)
        
        for pattern in self.xss_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError("Input contains potential XSS patterns")
        
        return sanitized
    
    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file paths to prevent directory traversal."""
        # Remove any path traversal attempts
        dangerous_patterns = ["..", "/", "\\", "~"]
        
        for pattern in dangerous_patterns:
            if pattern in file_path:
                raise ValueError("File path contains dangerous characters")
        
        # Only allow alphanumeric, dash, underscore, and dot
        if not re.match(r'^[a-zA-Z0-9._-]+$', file_path):
            raise ValueError("File path contains invalid characters")
        
        return file_path

# Usage in agent tools
sanitizer = InputSanitizer()

class DatabaseQueryTool:
    async def execute(self, query_input: str, context):
        # Sanitize the input before using in database queries
        try:
            sanitized_input = sanitizer.sanitize_sql_input(query_input)
            # Use parameterized queries, never string concatenation
            result = await database.execute(
                "SELECT * FROM table WHERE column = $1",
                sanitized_input
            )
            return result
        except ValueError as e:
            return f"Security error: {e}"
```

### Content Filtering

Implement content filtering to prevent inappropriate or harmful outputs:

```python
import openai
from typing import List, Dict, Any

class ContentFilter:
    def __init__(self):
        self.blocked_topics = [
            "violence", "hate_speech", "illegal_activities", 
            "personal_information", "harmful_instructions"
        ]
        
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b'  # Phone number pattern
        ]
    
    async def filter_input(self, content: str) -> Dict[str, Any]:
        """Filter user input for inappropriate content."""
        issues = []
        
        # Check for sensitive information
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content):
                issues.append("Contains potential sensitive information")
                break
        
        # Check for inappropriate content using OpenAI Moderation API
        try:
            response = await openai.Moderation.acreate(input=content)
            if response.results[0].flagged:
                flagged_categories = [
                    category for category, flagged in response.results[0].categories.items()
                    if flagged
                ]
                issues.extend(flagged_categories)
        except Exception as e:
            # Log the error but don't block the request
            logger.warning("Content moderation check failed", error=str(e))
        
        return {
            "allowed": len(issues) == 0,
            "issues": issues,
            "filtered_content": self._redact_sensitive_info(content) if issues else content
        }
    
    def _redact_sensitive_info(self, content: str) -> str:
        """Redact sensitive information from content."""
        redacted = content
        
        # Redact SSN
        redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', redacted)
        
        # Redact credit card numbers
        redacted = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', redacted)
        
        # Redact email addresses
        redacted = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', redacted)
        
        # Redact phone numbers
        redacted = re.sub(r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b', 'XXX-XXX-XXXX', redacted)
        
        return redacted

# Usage in agent processing
content_filter = ContentFilter()

async def process_user_message(message: str, context):
    # Filter input content
    filter_result = await content_filter.filter_input(message)
    
    if not filter_result["allowed"]:
        return {
            "error": "Content not allowed",
            "issues": filter_result["issues"]
        }
    
    # Process the message with the agent
    return await agent.process(filter_result["filtered_content"], context)
```

## Secure Communication

### TLS/SSL Configuration

Ensure all communications are encrypted:

```python
import ssl
import asyncio
from pathlib import Path

def create_ssl_context(cert_path: str, key_path: str, ca_path: str = None) -> ssl.SSLContext:
    """Create a secure SSL context."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    
    # Load certificate and private key
    context.load_cert_chain(cert_path, key_path)
    
    # Configure for security
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.check_hostname = False  # Set to True in production with proper certificates
    context.verify_mode = ssl.CERT_REQUIRED if ca_path else ssl.CERT_NONE
    
    if ca_path:
        context.load_verify_locations(ca_path)
    
    return context

# Usage with FastAPI/Uvicorn
ssl_context = create_ssl_context(
    cert_path="/path/to/cert.pem",
    key_path="/path/to/key.pem"
)

# Start server with TLS
uvicorn.run(
    app,
    host="0.0.0.0",
    port=443,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem",
    ssl_version=ssl.PROTOCOL_TLS,
    ssl_ciphers="TLSv1.2"
)
```

### Request Signing

Implement request signing for API security:

```python
import hmac
import hashlib
import base64
from datetime import datetime, timedelta

class RequestSigner:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def sign_request(self, method: str, uri: str, body: str, timestamp: str) -> str:
        """Sign a request with HMAC-SHA256."""
        message = f"{method}\n{uri}\n{body}\n{timestamp}"
        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def verify_request(self, method: str, uri: str, body: str, timestamp: str, signature: str) -> bool:
        """Verify a signed request."""
        # Check timestamp to prevent replay attacks
        try:
            request_time = datetime.fromisoformat(timestamp)
            if abs((datetime.utcnow() - request_time).total_seconds()) > 300:  # 5 minutes
                return False
        except ValueError:
            return False
        
        expected_signature = self.sign_request(method, uri, body, timestamp)
        return hmac.compare_digest(signature, expected_signature)

# Middleware for request verification
@app.middleware("http")
async def verify_request_signature(request: Request, call_next):
    if request.url.path.startswith("/api/secure/"):
        signature = request.headers.get("X-Signature")
        timestamp = request.headers.get("X-Timestamp")
        
        if not signature or not timestamp:
            return Response("Missing signature headers", status_code=401)
        
        body = await request.body()
        
        signer = RequestSigner(os.getenv("API_SECRET_KEY"))
        if not signer.verify_request(
            request.method,
            str(request.url),
            body.decode(),
            timestamp,
            signature
        ):
            return Response("Invalid signature", status_code=401)
    
    return await call_next(request)
```

## Data Protection

### Data Encryption

Encrypt sensitive data at rest and in transit:

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

class DataEncryption:
    def __init__(self, password: str, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)
        self.salt = salt
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt string data."""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data back to string."""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def encrypt_dict(self, data: dict) -> bytes:
        """Encrypt dictionary data."""
        import json
        json_str = json.dumps(data, sort_keys=True)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: bytes) -> dict:
        """Decrypt data back to dictionary."""
        import json
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)

# Usage for storing sensitive conversation data
encryption = DataEncryption(os.getenv("ENCRYPTION_PASSWORD"))

class SecureMemoryProvider:
    def __init__(self, base_provider, encryption_key):
        self.base_provider = base_provider
        self.encryption = DataEncryption(encryption_key)
    
    async def store_conversation(self, conversation_id: str, messages: list):
        # Encrypt sensitive message content
        encrypted_messages = []
        for message in messages:
            if message.get('role') == 'user':
                # Encrypt user messages
                encrypted_content = self.encryption.encrypt(message['content'])
                encrypted_message = {
                    **message,
                    'content': base64.b64encode(encrypted_content).decode(),
                    'encrypted': True
                }
                encrypted_messages.append(encrypted_message)
            else:
                encrypted_messages.append(message)
        
        await self.base_provider.store_conversation(conversation_id, encrypted_messages)
    
    async def get_conversation(self, conversation_id: str):
        messages = await self.base_provider.get_conversation(conversation_id)
        
        # Decrypt user messages
        decrypted_messages = []
        for message in messages:
            if message.get('encrypted'):
                encrypted_content = base64.b64decode(message['content'])
                decrypted_content = self.encryption.decrypt(encrypted_content)
                decrypted_message = {
                    **message,
                    'content': decrypted_content,
                    'encrypted': False
                }
                decrypted_messages.append(decrypted_message)
            else:
                decrypted_messages.append(message)
        
        return decrypted_messages
```

### Personal Data Handling

Implement GDPR-compliant data handling:

```python
from typing import Dict, List, Any
from datetime import datetime, timedelta
import hashlib

class PersonalDataManager:
    def __init__(self):
        self.data_retention_days = 365
        self.anonymization_fields = ['email', 'phone', 'ssn', 'credit_card']
    
    def anonymize_personal_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data in a dictionary."""
        anonymized = data.copy()
        
        for field in self.anonymization_fields:
            if field in anonymized:
                # Hash the value instead of storing plaintext
                hashed_value = hashlib.sha256(
                    str(anonymized[field]).encode()
                ).hexdigest()[:16]
                anonymized[field] = f"anonymized_{hashed_value}"
        
        return anonymized
    
    def should_delete_data(self, created_at: datetime) -> bool:
        """Check if data should be deleted based on retention policy."""
        retention_deadline = datetime.utcnow() - timedelta(days=self.data_retention_days)
        return created_at < retention_deadline
    
    async def process_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """Process a user's data deletion request (Right to be Forgotten)."""
        deletion_report = {
            "user_id": user_id,
            "requested_at": datetime.utcnow(),
            "deleted_items": []
        }
        
        # Delete from conversations
        conversations_deleted = await self._delete_user_conversations(user_id)
        deletion_report["deleted_items"].append({
            "type": "conversations",
            "count": conversations_deleted
        })
        
        # Delete from user profiles
        profile_deleted = await self._delete_user_profile(user_id)
        deletion_report["deleted_items"].append({
            "type": "profile",
            "deleted": profile_deleted
        })
        
        # Delete from analytics (anonymize)
        analytics_anonymized = await self._anonymize_user_analytics(user_id)
        deletion_report["deleted_items"].append({
            "type": "analytics",
            "anonymized": analytics_anonymized
        })
        
        return deletion_report
    
    async def _delete_user_conversations(self, user_id: str) -> int:
        """Delete all conversations for a user."""
        # Implementation depends on your data storage
        # This is a placeholder
        return 0
    
    async def _delete_user_profile(self, user_id: str) -> bool:
        """Delete user profile data."""
        # Implementation depends on your data storage
        return True
    
    async def _anonymize_user_analytics(self, user_id: str) -> int:
        """Anonymize user data in analytics instead of deleting."""
        # Replace user_id with hashed version in analytics
        anonymous_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        # Update analytics records
        return 0

# GDPR compliance middleware
class GDPRMiddleware:
    def __init__(self):
        self.data_manager = PersonalDataManager()
    
    async def __call__(self, request: Request, call_next):
        # Add privacy headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
```

## Rate Limiting and DDoS Protection

### Advanced Rate Limiting

Implement sophisticated rate limiting:

```python
import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        now = time.time()
        
        # Refill tokens
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class RateLimiter:
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def get_rate_limit_key(self, request) -> str:
        """Generate rate limit key based on user/IP."""
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        return f"ip:{request.client.host}"
    
    def check_rate_limit(self, key: str, limit_type: str = "default") -> tuple[bool, dict]:
        """Check if request is within rate limits."""
        
        # Different limits for different types
        limits = {
            "default": (100, 60),      # 100 requests per minute
            "chat": (20, 60),          # 20 chat messages per minute
            "api": (1000, 3600),       # 1000 API calls per hour
            "upload": (5, 300),        # 5 uploads per 5 minutes
        }
        
        if limit_type not in limits:
            limit_type = "default"
        
        requests_per_period, period_seconds = limits[limit_type]
        
        # Get or create token bucket
        bucket_key = f"{key}:{limit_type}"
        if bucket_key not in self.buckets:
            refill_rate = requests_per_period / period_seconds
            self.buckets[bucket_key] = TokenBucket(requests_per_period, refill_rate)
        
        bucket = self.buckets[bucket_key]
        allowed = bucket.consume()
        
        # Track request history
        now = time.time()
        history = self.request_history[bucket_key]
        history.append(now)
        
        # Calculate current usage
        recent_requests = sum(1 for req_time in history if now - req_time < period_seconds)
        
        rate_limit_info = {
            "allowed": allowed,
            "limit": requests_per_period,
            "remaining": max(0, int(bucket.tokens)),
            "reset_time": int(now + period_seconds),
            "current_usage": recent_requests
        }
        
        return allowed, rate_limit_info

# Rate limiting middleware
rate_limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Determine rate limit type based on endpoint
    limit_type = "default"
    if request.url.path.startswith("/chat"):
        limit_type = "chat"
    elif request.url.path.startswith("/api"):
        limit_type = "api"
    elif request.url.path.startswith("/upload"):
        limit_type = "upload"
    
    # Check rate limit
    key = rate_limiter.get_rate_limit_key(request)
    allowed, rate_info = rate_limiter.check_rate_limit(key, limit_type)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": rate_info["reset_time"]},
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_time"]),
                "Retry-After": str(rate_info["reset_time"] - int(time.time()))
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
    
    return response
```

### DDoS Protection

Implement DDoS protection mechanisms:

```python
import asyncio
from collections import Counter, defaultdict
import ipaddress

class DDoSProtection:
    def __init__(self):
        self.request_counts = defaultdict(Counter)
        self.blocked_ips = set()
        self.suspicious_patterns = []
        self.cleanup_interval = 300  # 5 minutes
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
    
    def is_suspicious_request(self, request) -> tuple[bool, str]:
        """Detect suspicious request patterns."""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return True, "Blocked IP"
        
        # Check for suspicious user agents
        suspicious_agents = [
            "bot", "crawler", "spider", "scraper", 
            "scanner", "curl", "wget", "python-requests"
        ]
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            return True, "Suspicious user agent"
        
        # Check request frequency
        current_minute = int(time.time() // 60)
        minute_requests = self.request_counts[client_ip][current_minute]
        
        if minute_requests > 100:  # More than 100 requests per minute
            return True, "High request frequency"
        
        # Check for rapid successive requests
        if self._check_rapid_requests(client_ip):
            return True, "Rapid successive requests"
        
        return False, ""
    
    def _check_rapid_requests(self, client_ip: str) -> bool:
        """Check for rapid successive requests from same IP."""
        now = time.time()
        recent_requests = [
            req_time for req_time in self.request_history.get(client_ip, [])
            if now - req_time < 1  # Requests in last second
        ]
        return len(recent_requests) > 10  # More than 10 requests per second
    
    def block_ip(self, client_ip: str, duration: int = 3600):
        """Block an IP address for specified duration."""
        self.blocked_ips.add(client_ip)
        
        # Schedule unblocking
        async def unblock_later():
            await asyncio.sleep(duration)
            self.blocked_ips.discard(client_ip)
        
        asyncio.create_task(unblock_later())
    
    async def _cleanup_task(self):
        """Periodically clean up old request data."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            
            current_time = int(time.time() // 60)
            cutoff_time = current_time - 60  # Keep last hour
            
            for ip in list(self.request_counts.keys()):
                # Remove old entries
                self.request_counts[ip] = Counter({
                    minute: count for minute, count in self.request_counts[ip].items()
                    if minute > cutoff_time
                })
                
                # Remove empty counters
                if not self.request_counts[ip]:
                    del self.request_counts[ip]

# DDoS protection middleware
ddos_protection = DDoSProtection()

@app.middleware("http")
async def ddos_protection_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    # Check for suspicious activity
    is_suspicious, reason = ddos_protection.is_suspicious_request(request)
    
    if is_suspicious:
        # Log the suspicious activity
        logger.warning(
            "Suspicious request blocked",
            client_ip=client_ip,
            reason=reason,
            user_agent=request.headers.get("user-agent"),
            path=request.url.path
        )
        
        # Block the IP for repeat offenses
        if reason in ["High request frequency", "Rapid successive requests"]:
            ddos_protection.block_ip(client_ip, 3600)  # Block for 1 hour
        
        return JSONResponse(
            status_code=429,
            content={"error": "Request blocked", "reason": reason}
        )
    
    # Track the request
    current_minute = int(time.time() // 60)
    ddos_protection.request_counts[client_ip][current_minute] += 1
    
    return await call_next(request)
```

## Security Monitoring

### Security Event Logging

Implement comprehensive security logging:

```python
import json
from datetime import datetime
from enum import Enum

class SecurityEventType(Enum):
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    ADMIN_ACTION = "admin_action"
    SECURITY_VIOLATION = "security_violation"

class SecurityLogger:
    def __init__(self):
        self.logger = structlog.get_logger("security")
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        details: dict = None,
        severity: str = "info"
    ):
        """Log a security event with structured data."""
        
        event_data = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {}
        }
        
        # Add geolocation if available
        if ip_address:
            event_data["location"] = self._get_location(ip_address)
        
        # Log with appropriate level
        if severity == "critical":
            self.logger.critical("Security event", **event_data)
        elif severity == "warning":
            self.logger.warning("Security event", **event_data)
        else:
            self.logger.info("Security event", **event_data)
        
        # Send to SIEM if configured
        if hasattr(self, 'siem_client'):
            asyncio.create_task(self._send_to_siem(event_data))
    
    def _get_location(self, ip_address: str) -> dict:
        """Get geolocation for IP address."""
        # Implementation would use a geolocation service
        return {"country": "Unknown", "city": "Unknown"}
    
    async def _send_to_siem(self, event_data: dict):
        """Send security event to SIEM system."""
        # Implementation would send to your SIEM system
        pass

# Usage throughout the application
security_logger = SecurityLogger()

# Authentication events
@app.post("/auth/login")
async def login(credentials: LoginCredentials, request: Request):
    try:
        user = await authenticate_user(credentials)
        
        security_logger.log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            user_id=user.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            details={"login_method": "password"}
        )
        
        return {"token": create_token(user)}
        
    except AuthenticationError as e:
        security_logger.log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            details={"error": str(e), "username": credentials.username},
            severity="warning"
        )
        
        raise HTTPException(status_code=401, detail="Authentication failed")

# Data access events
async def access_sensitive_data(user_id: str, data_type: str, request: Request):
    security_logger.log_security_event(
        SecurityEventType.DATA_ACCESS,
        user_id=user_id,
        ip_address=request.client.host,
        details={
            "data_type": data_type,
            "access_method": "api"
        }
    )
```

## Security Best Practices

### Secure Configuration

```python
import os
from typing import Dict, Any

class SecurityConfig:
    def __init__(self):
        self.settings = self._load_secure_settings()
        self._validate_settings()
    
    def _load_secure_settings(self) -> Dict[str, Any]:
        """Load security settings from environment variables."""
        return {
            # Authentication
            "jwt_secret": self._get_required_env("JWT_SECRET_KEY"),
            "jwt_expiry": int(os.getenv("JWT_EXPIRY_SECONDS", "3600")),
            
            # Encryption
            "encryption_key": self._get_required_env("ENCRYPTION_KEY"),
            
            # Rate limiting
            "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            "max_requests_per_minute": int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60")),
            
            # CORS
            "cors_origins": os.getenv("CORS_ORIGINS", "").split(","),
            "cors_credentials": os.getenv("CORS_CREDENTIALS", "false").lower() == "true",
            
            # Security headers
            "security_headers_enabled": os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true",
            
            # Content filtering
            "content_filtering_enabled": os.getenv("CONTENT_FILTERING_ENABLED", "true").lower() == "true",
            
            # Logging
            "security_logging_enabled": os.getenv("SECURITY_LOGGING_ENABLED", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _validate_settings(self):
        """Validate security settings."""
        # Validate JWT secret strength
        if len(self.settings["jwt_secret"]) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        
        # Validate encryption key
        if len(self.settings["encryption_key"]) < 32:
            raise ValueError("Encryption key must be at least 32 characters long")
        
        # Validate CORS origins
        if not self.settings["cors_origins"] or self.settings["cors_origins"] == [""]:
            logger.warning("CORS origins not configured - this may be a security risk")

# Initialize security configuration
security_config = SecurityConfig()
```

### Security Checklist

Use this checklist for production deployments:

1. **Authentication & Authorization**
   - [ ] Strong password requirements implemented
   - [ ] JWT tokens have appropriate expiry times
   - [ ] API keys are properly secured and rotated
   - [ ] Role-based access control is implemented
   - [ ] Authentication attempts are logged

2. **Input Validation**
   - [ ] All user inputs are validated and sanitized
   - [ ] SQL injection protection is in place
   - [ ] XSS protection is implemented
   - [ ] File upload restrictions are enforced
   - [ ] Content filtering is enabled

3. **Communication Security**
   - [ ] TLS/SSL is enforced for all communications
   - [ ] Certificate validation is properly configured
   - [ ] Request signing is implemented for sensitive APIs
   - [ ] CORS is properly configured

4. **Data Protection**
   - [ ] Sensitive data is encrypted at rest
   - [ ] Personal data handling complies with regulations
   - [ ] Data retention policies are implemented
   - [ ] Secure data deletion procedures are in place

5. **Infrastructure Security**
   - [ ] Rate limiting is configured
   - [ ] DDoS protection is in place
   - [ ] Security headers are enabled
   - [ ] Regular security updates are applied

6. **Monitoring & Logging**
   - [ ] Security events are logged
   - [ ] Anomaly detection is configured
   - [ ] Incident response procedures are documented
   - [ ] Regular security audits are performed

This comprehensive security guide provides the foundation for deploying secure JAF applications in production environments.