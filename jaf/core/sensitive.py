"""
Sensitive content detection and handling for JAF tools.

This module provides functionality to automatically detect sensitive content
in tool inputs and outputs using LLM Guard scanners, and ensures proper
redaction during tracing.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Protocol
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional llm-guard imports
try:
    from llm_guard import input_scanners, output_scanners
    from llm_guard.input_scanners.base import Scanner as InputScanner
    from llm_guard.output_scanners.base import Scanner as OutputScanner
    LLM_GUARD_AVAILABLE = True
    logger.info("LLM Guard available for sensitive content detection")
except ImportError:
    LLM_GUARD_AVAILABLE = False
    logger.debug("LLM Guard not available - sensitive content detection will use basic heuristics")
    InputScanner = object
    OutputScanner = object


@dataclass
class SensitiveContentConfig:
    """Configuration for sensitive content detection."""
    
    # Enable automatic sensitivity detection
    auto_detect_sensitive: bool = True
    
    # LLM Guard scanner configurations
    enable_secrets_detection: bool = True
    enable_pii_detection: bool = False  # Disabled by default due to model requirements
    enable_code_detection: bool = False  # Disabled by default due to model requirements
    
    # Custom sensitivity patterns (regex)
    custom_patterns: List[str] = None
    
    # Sensitivity score threshold (0.0 - 1.0)
    sensitivity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.custom_patterns is None:
            self.custom_patterns = []


class SensitiveContentDetector:
    """Detects sensitive content in tool inputs and outputs using LLM Guard."""
    
    def __init__(self, config: Optional[SensitiveContentConfig] = None):
        self.config = config or SensitiveContentConfig()
        self._input_scanners: List[InputScanner] = []
        self._output_scanners: List[OutputScanner] = []
        
        if LLM_GUARD_AVAILABLE:
            self._initialize_scanners()
        else:
            logger.warning("LLM Guard not available - using fallback heuristic detection")
    
    def _initialize_scanners(self) -> None:
        """Initialize LLM Guard scanners based on configuration."""
        if not LLM_GUARD_AVAILABLE:
            return
            
        try:
            # Input scanners - only use simple ones that don't require model downloads
            if self.config.enable_secrets_detection:
                self._input_scanners.append(input_scanners.Secrets())
                
            # Skip complex scanners that require internet/models:
            # - Anonymize scanner (requires vault and models)  
            # - Code scanner (requires model downloads)
            
            # Output scanners - skip complex ones
            # - Sensitive scanner (requires model downloads)
            # - Code scanner (requires model downloads)
                
            logger.info(f"Initialized {len(self._input_scanners)} input scanners and {len(self._output_scanners)} output scanners")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Guard scanners: {e}")
            # Fall back to heuristic detection
    
    def detect_sensitive_input(self, content: str, tool_name: str = "") -> Dict[str, Any]:
        """
        Detect sensitive content in tool input.
        
        Args:
            content: The input content to scan
            tool_name: Name of the tool (for context)
            
        Returns:
            Dict with detection results including 'is_sensitive' boolean and 'details'
        """
        if not content or not isinstance(content, str):
            return {"is_sensitive": False, "details": [], "score": 0.0}
            
        result = {
            "is_sensitive": False,
            "details": [],
            "score": 0.0,
            "redacted_content": content
        }
        
        # Heuristic detection for tool names suggesting sensitivity
        if self._is_tool_name_sensitive(tool_name):
            result["is_sensitive"] = True
            result["details"].append({"type": "tool_name", "reason": f"Tool '{tool_name}' suggests sensitive operations"})
            result["score"] = 0.8
        
        # LLM Guard detection
        if LLM_GUARD_AVAILABLE and self._input_scanners:
            try:
                sanitized_content = content
                detection_details = []
                max_score = 0.0
                
                for scanner in self._input_scanners:
                    # Use the correct API: scanner.scan(prompt) returns (sanitized, is_valid, risk_score)
                    sanitized_content, is_valid, risk_score = scanner.scan(content)
                    if not is_valid:
                        detection_details.append({
                            "type": scanner.__class__.__name__,
                            "reason": f"Detected by {scanner.__class__.__name__}",
                            "risk_score": risk_score
                        })
                        max_score = max(max_score, risk_score)
                
                if detection_details:
                    result["is_sensitive"] = max_score >= self.config.sensitivity_threshold
                    result["details"].extend(detection_details)
                    result["score"] = max(result["score"], max_score)
                    result["redacted_content"] = sanitized_content
                    
            except Exception as e:
                logger.warning(f"LLM Guard input scanning failed: {e}")
        
        # Fallback heuristic detection
        if not LLM_GUARD_AVAILABLE or not result["details"]:
            heuristic_result = self._heuristic_detection(content)
            if heuristic_result["is_sensitive"]:
                result["is_sensitive"] = True
                result["details"].extend(heuristic_result["details"])
                result["score"] = max(result["score"], heuristic_result["score"])
        
        return result
    
    def detect_sensitive_output(self, content: str, tool_name: str = "") -> Dict[str, Any]:
        """
        Detect sensitive content in tool output.
        
        Args:
            content: The output content to scan
            tool_name: Name of the tool (for context)
            
        Returns:
            Dict with detection results including 'is_sensitive' boolean and 'details'
        """
        if not content or not isinstance(content, str):
            return {"is_sensitive": False, "details": [], "score": 0.0}
            
        result = {
            "is_sensitive": False,
            "details": [],
            "score": 0.0,
            "redacted_content": content
        }
        
        # Heuristic detection for tool names suggesting sensitivity
        if self._is_tool_name_sensitive(tool_name):
            result["is_sensitive"] = True
            result["details"].append({"type": "tool_name", "reason": f"Tool '{tool_name}' suggests sensitive operations"})
            result["score"] = 0.8
        
        # LLM Guard detection
        if LLM_GUARD_AVAILABLE and self._output_scanners:
            try:
                sanitized_content = content
                detection_details = []
                max_score = 0.0
                
                for scanner in self._output_scanners:
                    # Use the correct API: scanner.scan(prompt) returns (sanitized, is_valid, risk_score)
                    sanitized_content, is_valid, risk_score = scanner.scan(content)
                    if not is_valid:
                        detection_details.append({
                            "type": scanner.__class__.__name__,
                            "reason": f"Detected by {scanner.__class__.__name__}",
                            "risk_score": risk_score
                        })
                        max_score = max(max_score, risk_score)
                
                if detection_details:
                    result["is_sensitive"] = max_score >= self.config.sensitivity_threshold
                    result["details"].extend(detection_details)
                    result["score"] = max(result["score"], max_score)
                    result["redacted_content"] = sanitized_content
                    
            except Exception as e:
                logger.warning(f"LLM Guard output scanning failed: {e}")
        
        # Fallback heuristic detection
        if not LLM_GUARD_AVAILABLE or not result["details"]:
            heuristic_result = self._heuristic_detection(content)
            if heuristic_result["is_sensitive"]:
                result["is_sensitive"] = True
                result["details"].extend(heuristic_result["details"])
                result["score"] = max(result["score"], heuristic_result["score"])
        
        return result
    
    def _is_tool_name_sensitive(self, tool_name: str) -> bool:
        """Check if tool name suggests sensitive operations."""
        if not tool_name:
            return False
            
        sensitive_keywords = [
            "secret", "password", "token", "key", "credential", "auth", "login",
            "pii", "personal", "private", "confidential", "sensitive", "secure",
            "decrypt", "encrypt", "wallet", "payment", "card", "ssn", "social",
            "medical", "health", "financial", "bank", "account", "balance"
        ]
        
        tool_name_lower = tool_name.lower()
        return any(keyword in tool_name_lower for keyword in sensitive_keywords)
    
    def _heuristic_detection(self, content: str) -> Dict[str, Any]:
        """Basic heuristic detection for sensitive content patterns."""
        import re
        
        patterns = [
            # Credit card numbers
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "credit_card"),
            # Social Security Numbers
            (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', "ssn"),
            # Email addresses (can contain PII)
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
            # API keys/tokens (improved pattern to handle common formats)
            (r'\bsk-[A-Za-z0-9]+\b', "api_key"),  # OpenAI style keys
            (r'\b[A-Za-z0-9]{32,}\b', "api_key"),  # Generic long alphanumeric keys
            (r'\bAKIA[A-Z0-9]{16}\b', "api_key"),  # AWS access keys
            # Passwords in text
            (r'\bpassword[:\s]*[A-Za-z0-9!@#$%^&*]+', "password"),
            # Phone numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "phone"),
        ]
        
        # Add custom patterns from config
        for pattern in self.config.custom_patterns:
            patterns.append((pattern, "custom"))
        
        details = []
        for pattern, type_name in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                details.append({
                    "type": type_name,
                    "reason": f"Found {len(matches)} matches for {type_name}",
                    "count": len(matches)
                })
        
        return {
            "is_sensitive": bool(details),
            "details": details,
            "score": 0.8 if details else 0.0
        }


# Global detector instance
_detector: Optional[SensitiveContentDetector] = None


def get_sensitive_detector(config: Optional[SensitiveContentConfig] = None) -> SensitiveContentDetector:
    """Get or create the global sensitive content detector."""
    global _detector
    if _detector is None or config is not None:
        _detector = SensitiveContentDetector(config)
    return _detector


def is_content_sensitive(content: str, tool_name: str = "", is_input: bool = True) -> bool:
    """
    Quick check if content is sensitive.
    
    Args:
        content: Content to check
        tool_name: Name of the tool (for context)
        is_input: True if checking input, False if checking output
        
    Returns:
        Boolean indicating if content is sensitive
    """
    detector = get_sensitive_detector()
    if is_input:
        result = detector.detect_sensitive_input(content, tool_name)
    else:
        result = detector.detect_sensitive_output(content, tool_name)
    return result["is_sensitive"]