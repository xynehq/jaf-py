"""
Tests for sensitive content detection functionality.
"""

import pytest
from jaf.core.sensitive import (
    SensitiveContentDetector,
    SensitiveContentConfig,
    is_content_sensitive,
    get_sensitive_detector,
)


class TestSensitiveContentDetector:
    """Test the SensitiveContentDetector class."""
    
    def test_default_config(self):
        """Test detector with default configuration."""
        detector = SensitiveContentDetector()
        assert detector.config.auto_detect_sensitive is True
        assert detector.config.enable_secrets_detection is True
        assert detector.config.enable_pii_detection is True
        assert detector.config.sensitivity_threshold == 0.7
    
    def test_custom_config(self):
        """Test detector with custom configuration."""
        config = SensitiveContentConfig(
            auto_detect_sensitive=False,
            enable_secrets_detection=False,
            sensitivity_threshold=0.5
        )
        detector = SensitiveContentDetector(config)
        assert detector.config.auto_detect_sensitive is False
        assert detector.config.enable_secrets_detection is False
        assert detector.config.sensitivity_threshold == 0.5
    
    def test_tool_name_sensitivity(self):
        """Test sensitivity detection based on tool names."""
        detector = SensitiveContentDetector()
        
        # Test sensitive tool names
        sensitive_names = [
            "get_secret_balance",
            "fetch_password",
            "retrieve_api_key", 
            "personal_info",
            "decrypt_data",
            "get_ssn",
        ]
        
        for name in sensitive_names:
            result = detector.detect_sensitive_input("test content", name)
            assert result["is_sensitive"], f"Tool name '{name}' should be detected as sensitive"
        
        # Test non-sensitive tool names
        non_sensitive_names = [
            "get_weather",
            "calculate_sum",
            "public_info",
            "list_files",
        ]
        
        for name in non_sensitive_names:
            result = detector.detect_sensitive_input("test content", name)
            # Should not be sensitive based on tool name alone (unless content triggers it)
            if result["is_sensitive"]:
                # Check if it was detected due to content, not tool name
                assert not any(d["type"] == "tool_name" for d in result["details"])
    
    def test_heuristic_detection(self):
        """Test heuristic pattern-based detection."""
        detector = SensitiveContentDetector()
        
        # Test credit card patterns
        cc_content = "My credit card number is 1234-5678-9012-3456"
        result = detector.detect_sensitive_input(cc_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "credit_card" for d in result["details"])
        
        # Test SSN patterns
        ssn_content = "My social security number is 123-45-6789"
        result = detector.detect_sensitive_input(ssn_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "ssn" for d in result["details"])
        
        # Test email patterns
        email_content = "Contact me at john.doe@example.com"
        result = detector.detect_sensitive_input(email_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "email" for d in result["details"])
        
        # Test API key patterns
        api_content = "Here is the key: sk-1234567890abcdef1234567890abcdef"
        result = detector.detect_sensitive_input(api_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "api_key" for d in result["details"])
        
        # Test password patterns
        pwd_content = "password: mySecretPassword123"
        result = detector.detect_sensitive_input(pwd_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "password" for d in result["details"])
    
    def test_non_sensitive_content(self):
        """Test that normal content is not flagged as sensitive."""
        detector = SensitiveContentDetector()
        
        normal_content = "The weather is sunny today. Temperature is 75 degrees."
        result = detector.detect_sensitive_input(normal_content)
        assert not result["is_sensitive"]
        assert len(result["details"]) == 0
        assert result["score"] == 0.0
    
    def test_custom_patterns(self):
        """Test custom sensitivity patterns."""
        config = SensitiveContentConfig(
            custom_patterns=[
                r'\btop[_-]?secret\b',
                r'\bclassified\b',
            ]
        )
        detector = SensitiveContentDetector(config)
        
        # Test custom patterns
        classified_content = "This document is classified and top-secret"
        result = detector.detect_sensitive_input(classified_content)
        assert result["is_sensitive"]
        assert any(d["type"] == "custom" for d in result["details"])
    
    def test_empty_content(self):
        """Test handling of empty or None content."""
        detector = SensitiveContentDetector()
        
        # Test empty string
        result = detector.detect_sensitive_input("")
        assert not result["is_sensitive"]
        
        # Test None (should be handled gracefully)
        result = detector.detect_sensitive_input(None)
        assert not result["is_sensitive"]
    
    def test_output_detection(self):
        """Test detection of sensitive content in outputs."""
        detector = SensitiveContentDetector()
        
        # Test output with PII
        pii_output = "User info: John Doe, SSN: 123-45-6789, Email: john@example.com"
        result = detector.detect_sensitive_output(pii_output)
        assert result["is_sensitive"]
        
        # Should detect multiple types
        types_found = {d["type"] for d in result["details"]}
        assert "ssn" in types_found
        assert "email" in types_found


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_is_content_sensitive_input(self):
        """Test the is_content_sensitive function for inputs."""
        # Test with sensitive content
        sensitive_content = "API key: sk-1234567890abcdef"
        assert is_content_sensitive(sensitive_content, is_input=True)
        
        # Test with normal content
        normal_content = "Hello world"
        assert not is_content_sensitive(normal_content, is_input=True)
    
    def test_is_content_sensitive_output(self):
        """Test the is_content_sensitive function for outputs."""
        # Test with sensitive output
        sensitive_output = "User password: secretPassword123"
        assert is_content_sensitive(sensitive_output, is_input=False)
        
        # Test with normal output
        normal_output = "Operation completed successfully"
        assert not is_content_sensitive(normal_output, is_input=False)
    
    def test_get_sensitive_detector_singleton(self):
        """Test that get_sensitive_detector returns the same instance."""
        detector1 = get_sensitive_detector()
        detector2 = get_sensitive_detector()
        assert detector1 is detector2
    
    def test_get_sensitive_detector_with_config(self):
        """Test that get_sensitive_detector creates new instance with config."""
        config = SensitiveContentConfig(sensitivity_threshold=0.9)
        detector1 = get_sensitive_detector(config)
        detector2 = get_sensitive_detector()
        
        # Should be the same instance since we set a global one
        assert detector1 is detector2
        assert detector1.config.sensitivity_threshold == 0.9


@pytest.mark.integration
class TestSensitiveIntegration:
    """Integration tests for sensitive content detection."""
    
    @pytest.fixture
    def sample_sensitive_data(self):
        """Provide sample sensitive data for testing."""
        return {
            "user_info": {
                "name": "John Doe",
                "email": "john.doe@example.com", 
                "ssn": "123-45-6789",
                "phone": "555-123-4567",
                "api_key": "sk-1234567890abcdef1234567890abcdef"
            },
            "financial_data": {
                "account": "1234567890",
                "routing": "987654321",
                "card": "4532-1234-5678-9012"
            }
        }
    
    def test_complex_sensitive_detection(self, sample_sensitive_data):
        """Test detection of complex sensitive data structures."""
        import json
        detector = SensitiveContentDetector()
        
        # Convert to JSON string as would happen in real tool execution
        json_data = json.dumps(sample_sensitive_data)
        
        result = detector.detect_sensitive_input(json_data)
        assert result["is_sensitive"]
        
        # Should detect multiple types of sensitive information
        types_detected = {d["type"] for d in result["details"]}
        expected_types = {"email", "ssn", "api_key", "credit_card"}
        
        # At least some of the expected types should be detected
        assert len(types_detected.intersection(expected_types)) > 0
    
    def test_tool_name_context_influence(self):
        """Test how tool names influence sensitivity detection."""
        detector = SensitiveContentDetector()
        
        # Normal content that might not be sensitive alone
        borderline_content = "user@example.com"
        
        # With sensitive tool name, should be flagged
        result_sensitive_tool = detector.detect_sensitive_input(
            borderline_content, "get_user_secrets"
        )
        assert result_sensitive_tool["is_sensitive"]
        
        # With normal tool name, might still be flagged due to email pattern
        result_normal_tool = detector.detect_sensitive_input(
            borderline_content, "send_notification"
        )
        # Email should still be detected regardless of tool name
        assert result_normal_tool["is_sensitive"]


if __name__ == "__main__":
    pytest.main([__file__])