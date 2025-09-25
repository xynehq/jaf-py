"""
Simple tests for sensitive content detection functionality without pytest.
"""

from jaf.core.sensitive import (
    SensitiveContentDetector,
    SensitiveContentConfig,
    is_content_sensitive,
    get_sensitive_detector,
)


def test_tool_name_sensitivity():
    """Test sensitivity detection based on tool names."""
    print("Testing tool name sensitivity...")
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
        print(f"  ✓ {name} correctly detected as sensitive")


def test_heuristic_detection():
    """Test heuristic pattern-based detection."""
    print("Testing heuristic pattern detection...")
    detector = SensitiveContentDetector()
    
    test_cases = [
        ("My credit card number is 1234-5678-9012-3456", "credit_card"),
        ("My social security number is 123-45-6789", "ssn"),
        ("Contact me at john.doe@example.com", "email"),
        ("Here is the key: sk-1234567890abcdef1234567890abcdef", "api_key"),
        ("password: mySecretPassword123", "password"),
    ]
    
    for content, expected_type in test_cases:
        result = detector.detect_sensitive_input(content)
        assert result["is_sensitive"], f"Content with {expected_type} should be sensitive"
        types_found = [d["type"] for d in result["details"]]
        assert expected_type in types_found, f"Expected {expected_type} in {types_found}"
        print(f"  ✓ {expected_type} pattern correctly detected")


def test_non_sensitive_content():
    """Test that normal content is not flagged as sensitive."""
    print("Testing non-sensitive content...")
    detector = SensitiveContentDetector()
    
    normal_content = "The weather is sunny today. Temperature is 75 degrees."
    result = detector.detect_sensitive_input(normal_content)
    assert not result["is_sensitive"], "Normal content should not be sensitive"
    assert len(result["details"]) == 0, "Normal content should have no detection details"
    assert result["score"] == 0.0, "Normal content should have zero sensitivity score"
    print("  ✓ Normal content correctly identified as non-sensitive")


def test_custom_patterns():
    """Test custom sensitivity patterns."""
    print("Testing custom patterns...")
    config = SensitiveContentConfig(
        custom_patterns=[
            r'\btop[_-]?secret\b',
            r'\bclassified\b',
        ]
    )
    detector = SensitiveContentDetector(config)
    
    classified_content = "This document is classified and top-secret"
    result = detector.detect_sensitive_input(classified_content)
    assert result["is_sensitive"], "Custom pattern should be detected"
    types_found = [d["type"] for d in result["details"]]
    assert "custom" in types_found, f"Expected 'custom' in {types_found}"
    print("  ✓ Custom patterns correctly detected")


def test_global_functions():
    """Test global utility functions."""
    print("Testing global utility functions...")
    
    # Test with sensitive content
    sensitive_content = "API key: sk-1234567890abcdef1234567890abcdef"
    assert is_content_sensitive(sensitive_content, is_input=True), "Should detect sensitive content"
    print("  ✓ is_content_sensitive works for sensitive content")
    
    # Test with normal content
    normal_content = "Hello world"
    assert not is_content_sensitive(normal_content, is_input=True), "Should not detect normal content"
    print("  ✓ is_content_sensitive works for normal content")
    
    # Test singleton behavior
    detector1 = get_sensitive_detector()
    detector2 = get_sensitive_detector()
    assert detector1 is detector2, "Should return same instance"
    print("  ✓ get_sensitive_detector singleton behavior works")


def test_output_detection():
    """Test detection of sensitive content in outputs."""
    print("Testing output detection...")
    detector = SensitiveContentDetector()
    
    # Test output with PII
    pii_output = "User info: John Doe, SSN: 123-45-6789, Email: john@example.com"
    result = detector.detect_sensitive_output(pii_output)
    assert result["is_sensitive"], "PII output should be sensitive"
    
    # Should detect multiple types
    types_found = {d["type"] for d in result["details"]}
    assert "ssn" in types_found, "Should detect SSN"
    assert "email" in types_found, "Should detect email"
    print("  ✓ Output detection correctly identifies multiple PII types")


def test_empty_content():
    """Test handling of empty or None content."""
    print("Testing empty content handling...")
    detector = SensitiveContentDetector()
    
    # Test empty string
    result = detector.detect_sensitive_input("")
    assert not result["is_sensitive"], "Empty string should not be sensitive"
    print("  ✓ Empty string handled correctly")
    
    # Test None (should be handled gracefully)
    result = detector.detect_sensitive_input(None)
    assert not result["is_sensitive"], "None should not be sensitive"
    print("  ✓ None value handled correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Sensitive Content Detection Tests")
    print("=" * 60)
    
    try:
        test_tool_name_sensitivity()
        test_heuristic_detection()
        test_non_sensitive_content()
        test_custom_patterns()
        test_global_functions()
        test_output_detection()
        test_empty_content()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)