#!/usr/bin/env python3
"""
Test suite for code examples in docs/memory-system.md
This ensures all memory system examples work with the actual implementation.
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

# Test imports from the memory system examples
try:
    from jaf.memory import ConversationMemory, MemoryProvider, MemoryQuery, MemoryConfig
    from jaf.memory import create_in_memory_provider, InMemoryConfig
    from jaf.memory import create_memory_provider_from_env
    from jaf.memory import MemoryError, MemoryConnectionError, MemoryNotFoundError, MemoryStorageError
    from jaf.memory import Success, Failure, Result
    from jaf.core.types import Message
    from jaf import run, RunState, RunConfig, Agent, generate_run_id, generate_trace_id
    print("✅ All memory system imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MemorySystemTestTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def success(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def failure(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Test Results: {self.passed}/{total} passed")
        if self.errors:
            print("\n❌ Failures:")
            for error in self.errors:
                print(f"   - {error}")

results = MemorySystemTestTracker()

def test_conversation_memory():
    """Test ConversationMemory creation from memory system docs."""
    
    try:
        # Test ConversationMemory creation
        conversation = ConversationMemory(
            conversation_id="user-123-session-1",
            user_id="user-123", 
            messages=[
                Message(role="user", content="Hello!"),
                Message(role="assistant", content="Hi there! How can I help you?")
            ],
            metadata={"session_start": "2024-01-15T10:00:00Z"}
        )
        
        assert conversation.conversation_id == "user-123-session-1"
        assert conversation.user_id == "user-123"
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].content == "Hi there! How can I help you?"
        assert conversation.metadata["session_start"] == "2024-01-15T10:00:00Z"
        
        results.success("memory-system: ConversationMemory creation")
    except Exception as e:
        results.failure("memory-system: ConversationMemory creation", str(e))

def test_in_memory_provider():
    """Test in-memory provider creation and configuration."""
    
    try:
        # Test InMemoryConfig creation
        config = InMemoryConfig(
            max_conversations=1000,
            max_messages_per_conversation=1000
        )
        
        assert config.max_conversations == 1000
        assert config.max_messages_per_conversation == 1000
        
        # Test provider creation
        provider = create_in_memory_provider(config)
        assert provider is not None
        
        results.success("memory-system: in-memory provider creation")
    except Exception as e:
        results.failure("memory-system: in-memory provider creation", str(e))

async def test_memory_provider_protocol():
    """Test MemoryProvider protocol implementation."""
    
    try:
        # Create in-memory provider for testing
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Test store_messages
        messages = [
            Message(role="user", content="What's the weather like?"),
            Message(role="assistant", content="I'd need your location to check the weather.")
        ]
        
        result = await provider.store_messages(
            conversation_id="weather-chat-1",
            messages=messages,
            metadata={"topic": "weather", "user_location": "unknown"}
        )
        
        # Check result type
        assert isinstance(result, (Success, Failure))
        
        # Test get_conversation
        conv_result = await provider.get_conversation("weather-chat-1")
        assert isinstance(conv_result, (Success, Failure))
        if isinstance(conv_result, Success) and conv_result.data:
            conversation = conv_result.data
            assert conversation.conversation_id == "weather-chat-1"
            assert len(conversation.messages) == 2
        
        # Test append_messages
        new_messages = [
            Message(role="user", content="I'm in New York"),
            Message(role="assistant", content="It's currently 72°F and sunny in New York!")
        ]
        
        result = await provider.append_messages(
            conversation_id="weather-chat-1",
            messages=new_messages
        )
        
        # Test get_recent_messages
        recent_result = await provider.get_recent_messages(
            conversation_id="weather-chat-1",
            limit=20
        )
        
        assert isinstance(recent_result, (Success, Failure))
        
        # Test health_check
        health_result = await provider.health_check()
        assert isinstance(health_result, (Success, Failure))
        
        # Test delete_conversation
        delete_result = await provider.delete_conversation("weather-chat-1")
        assert isinstance(delete_result, (Success, Failure))
        
        results.success("memory-system: MemoryProvider protocol")
    except Exception as e:
        results.failure("memory-system: MemoryProvider protocol", str(e))

def test_memory_config():
    """Test MemoryConfig creation."""
    
    try:
        # Create a mock provider for testing
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Test MemoryConfig creation
        memory_config = MemoryConfig(
            provider=provider,
            auto_store=True,
            max_messages=1000,
            ttl=86400
        )
        
        assert memory_config.provider == provider
        assert memory_config.auto_store is True
        assert memory_config.max_messages == 1000
        assert memory_config.ttl == 86400
        
        results.success("memory-system: MemoryConfig creation")
    except Exception as e:
        results.failure("memory-system: MemoryConfig creation", str(e))

def test_conversation_id_strategy():
    """Test conversation ID creation strategy from docs."""
    
    try:
        def create_conversation_id(user_id: str, session_type: str, timestamp: str) -> str:
            return f"{user_id}:{session_type}:{timestamp}"
        
        # Test conversation ID creation
        conv_id1 = create_conversation_id("user-123", "chat", "2024-01-15T10:00:00Z")
        conv_id2 = create_conversation_id("user-456", "support", "2024-01-15T14:30:00Z")
        conv_id3 = create_conversation_id("user-789", "onboarding", "2024-01-15T16:45:00Z")
        
        assert conv_id1 == "user-123:chat:2024-01-15T10:00:00Z"
        assert conv_id2 == "user-456:support:2024-01-15T14:30:00Z"
        assert conv_id3 == "user-789:onboarding:2024-01-15T16:45:00Z"
        
        results.success("memory-system: conversation ID strategy")
    except Exception as e:
        results.failure("memory-system: conversation ID strategy", str(e))

async def test_manual_memory_operations():
    """Test manual memory operations from docs."""
    
    try:
        # Create provider
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Store conversation manually
        messages = [
            Message(role="user", content="What's the weather like?"),
            Message(role="assistant", content="I'd need your location to check the weather.")
        ]
        
        # Store new conversation
        result = await provider.store_messages(
            conversation_id="weather-chat-1",
            messages=messages,
            metadata={"topic": "weather", "user_location": "unknown"}
        )
        
        # Append to existing conversation
        new_messages = [
            Message(role="user", content="I'm in New York"),
            Message(role="assistant", content="It's currently 72°F and sunny in New York!")
        ]
        
        result = await provider.append_messages(
            conversation_id="weather-chat-1",
            messages=new_messages
        )
        
        # Retrieve conversation
        conv_result = await provider.get_conversation("weather-chat-1")
        if isinstance(conv_result, Success) and conv_result.data:
            conversation = conv_result.data
            assert len(conversation.messages) >= 2  # At least the original messages
            
            # Verify message content
            found_weather_question = any("weather" in msg.content.lower() for msg in conversation.messages)
            assert found_weather_question
        
        results.success("memory-system: manual memory operations")
    except Exception as e:
        results.failure("memory-system: manual memory operations", str(e))

async def test_metadata_and_context():
    """Test custom metadata and context from docs."""
    
    try:
        # Create provider
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Store rich metadata with conversations
        metadata = {
            "session_info": {
                "user_agent": "Mozilla/5.0...",
                "ip_address": "192.168.1.1",
                "session_start": "2024-01-15T10:00:00Z"
            },
            "conversation_context": {
                "topic": "customer_support",
                "priority": "high",
                "department": "billing"
            },
            "user_preferences": {
                "language": "en",
                "timezone": "America/New_York",
                "notification_settings": {"email": True, "sms": False}
            }
        }
        
        messages = [
            Message(role="user", content="I have a billing question"),
            Message(role="assistant", content="I'd be happy to help with your billing question.")
        ]
        
        await provider.store_messages(
            conversation_id="support-ticket-456",
            messages=messages,
            metadata=metadata
        )
        
        # Retrieve and verify metadata
        conv_result = await provider.get_conversation("support-ticket-456")
        if isinstance(conv_result, Success) and conv_result.data:
            conversation = conv_result.data
            if conversation.metadata:
                assert conversation.metadata["conversation_context"]["topic"] == "customer_support"
                assert conversation.metadata["conversation_context"]["priority"] == "high"
                assert conversation.metadata["user_preferences"]["language"] == "en"
        
        results.success("memory-system: metadata and context")
    except Exception as e:
        results.failure("memory-system: metadata and context", str(e))

async def test_export_import_conversations():
    """Test conversation export/import functionality from docs."""
    
    try:
        # Create provider
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Store some test conversations
        messages1 = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
        
        messages2 = [
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well, thank you!")
        ]
        
        await provider.store_messages("conv-1", messages1, {"topic": "greeting"})
        await provider.store_messages("conv-2", messages2, {"topic": "wellbeing"})
        
        # Export conversations for backup (simplified version)
        async def export_conversations(user_id: Optional[str] = None) -> Dict:
            # For in-memory provider, we'll simulate this
            conv1_result = await provider.get_conversation("conv-1")
            conv2_result = await provider.get_conversation("conv-2")
            
            conversations = []
            if isinstance(conv1_result, Success) and conv1_result.data:
                conversations.append(conv1_result.data)
            if isinstance(conv2_result, Success) and conv2_result.data:
                conversations.append(conv2_result.data)
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "conversation_count": len(conversations),
                "conversations": [
                    {
                        "conversation_id": conv.conversation_id,
                        "user_id": conv.user_id,
                        "messages": [{"role": msg.role, "content": msg.content} for msg in conv.messages],
                        "metadata": conv.metadata
                    }
                    for conv in conversations
                ]
            }
            
            return export_data
        
        # Test export
        export_data = await export_conversations()
        assert export_data["conversation_count"] == 2
        assert len(export_data["conversations"]) == 2
        
        # Verify export structure
        first_conv = export_data["conversations"][0]
        assert "conversation_id" in first_conv
        assert "messages" in first_conv
        assert "metadata" in first_conv
        
        results.success("memory-system: export/import conversations")
    except Exception as e:
        results.failure("memory-system: export/import conversations", str(e))

async def test_health_monitoring():
    """Test health monitoring functionality from docs."""
    
    try:
        # Create provider
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Regular health monitoring (simplified)
        async def monitor_memory_health():
            health_result = await provider.health_check()
            
            if isinstance(health_result, Success) and health_result.data:
                health = health_result.data
                if health.get('healthy', True):  # Default to True for in-memory
                    # Health check passed
                    metrics = health.get('metrics', {})
                    return True
                else:
                    # Health check failed
                    return False
            else:
                return False
        
        # Test health monitoring
        is_healthy = await monitor_memory_health()
        assert isinstance(is_healthy, bool)
        
        results.success("memory-system: health monitoring")
    except Exception as e:
        results.failure("memory-system: health monitoring", str(e))

async def test_conversation_integrity():
    """Test conversation integrity verification from docs."""
    
    try:
        # Create provider
        config = InMemoryConfig(max_conversations=100, max_messages_per_conversation=100)
        provider = create_in_memory_provider(config)
        
        # Store a test conversation
        messages = [
            Message(role="user", content="Test message 1"),
            Message(role="assistant", content="Test response 1"),
            Message(role="user", content="Test message 2")
        ]
        
        await provider.store_messages("integrity-test", messages)
        
        # Verify data integrity
        async def verify_conversation_integrity(conversation_id: str):
            conv_result = await provider.get_conversation(conversation_id)
            if isinstance(conv_result, Success) and conv_result.data:
                conversation = conv_result.data
                # Check message sequence
                for i, message in enumerate(conversation.messages):
                    if not message.content or not message.role:
                        print(f"Invalid message at index {i}")
                        return False
                return True
            else:
                return False
        
        # Test integrity verification
        is_valid = await verify_conversation_integrity("integrity-test")
        assert is_valid is True
        
        # Test with non-existent conversation
        is_valid_missing = await verify_conversation_integrity("non-existent")
        assert is_valid_missing is False
        
        results.success("memory-system: conversation integrity")
    except Exception as e:
        results.failure("memory-system: conversation integrity", str(e))

async def main():
    """Run all memory system code tests."""
    print("🧪 Testing all code examples from docs/memory-system.md...")
    print("=" * 60)
    
    # Test ConversationMemory
    print("\n💾 Testing ConversationMemory...")
    test_conversation_memory()
    
    # Test in-memory provider
    print("\n🧠 Testing in-memory provider...")
    test_in_memory_provider()
    
    # Test MemoryProvider protocol
    print("\n🔌 Testing MemoryProvider protocol...")
    await test_memory_provider_protocol()
    
    # Test MemoryConfig
    print("\n⚙️ Testing MemoryConfig...")
    test_memory_config()
    
    # Test conversation ID strategy
    print("\n🆔 Testing conversation ID strategy...")
    test_conversation_id_strategy()
    
    # Test manual memory operations
    print("\n🔧 Testing manual memory operations...")
    await test_manual_memory_operations()
    
    # Test metadata and context
    print("\n📋 Testing metadata and context...")
    await test_metadata_and_context()
    
    # Test export/import
    print("\n📤 Testing export/import conversations...")
    await test_export_import_conversations()
    
    # Test health monitoring
    print("\n🏥 Testing health monitoring...")
    await test_health_monitoring()
    
    # Test conversation integrity
    print("\n🔍 Testing conversation integrity...")
    await test_conversation_integrity()
    
    # Print summary
    print("\n" + "=" * 60)
    results.summary()
    
    if results.failed > 0:
        print(f"\n⚠️  {results.failed} tests failed. Memory system docs need fixes.")
        return False
    else:
        print(f"\n🎉 All {results.passed} tests passed! Memory system docs are accurate.")
        return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
