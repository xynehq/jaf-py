"""
Regression tests for attachment loss across JAF's persisted memory providers.

Bug: `Message.attachments` (and list-based multi-part `content`) were silently
dropped by `jaf.memory.utils.serialize_message` / `deserialize_message`, which
every DB-backed memory provider (Postgres, Redis) funnels through when storing
or loading a conversation. Because `regenerate_conversation()` always reloads
its working conversation from the memory provider (see `jaf/core/regeneration.py`),
any attachment sent on the *first* turn of a conversation was silently lost the
moment that turn was auto-stored to memory -- so retry/regenerate requests
never saw the original attachment, even though the caller resent it.

These tests cover:
1. `serialize_message` / `deserialize_message` round-trip attachments and
   multi-part content directly.
2. `prepare_message_list_for_db` / `extract_messages_from_db_row` (the exact
   JSON round trip Postgres/Redis providers perform) preserve attachments.
3. End-to-end: a "pure" regeneration (retry) against a memory provider that
   round-trips messages through JSON (simulating Postgres/Redis) still has
   the original user attachment available to the model.
"""

from typing import Any, Dict, List, Optional

import pytest

from jaf.core.types import (
    Agent,
    ContentRole,
    Message,
    MessageContentPart,
    Attachment,
    RegenerationRequest,
    generate_message_id,
)
from jaf.core.regeneration import regenerate_conversation
from jaf.memory.types import InMemoryConfig, MemoryConfig, Result
from jaf.memory.providers.in_memory import InMemoryProvider
from jaf.memory.utils import (
    serialize_message,
    deserialize_message,
    prepare_message_list_for_db,
    extract_messages_from_db_row,
)
from jaf.core.types import RunConfig


def _sample_image_attachment() -> Attachment:
    return Attachment(
        kind="image",
        mime_type="image/png",
        name="receipt.png",
        data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI/0VKOuQAAAABJRU5ErkJggg==",
    )


class TestMessageSerializationRoundTrip:
    """Direct unit tests for jaf.memory.utils serialize/deserialize helpers."""

    def test_attachments_survive_dict_round_trip(self):
        attachment = _sample_image_attachment()
        msg = Message(role=ContentRole.USER, content="Please analyze this image", attachments=[attachment])

        restored = deserialize_message(serialize_message(msg))

        assert restored.attachments is not None
        assert len(restored.attachments) == 1
        assert restored.attachments[0].kind == "image"
        assert restored.attachments[0].mime_type == "image/png"
        assert restored.attachments[0].name == "receipt.png"
        assert restored.attachments[0].data == attachment.data

    def test_no_attachments_round_trips_to_none(self):
        msg = Message(role=ContentRole.USER, content="Hello, no attachments here")
        restored = deserialize_message(serialize_message(msg))
        assert restored.attachments is None

    def test_multipart_content_survives_dict_round_trip(self):
        parts = [
            MessageContentPart(type="text", text="Look at this: "),
            MessageContentPart(
                type="image_url", image_url={"url": "data:image/png;base64,abc123"}
            ),
        ]
        msg = Message(role=ContentRole.USER, content=parts)

        restored = deserialize_message(serialize_message(msg))

        assert isinstance(restored.content, list)
        assert len(restored.content) == 2
        assert restored.content[0].type == "text"
        assert restored.content[0].text == "Look at this: "
        assert restored.content[1].type == "image_url"
        assert restored.content[1].image_url == {"url": "data:image/png;base64,abc123"}

    def test_attachments_survive_json_db_round_trip(self):
        """Simulates exactly what Postgres/Redis providers do: JSON-encode a
        list of messages for storage, then decode them back."""
        attachment = _sample_image_attachment()
        messages = [
            Message(role=ContentRole.USER, content="Analyze this file", attachments=[attachment]),
            Message(role=ContentRole.ASSISTANT, content="Sure, here's the analysis..."),
        ]

        db_json = prepare_message_list_for_db(messages)
        restored = extract_messages_from_db_row(db_json)

        assert len(restored) == 2
        assert restored[0].attachments is not None
        assert restored[0].attachments[0].data == attachment.data
        assert restored[1].attachments is None


class _JsonRoundTrippingInMemoryProvider(InMemoryProvider):
    """Wraps the real in-memory provider but forces every stored/loaded
    message through the same JSON (de)serialization Postgres/Redis use, so
    tests can catch bugs in `serialize_message`/`deserialize_message` that
    the plain in-memory provider (which stores live Python objects) would
    never surface."""

    async def store_messages(
        self,
        conversation_id: str,
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result:
        round_tripped = extract_messages_from_db_row(prepare_message_list_for_db(messages))
        return await super().store_messages(conversation_id, round_tripped, metadata)


class MockModelProvider:
    """Mock model provider that records the state it was called with so tests
    can assert on exactly what would have been sent to the LLM."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        self.seen_states: List[Any] = []

    async def get_completion(self, state, agent, config):
        self.seen_states.append(state)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return {"message": {"content": response}}


def _test_agent() -> Agent:
    return Agent(
        name="TestAgent",
        instructions=lambda state: "You are a helpful assistant.",
        tools=[],
    )


@pytest.mark.asyncio
async def test_pure_regeneration_preserves_user_attachment_through_persisted_memory():
    """End-to-end regression test for the retry/regenerate attachment-loss bug.

    Reproduces the real flow: a user sends a message with an image attachment,
    the turn is auto-stored to a JSON-persisted memory provider, and then the
    assistant's reply is regenerated (retry) using only the stored
    conversation -- no fresh attachments are supplied by the caller. Before
    the fix, the attachment would be missing from the state handed to the
    model on regeneration; after the fix, it survives.
    """
    provider = _JsonRoundTrippingInMemoryProvider(
        InMemoryConfig(type="memory", max_conversations=10, max_messages_per_conversation=100)
    )
    conversation_id = "conv-attachment-retry"
    attachment = _sample_image_attachment()

    user_message_id = generate_message_id()
    assistant_message_id = generate_message_id()

    await provider.store_messages(
        conversation_id,
        [
            Message(
                role=ContentRole.USER,
                content="What's in this receipt?",
                attachments=[attachment],
                message_id=user_message_id,
            ),
            Message(
                role=ContentRole.ASSISTANT,
                content="It looks like a grocery receipt.",
                message_id=assistant_message_id,
            ),
        ],
        {"user_id": "test-user"},
    )

    model_provider = MockModelProvider(["It looks like a grocery receipt, take two."])
    run_config = RunConfig(
        agent_registry={"TestAgent": _test_agent()},
        model_provider=model_provider,
        memory=MemoryConfig(provider=provider, auto_store=True),
        conversation_id=conversation_id,
        max_turns=10,
    )

    regeneration_request = RegenerationRequest(
        conversation_id=conversation_id,
        message_id=assistant_message_id,
        context={},
    )

    result = await regenerate_conversation(
        regeneration_request, run_config, {"user_id": "test-user"}, "TestAgent"
    )

    assert result.outcome.status == "completed", getattr(result.outcome, "error", None)
    assert model_provider.seen_states, "Model provider was never invoked"

    # The exact state passed to the model for the regenerated turn must still
    # carry the original user attachment.
    seen_user_messages = [
        m for m in model_provider.seen_states[-1].messages if m.role in ("user", ContentRole.USER)
    ]
    assert seen_user_messages, "No user message found in state sent to the model"
    assert any(
        m.attachments and any(a.kind == "image" and a.data == attachment.data for a in m.attachments)
        for m in seen_user_messages
    ), "Original attachment was lost by regeneration time"

    # Also verify what got persisted back to memory still has the attachment.
    stored = await provider.get_conversation(conversation_id)
    stored_user_messages = [m for m in stored.data.messages if m.role in ("user", ContentRole.USER)]
    assert any(m.attachments for m in stored_user_messages), (
        "Attachment missing from conversation persisted after regeneration"
    )
