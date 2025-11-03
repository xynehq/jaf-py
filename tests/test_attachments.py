"""
Tests for attachment functionality in the JAF framework.

This module tests attachment creation, validation, document processing,
and server integration.
"""

import base64
import json
import pytest
from typing import List

from jaf.core.types import Message, MessageContentPart, Attachment, get_text_content
from jaf.utils.attachments import (
    make_image_attachment,
    make_file_attachment,
    make_document_attachment,
    validate_attachment,
    AttachmentValidationError,
    ATTACHMENT_LIMITS,
)
from jaf.utils.document_processor import (
    is_document_supported,
    get_document_description,
    check_dependencies,
)
from jaf.server.types import HttpMessage, HttpAttachment, HttpMessageContentPart


class TestAttachmentCreation:
    """Test attachment creation utilities."""

    def test_make_image_attachment_with_data(self):
        """Test creating image attachment with base64 data."""
        # 1x1 red pixel PNG
        red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI/0VKOuQAAAABJRU5ErkJggg=="

        attachment = make_image_attachment(
            data=red_pixel_b64, mime_type="image/png", name="red_pixel.png"
        )

        assert attachment.kind == "image"
        assert attachment.mime_type == "image/png"
        assert attachment.name == "red_pixel.png"
        assert attachment.data == red_pixel_b64
        assert attachment.url is None

    def test_make_image_attachment_with_url(self):
        """Test creating image attachment with URL."""
        attachment = make_image_attachment(
            url="https://example.com/image.jpg", mime_type="image/jpeg", name="example.jpg"
        )

        assert attachment.kind == "image"
        assert attachment.mime_type == "image/jpeg"
        assert attachment.name == "example.jpg"
        assert attachment.url == "https://example.com/image.jpg"
        assert attachment.data is None

    def test_make_file_attachment(self):
        """Test creating file attachment."""
        text_data = base64.b64encode(b"Hello, world!").decode("ascii")

        attachment = make_file_attachment(
            data=text_data, mime_type="text/plain", name="hello.txt", format="txt"
        )

        assert attachment.kind == "file"
        assert attachment.mime_type == "text/plain"
        assert attachment.name == "hello.txt"
        assert attachment.format == "txt"
        assert attachment.data == text_data

    def test_make_document_attachment(self):
        """Test creating document attachment."""
        json_data = base64.b64encode(b'{"test": true}').decode("ascii")

        attachment = make_document_attachment(
            data=json_data, mime_type="application/json", name="test.json", use_litellm_format=True
        )

        assert attachment.kind == "document"
        assert attachment.mime_type == "application/json"
        assert attachment.name == "test.json"
        assert attachment.use_litellm_format is True
        assert attachment.data == json_data

    def test_attachment_validation_errors(self):
        """Test attachment validation errors."""
        # No data or URL
        with pytest.raises(AttachmentValidationError, match="must have either url or data"):
            make_image_attachment()

        # Invalid MIME type for image
        with pytest.raises(AttachmentValidationError, match="MIME type.*not allowed"):
            make_image_attachment(data="test", mime_type="application/pdf")

        # Invalid filename
        with pytest.raises(AttachmentValidationError, match="path separators"):
            make_file_attachment(data="test", name="../../../etc/passwd")

        # Format too long
        with pytest.raises(AttachmentValidationError, match="format must be 10 characters or less"):
            make_file_attachment(data="test", format="verylongformat")


class TestAttachmentValidation:
    """Test attachment validation functions."""

    def test_validate_attachment_success(self):
        """Test successful attachment validation."""
        attachment = Attachment(
            kind="image",
            mime_type="image/png",
            name="test.png",
            data="dGVzdA==",  # "test" in base64
        )

        # Should not raise an exception
        validate_attachment(attachment)

    def test_validate_attachment_failures(self):
        """Test attachment validation failures."""
        # Missing data and URL
        with pytest.raises(AttachmentValidationError):
            validate_attachment(Attachment(kind="image"))

        # Invalid base64
        with pytest.raises(AttachmentValidationError):
            validate_attachment(Attachment(kind="image", data="invalid-base64!!!"))


class TestMessageContentTypes:
    """Test message content types and utilities."""

    def test_get_text_content_string(self):
        """Test getting text content from string."""
        content = "Hello, world!"
        assert get_text_content(content) == "Hello, world!"

    def test_get_text_content_parts(self):
        """Test getting text content from content parts."""
        parts = [
            MessageContentPart(type="text", text="Hello, "),
            MessageContentPart(type="image_url", image_url={"url": "test.jpg"}),
            MessageContentPart(type="text", text="world!"),
        ]

        assert get_text_content(parts) == "Hello,  world!"

    def test_message_with_attachments(self):
        """Test creating message with attachments."""
        attachment = Attachment(
            kind="image", mime_type="image/png", name="test.png", data="dGVzdA=="
        )

        message = Message(
            role="user", content="Please analyze this image", attachments=[attachment]
        )

        assert message.role == "user"
        assert get_text_content(message.content) == "Please analyze this image"
        assert len(message.attachments) == 1
        assert message.attachments[0].kind == "image"


class TestDocumentProcessor:
    """Test document processing utilities."""

    def test_is_document_supported(self):
        """Test document support detection."""
        assert is_document_supported("application/pdf") is True
        assert is_document_supported("text/plain") is True
        assert is_document_supported("text/csv") is True
        assert is_document_supported("application/json") is True
        assert is_document_supported("image/png") is False
        assert is_document_supported("unknown/type") is False
        assert is_document_supported(None) is False

    def test_get_document_description(self):
        """Test document description generation."""
        assert "PDF" in get_document_description("application/pdf")
        assert "text" in get_document_description("text/plain")
        assert "CSV" in get_document_description("text/csv")
        assert "JSON" in get_document_description("application/json")
        assert "document content" == get_document_description("unknown/type")
        assert "document content" == get_document_description(None)

    def test_check_dependencies(self):
        """Test dependency checking."""
        deps = check_dependencies()

        assert isinstance(deps, dict)
        assert "pdf" in deps
        assert "docx" in deps
        assert "excel" in deps
        assert "image" in deps
        assert "magic" in deps

        # All values should be boolean
        for value in deps.values():
            assert isinstance(value, bool)


class TestServerTypes:
    """Test server type conversions."""

    def test_http_message_basic(self):
        """Test basic HTTP message."""
        http_msg = HttpMessage(role="user", content="Hello!")

        assert http_msg.role == "user"
        assert http_msg.content == "Hello!"
        assert http_msg.attachments is None

    def test_http_message_with_attachments(self):
        """Test HTTP message with attachments."""
        attachment = HttpAttachment(
            kind="image", mime_type="image/png", name="test.png", data="dGVzdA=="
        )

        http_msg = HttpMessage(role="user", content="Analyze this image", attachments=[attachment])

        assert http_msg.role == "user"
        assert http_msg.content == "Analyze this image"
        assert len(http_msg.attachments) == 1
        assert http_msg.attachments[0].kind == "image"

    def test_http_message_multipart_content(self):
        """Test HTTP message with multi-part content."""
        parts = [
            HttpMessageContentPart(type="text", text="Look at this: "),
            HttpMessageContentPart(
                type="image_url", image_url={"url": "https://example.com/image.jpg"}
            ),
        ]

        http_msg = HttpMessage(role="user", content=parts)

        assert http_msg.role == "user"
        assert isinstance(http_msg.content, list)
        assert len(http_msg.content) == 2
        assert http_msg.content[0].type == "text"
        assert http_msg.content[1].type == "image_url"


class TestAttachmentLimits:
    """Test attachment limits and constants."""

    def test_attachment_limits_constants(self):
        """Test attachment limits constants."""
        assert "MAX_SIZE" in ATTACHMENT_LIMITS
        assert "MAX_FILENAME_LENGTH" in ATTACHMENT_LIMITS
        assert "ALLOWED_IMAGE_MIME_TYPES" in ATTACHMENT_LIMITS
        assert "ALLOWED_DOCUMENT_MIME_TYPES" in ATTACHMENT_LIMITS

        assert ATTACHMENT_LIMITS["MAX_SIZE"] > 0
        assert ATTACHMENT_LIMITS["MAX_FILENAME_LENGTH"] > 0
        assert len(ATTACHMENT_LIMITS["ALLOWED_IMAGE_MIME_TYPES"]) > 0
        assert len(ATTACHMENT_LIMITS["ALLOWED_DOCUMENT_MIME_TYPES"]) > 0

    def test_allowed_mime_types(self):
        """Test allowed MIME types."""
        image_types = ATTACHMENT_LIMITS["ALLOWED_IMAGE_MIME_TYPES"]
        doc_types = ATTACHMENT_LIMITS["ALLOWED_DOCUMENT_MIME_TYPES"]

        # Check common image types
        assert "image/jpeg" in image_types
        assert "image/png" in image_types
        assert "image/gif" in image_types

        # Check common document types
        assert "application/pdf" in doc_types
        assert "text/plain" in doc_types
        assert "text/csv" in doc_types
        assert "application/json" in doc_types


@pytest.mark.asyncio
class TestAsyncDocumentProcessor:
    """Test async document processing functionality."""

    async def test_extract_text_content(self):
        """Test extracting text content from simple text."""
        from jaf.utils.document_processor import _extract_text_content

        text_data = b"Hello, world!\nThis is a test."
        result = _extract_text_content(text_data, "text/plain")

        assert result.content == "Hello, world!\nThis is a test."
        assert result.metadata is None

    async def test_extract_csv_content(self):
        """Test extracting CSV content."""
        from jaf.utils.document_processor import _extract_text_content

        csv_data = b"Name,Age\nJohn,30\nJane,25"
        result = _extract_text_content(csv_data, "text/csv")

        assert "CSV File Content" in result.content
        assert "Name,Age" in result.content
        assert result.metadata is not None
        assert "rows" in result.metadata
        assert "columns" in result.metadata

    async def test_extract_json_content(self):
        """Test extracting JSON content."""
        from jaf.utils.document_processor import _extract_json_content

        json_data = b'{"name": "test", "value": 42}'
        result = _extract_json_content(json_data)

        assert "JSON File Content" in result.content
        assert '"name": "test"' in result.content
        assert result.metadata is not None
        assert result.metadata["type"] == "dict"


# Integration tests would go here if we had a running server
# For now, we'll keep these as unit tests to avoid external dependencies
