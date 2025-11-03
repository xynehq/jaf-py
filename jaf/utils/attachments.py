"""
Attachment validation and utility functions for the JAF framework.

This module provides type-safe attachment creation and validation with
comprehensive error handling and security checks.
"""

import base64
import re
from typing import Union, Optional, List
from urllib.parse import urlparse

from ..core.types import Attachment


class AttachmentValidationError(Exception):
    """Exception raised when attachment validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


# Constants
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILENAME_LENGTH = 255
BASE64_SIZE_RATIO = 0.75  # Base64 decoded size is approximately 3/4 of the encoded size
MAX_FORMAT_LENGTH = 10

ALLOWED_IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
]

ALLOWED_DOCUMENT_MIME_TYPES = [
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]


def _validate_base64(data: str) -> bool:
    """Validate base64 string format."""
    try:
        # Basic base64 pattern check
        base64_pattern = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
        if not base64_pattern.match(data):
            return False

        # Try to decode to verify it's valid base64
        decoded = base64.b64decode(data)
        reencoded = base64.b64encode(decoded).decode("ascii")

        # Account for padding differences
        normalized_input = data.rstrip("=")
        normalized_reencoded = reencoded.rstrip("=")

        return normalized_input == normalized_reencoded
    except Exception:
        return False


def _validate_attachment_size(data: Optional[str]) -> None:
    """Validate attachment size doesn't exceed limits."""
    if data:
        # Calculate exact decoded size for base64 data
        # Remove padding to get accurate count
        data_without_padding = data.rstrip("=")
        # Each 4 base64 chars encode 3 bytes, with the last group potentially having padding
        exact_groups = len(data_without_padding) // 4
        remaining_chars = len(data_without_padding) % 4

        decoded_size = exact_groups * 3
        if remaining_chars == 2:
            decoded_size += 1  # 2 chars = 1 byte
        elif remaining_chars == 3:
            decoded_size += 2  # 3 chars = 2 bytes

        if decoded_size > MAX_ATTACHMENT_SIZE:
            size_mb = round(decoded_size / 1024 / 1024, 2)
            max_mb = MAX_ATTACHMENT_SIZE // 1024 // 1024
            raise AttachmentValidationError(
                f"Attachment size ({size_mb}MB) exceeds maximum allowed size ({max_mb}MB)"
            )


def _validate_filename(name: Optional[str]) -> None:
    """Validate filename for security and length constraints."""
    if not name:
        return

    if len(name) > MAX_FILENAME_LENGTH:
        raise AttachmentValidationError(
            f"Filename length ({len(name)}) exceeds maximum allowed length ({MAX_FILENAME_LENGTH})"
        )

    # Check for dangerous characters and control characters
    dangerous_chars = re.compile(r'[<>:"|?*]')
    control_chars = re.compile(r"[\x00-\x1f]")

    if dangerous_chars.search(name) or control_chars.search(name):
        raise AttachmentValidationError("Filename contains invalid characters")

    # Check for path traversal attempts
    if ".." in name or "/" in name or "\\" in name:
        raise AttachmentValidationError(
            "Filename cannot contain path separators or traversal sequences"
        )


def _validate_mime_type(mime_type: Optional[str], allowed_types: List[str], kind: str) -> None:
    """Validate MIME type against allowed types."""
    if mime_type:
        # Normalize the input mime_type
        normalized_mime_type = mime_type.lower().strip()

        # Normalize the allowed types list
        normalized_allowed_types = {t.lower().strip() for t in allowed_types}

        if normalized_mime_type not in normalized_allowed_types:
            raise AttachmentValidationError(
                f"MIME type '{mime_type}' is not allowed for {kind} attachments. "
                f"Allowed types: {', '.join(allowed_types)}"
            )


def _validate_url(url: Optional[str]) -> None:
    """Validate URL format and protocol."""
    if not url:
        return

    try:
        parsed = urlparse(url)
        allowed_protocols = ["http", "https", "data"]

        if parsed.scheme not in allowed_protocols:
            raise AttachmentValidationError(
                f"URL protocol '{parsed.scheme}' is not allowed. "
                f"Allowed protocols: {', '.join(allowed_protocols)}"
            )

        # Additional validation for data URLs
        if parsed.scheme == "data":
            # For data URLs, the "path" component in urlparse contains the mediatype and data
            # Proper data URL format: mediatype[;charset][;base64],data
            data_content_pattern = re.compile(r"^([^;,]+)(;[^;,]+)*(;base64)?,(.+)$")
            data_content = parsed.path

            # Some URLs might have query components that are part of the data
            if parsed.query:
                data_content += "?" + parsed.query

            if not data_content_pattern.match(data_content):
                raise AttachmentValidationError(
                    "Invalid data URL format: must match mediatype[;charset][;base64],data pattern"
                )

    except ValueError as e:
        raise AttachmentValidationError(f"Invalid URL: {e}")


def _process_base64_data(data: Union[bytes, str, None]) -> Optional[str]:
    """Process and validate base64 data."""
    if not data:
        return None

    if isinstance(data, bytes):
        base64_str = base64.b64encode(data).decode("ascii")
    else:
        base64_str = data

    # Validate base64 format if it was provided as string
    if isinstance(data, str) and not _validate_base64(base64_str):
        raise AttachmentValidationError("Invalid base64 data format")

    return base64_str


def make_image_attachment(
    data: Union[bytes, str, None] = None,
    url: Optional[str] = None,
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
) -> Attachment:
    """
    Create a validated image attachment.

    Args:
        data: Raw bytes or base64 string
        url: Remote or data URL
        mime_type: MIME type (e.g., 'image/png')
        name: Optional filename

    Returns:
        Validated Attachment object

    Raises:
        AttachmentValidationError: If validation fails
    """
    # Validate inputs
    _validate_filename(name)
    _validate_url(url)
    _validate_mime_type(mime_type, ALLOWED_IMAGE_MIME_TYPES, "image")

    # Process data to base64 first, so we can validate size for both bytes and string inputs
    base64_data = _process_base64_data(data)

    # Validate size if we have data
    if base64_data:
        _validate_attachment_size(base64_data)

    # Ensure at least one content source
    if not url and not base64_data:
        raise AttachmentValidationError("Image attachment must have either url or data")

    return Attachment(kind="image", mime_type=mime_type, name=name, url=url, data=base64_data)


def make_file_attachment(
    data: Union[bytes, str, None] = None,
    url: Optional[str] = None,
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
    format: Optional[str] = None,
) -> Attachment:
    """
    Create a validated file attachment.

    Args:
        data: Raw bytes or base64 string
        url: Remote or data URL
        mime_type: MIME type
        name: Optional filename
        format: Optional format identifier (e.g., 'pdf', 'txt')

    Returns:
        Validated Attachment object

    Raises:
        AttachmentValidationError: If validation fails
    """
    # Validate inputs
    _validate_filename(name)
    _validate_url(url)

    # Process data to base64 first, so we can validate size for both bytes and string inputs
    base64_data = _process_base64_data(data)

    # Validate size if we have data
    if base64_data:
        _validate_attachment_size(base64_data)

    # Ensure at least one content source
    if not url and not base64_data:
        raise AttachmentValidationError("File attachment must have either url or data")

    # Validate format if provided
    if format and len(format) > MAX_FORMAT_LENGTH:
        raise AttachmentValidationError("File format must be 10 characters or less")

    return Attachment(
        kind="file", mime_type=mime_type, name=name, url=url, data=base64_data, format=format
    )


def make_document_attachment(
    data: Union[bytes, str, None] = None,
    url: Optional[str] = None,
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
    format: Optional[str] = None,
    use_litellm_format: Optional[bool] = None,
) -> Attachment:
    """
    Create a validated document attachment.

    Args:
        data: Raw bytes or base64 string
        url: Remote or data URL
        mime_type: MIME type
        name: Optional filename
        format: Optional format identifier
        use_litellm_format: Whether to use LiteLLM native format

    Returns:
        Validated Attachment object

    Raises:
        AttachmentValidationError: If validation fails
    """
    # Additional validation for documents
    _validate_mime_type(mime_type, ALLOWED_DOCUMENT_MIME_TYPES, "document")

    attachment = make_file_attachment(
        data=data, url=url, mime_type=mime_type, name=name, format=format
    )

    return Attachment(
        kind="document",
        mime_type=attachment.mime_type,
        name=attachment.name,
        url=attachment.url,
        data=attachment.data,
        format=attachment.format,
        use_litellm_format=use_litellm_format,
    )


def validate_attachment(attachment: Attachment) -> None:
    """
    Validate an existing attachment object.

    Args:
        attachment: Attachment to validate

    Raises:
        AttachmentValidationError: If validation fails
    """
    try:
        if not attachment.url and not attachment.data:
            raise AttachmentValidationError(
                "Attachment must have either url or data", field="url/data"
            )

        if attachment.name:
            try:
                _validate_filename(attachment.name)
            except AttachmentValidationError as e:
                raise AttachmentValidationError(f"Invalid filename: {e}", field="name") from e

        if attachment.url:
            try:
                _validate_url(attachment.url)
            except AttachmentValidationError as e:
                raise AttachmentValidationError(f"Invalid URL: {e}", field="url") from e

        if attachment.data:
            try:
                _validate_attachment_size(attachment.data)
            except AttachmentValidationError as e:
                raise AttachmentValidationError(f"Size validation failed: {e}", field="data") from e

            if not _validate_base64(attachment.data):
                raise AttachmentValidationError(
                    "Invalid base64 data format in attachment", field="data"
                )

        # Validate MIME type based on attachment kind
        if attachment.kind == "image":
            try:
                _validate_mime_type(attachment.mime_type, ALLOWED_IMAGE_MIME_TYPES, "image")
            except AttachmentValidationError as e:
                raise AttachmentValidationError(
                    f"Image MIME type validation failed: {e}", field="mime_type"
                ) from e
        elif attachment.kind == "document":
            try:
                _validate_mime_type(attachment.mime_type, ALLOWED_DOCUMENT_MIME_TYPES, "document")
            except AttachmentValidationError as e:
                raise AttachmentValidationError(
                    f"Document MIME type validation failed: {e}", field="mime_type"
                ) from e
        elif attachment.kind == "file":
            # Files can have any MIME type, but still validate format
            if attachment.format and len(attachment.format) > MAX_FORMAT_LENGTH:
                raise AttachmentValidationError(
                    f'File format "{attachment.format}" exceeds maximum length of {MAX_FORMAT_LENGTH} characters',
                    field="format",
                )
    except AttachmentValidationError:
        raise
    except Exception as e:
        raise AttachmentValidationError(f"Unexpected validation error: {e}") from e


# Legacy function for backwards compatibility
def assert_non_empty_attachment(attachment: Attachment) -> None:
    """Legacy function for backwards compatibility."""
    validate_attachment(attachment)


# Export validation constants for external use
ATTACHMENT_LIMITS = {
    "MAX_SIZE": MAX_ATTACHMENT_SIZE,
    "MAX_FILENAME_LENGTH": MAX_FILENAME_LENGTH,
    "ALLOWED_IMAGE_MIME_TYPES": ALLOWED_IMAGE_MIME_TYPES,
    "ALLOWED_DOCUMENT_MIME_TYPES": ALLOWED_DOCUMENT_MIME_TYPES,
}
