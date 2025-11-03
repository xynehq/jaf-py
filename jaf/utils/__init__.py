"""
Utility modules for the JAF framework.

This package provides various utility functions and classes for working
with attachments, document processing, and other common tasks.
"""

# Import attachment utilities
from .attachments import (
    make_image_attachment,
    make_file_attachment,
    make_document_attachment,
    validate_attachment,
    assert_non_empty_attachment,
    AttachmentValidationError,
    ATTACHMENT_LIMITS,
)

# Import document processing utilities
from .document_processor import (
    extract_document_content,
    is_document_supported,
    get_document_description,
    get_missing_dependencies,
    check_dependencies,
    ProcessedDocument,
    DocumentProcessingError,
    NetworkError,
)

__all__ = [
    # Attachment utilities
    "make_image_attachment",
    "make_file_attachment",
    "make_document_attachment",
    "validate_attachment",
    "assert_non_empty_attachment",
    "AttachmentValidationError",
    "ATTACHMENT_LIMITS",
    # Document processing
    "extract_document_content",
    "is_document_supported",
    "get_document_description",
    "get_missing_dependencies",
    "check_dependencies",
    "ProcessedDocument",
    "DocumentProcessingError",
    "NetworkError",
]
