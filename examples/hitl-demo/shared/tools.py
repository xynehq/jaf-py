"""
File system tools for HITL demo.

This module provides file system operations with appropriate approval requirements.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field

from jaf.core.types import Tool, ToolSchema
from jaf.core.tool_results import ToolResult


@dataclass
class FileSystemContext:
    """Context for file system operations."""
    user_id: str
    working_directory: str
    permissions: List[str]


def get_context_attr(context, attr: str, default=None):
    """Helper to get attribute from context whether it's a dict or dataclass."""
    if hasattr(context, attr):
        return getattr(context, attr)
    elif isinstance(context, dict):
        return context.get(attr, default)
    else:
        return default


# Get demo directory path
DEMO_DIR = Path(__file__).parent.parent / "sandbox"


class ListFilesArgs(BaseModel):
    """Arguments for listing files."""
    directory: Optional[str] = Field(None, description="Directory to list (defaults to current directory)")


class ReadFileArgs(BaseModel):
    """Arguments for reading a file."""
    filepath: str = Field(description="Path to the file to read")


class DeleteFileArgs(BaseModel):
    """Arguments for deleting a file."""
    filepath: str = Field(description="Path to the file to delete")
    reason: Optional[str] = Field(None, description="Reason for deletion")


class EditFileArgs(BaseModel):
    """Arguments for editing a file."""
    filepath: str = Field(description="Path to the file to edit or create")
    content: str = Field(description="New content for the file")
    backup: Optional[bool] = Field(False, description="Whether to create a backup before editing")


class ListFilesTool:
    """Tool for listing files and directories."""

    def __init__(self, sandbox_dir: Path = DEMO_DIR):
        self.sandbox_dir = sandbox_dir

    @property
    def schema(self) -> ToolSchema[ListFilesArgs]:
        return ToolSchema(
            name="listFiles",
            description="List files and directories in the specified directory",
            parameters=ListFilesArgs
        )

    @property
    def needs_approval(self) -> bool:
        return False

    async def execute(self, args: ListFilesArgs, context: Any = None) -> str:
        try:
            target_dir = self.sandbox_dir
            if args.directory:
                target_dir = (self.sandbox_dir / args.directory).resolve()

            # Security check - ensure we stay within sandbox
            if not str(target_dir).startswith(str(self.sandbox_dir)):
                return f"Error: Access denied. Directory outside of sandbox: {target_dir}"

            if not target_dir.exists():
                return f"Error: Directory does not exist: {target_dir}"

            items = []
            for item in target_dir.iterdir():
                item_type = "directory" if item.is_dir() else "file"
                relative_path = item.relative_to(self.sandbox_dir)
                items.append({
                    "name": item.name,
                    "type": item_type,
                    "path": str(relative_path)
                })

            file_list = "\n".join([
                f"  {'📁' if item['type'] == 'directory' else '📄'} {item['name']}"
                for item in items
            ])

            return f"Directory listing for {target_dir}:\n{file_list}"

        except Exception as e:
            return f"Error listing directory: {str(e)}"


class ReadFileTool:
    """Tool for reading file contents."""

    def __init__(self, sandbox_dir: Path = DEMO_DIR):
        self.sandbox_dir = sandbox_dir

    @property
    def schema(self) -> ToolSchema[ReadFileArgs]:
        return ToolSchema(
            name="readFile",
            description="Read the contents of a file",
            parameters=ReadFileArgs
        )

    @property
    def needs_approval(self) -> bool:
        return False

    async def execute(self, args: ReadFileArgs, context: Any = None) -> str:
        try:
            target_path = (self.sandbox_dir / args.filepath).resolve()

            # Security check - ensure we stay within sandbox
            if not str(target_path).startswith(str(self.sandbox_dir)):
                return f"Error: Access denied. File outside of sandbox: {target_path}"

            if not target_path.exists():
                return f"Error: File does not exist: {args.filepath}"

            if not target_path.is_file():
                return f"Error: {args.filepath} is not a file"

            content = target_path.read_text(encoding='utf-8')
            return f"Contents of {args.filepath}:\n{content}"

        except Exception as e:
            return f"Error reading file: {str(e)}"


class DeleteFileTool:
    """Tool for deleting files."""

    def __init__(self, sandbox_dir: Path = DEMO_DIR):
        self.sandbox_dir = sandbox_dir

    @property
    def schema(self) -> ToolSchema[DeleteFileArgs]:
        return ToolSchema(
            name="deleteFile",
            description="Delete a file (requires approval)",
            parameters=DeleteFileArgs
        )

    @property
    def needs_approval(self) -> bool:
        return True

    async def execute(self, args: DeleteFileArgs, context: Any = None) -> str:
        try:
            target_path = (self.sandbox_dir / args.filepath).resolve()

            # Security check - ensure we stay within sandbox
            if not str(target_path).startswith(str(self.sandbox_dir)):
                return f"Error: Access denied. File outside of sandbox: {target_path}"

            if not target_path.exists():
                return f"Error: File does not exist: {args.filepath}"

            target_path.unlink()
            print(f"🗑️  File deleted: {args.filepath}")
            if args.reason:
                print(f"   Reason: {args.reason}")

            # Check for approval context
            deletion_confirmed = get_context_attr(context, 'deletion_confirmed')
            if deletion_confirmed:
                print(f"   Confirmed by: {deletion_confirmed.get('confirmed_by')}")
                print(f"   Backup created: {deletion_confirmed.get('backup_created')}")

            reason_text = f" (Reason: {args.reason})" if args.reason else ""
            return f"Successfully deleted file: {args.filepath}{reason_text}"

        except Exception as e:
            return f"Error deleting file: {str(e)}"


class EditFileTool:
    """Tool for editing or creating files."""

    def __init__(self, sandbox_dir: Path = DEMO_DIR):
        self.sandbox_dir = sandbox_dir

    @property
    def schema(self) -> ToolSchema[EditFileArgs]:
        return ToolSchema(
            name="editFile",
            description="Edit or create a file with new content (requires approval)",
            parameters=EditFileArgs
        )

    @property
    def needs_approval(self) -> bool:
        return True

    async def execute(self, args: EditFileArgs, context: Any = None) -> str:
        try:
            target_path = (self.sandbox_dir / args.filepath).resolve()

            # Security check - ensure we stay within sandbox
            if not str(target_path).startswith(str(self.sandbox_dir)):
                return f"Error: Access denied. File outside of sandbox: {target_path}"

            backup_path = ""
            if args.backup and target_path.exists():
                backup_path = f"{target_path}.backup.{int(asyncio.get_event_loop().time() * 1000)}"
                target_path.rename(backup_path)
                print(f"💾 Backup created: {backup_path}")

            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(args.content, encoding='utf-8')
            print(f"✏️  File edited: {args.filepath}")

            # Check for approval context
            editing_approved = get_context_attr(context, 'editing_approved')
            if editing_approved:
                print(f"   Approved by: {editing_approved.get('approved_by')}")
                print(f"   Safety level: {editing_approved.get('safety_level')}")

            backup_text = f" (Backup: {Path(backup_path).name})" if backup_path else ""
            return f"Successfully edited file: {args.filepath}{backup_text}"

        except Exception as e:
            return f"Error editing file: {str(e)}"


# Create tool instances with default sandbox configuration
list_files_tool = ListFilesTool()
read_file_tool = ReadFileTool()
delete_file_tool = DeleteFileTool()
edit_file_tool = EditFileTool()