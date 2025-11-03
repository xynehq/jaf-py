"""
Approval storage interface and implementations for Human-in-the-Loop (HITL) functionality.

This module provides persistent storage for tool approval decisions, enabling
the framework to maintain approval states across conversation sessions and
handle interruptions gracefully.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio

from ..core.types import RunId, ApprovalValue


class ApprovalStorageResult:
    """Result wrapper for approval storage operations."""

    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error

    @classmethod
    def success_result(cls, data: Any = None) -> "ApprovalStorageResult":
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def error_result(cls, error: str) -> "ApprovalStorageResult":
        """Create an error result."""
        return cls(success=False, error=error)


class ApprovalStorage(ABC):
    """Abstract interface for approval storage implementations."""

    @abstractmethod
    async def store_approval(
        self,
        run_id: RunId,
        tool_call_id: str,
        approval: ApprovalValue,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalStorageResult:
        """Store an approval decision for a tool call."""
        pass

    @abstractmethod
    async def get_approval(self, run_id: RunId, tool_call_id: str) -> ApprovalStorageResult:
        """Retrieve approval for a specific tool call. Returns None if not found."""
        pass

    @abstractmethod
    async def get_run_approvals(self, run_id: RunId) -> ApprovalStorageResult:
        """Get all approvals for a run as a Dict[str, ApprovalValue]."""
        pass

    @abstractmethod
    async def update_approval(
        self, run_id: RunId, tool_call_id: str, updates: Dict[str, Any]
    ) -> ApprovalStorageResult:
        """Update existing approval with additional context."""
        pass

    @abstractmethod
    async def delete_approval(self, run_id: RunId, tool_call_id: str) -> ApprovalStorageResult:
        """Delete approval for a tool call. Returns success status."""
        pass

    @abstractmethod
    async def clear_run_approvals(self, run_id: RunId) -> ApprovalStorageResult:
        """Clear all approvals for a run. Returns count of deleted approvals."""
        pass

    @abstractmethod
    async def get_stats(self) -> ApprovalStorageResult:
        """Get approval statistics."""
        pass

    @abstractmethod
    async def health_check(self) -> ApprovalStorageResult:
        """Health check for the approval storage."""
        pass

    @abstractmethod
    async def close(self) -> ApprovalStorageResult:
        """Close/cleanup the storage."""
        pass


class InMemoryApprovalStorage(ApprovalStorage):
    """In-memory implementation of ApprovalStorage for development and testing."""

    def __init__(self):
        self._approvals: Dict[str, Dict[str, ApprovalValue]] = {}
        self._lock = asyncio.Lock()

    def _get_run_key(self, run_id: RunId) -> str:
        """Generate a consistent key for a run."""
        return f"run:{run_id}"

    async def store_approval(
        self,
        run_id: RunId,
        tool_call_id: str,
        approval: ApprovalValue,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalStorageResult:
        """Store an approval decision."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)

                if run_key not in self._approvals:
                    self._approvals[run_key] = {}

                # Enhance approval with metadata if provided
                enhanced_approval = approval
                if metadata:
                    additional_context = {**(approval.additional_context or {}), **metadata}
                    enhanced_approval = ApprovalValue(
                        status=approval.status,
                        approved=approval.approved,
                        additional_context=additional_context,
                    )

                self._approvals[run_key][tool_call_id] = enhanced_approval

            return ApprovalStorageResult.success_result()
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to store approval: {e}")

    async def get_approval(self, run_id: RunId, tool_call_id: str) -> ApprovalStorageResult:
        """Retrieve approval for a specific tool call."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)
                run_approvals = self._approvals.get(run_key, {})
                approval = run_approvals.get(tool_call_id)

            return ApprovalStorageResult.success_result(approval)
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to get approval: {e}")

    async def get_run_approvals(self, run_id: RunId) -> ApprovalStorageResult:
        """Get all approvals for a run."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)
                run_approvals = self._approvals.get(run_key, {}).copy()

            return ApprovalStorageResult.success_result(run_approvals)
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to get run approvals: {e}")

    async def update_approval(
        self, run_id: RunId, tool_call_id: str, updates: Dict[str, Any]
    ) -> ApprovalStorageResult:
        """Update existing approval."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)

                if run_key not in self._approvals or tool_call_id not in self._approvals[run_key]:
                    return ApprovalStorageResult.error_result(
                        f"Approval not found for tool call {tool_call_id} in run {run_id}"
                    )

                existing = self._approvals[run_key][tool_call_id]

                # Merge additional context
                merged_context = {
                    **(existing.additional_context or {}),
                    **(updates.get("additional_context", {})),
                }

                updated_approval = ApprovalValue(
                    status=updates.get("status", existing.status),
                    approved=updates.get("approved", existing.approved),
                    additional_context=merged_context
                    if merged_context
                    else existing.additional_context,
                )

                self._approvals[run_key][tool_call_id] = updated_approval

            return ApprovalStorageResult.success_result()
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to update approval: {e}")

    async def delete_approval(self, run_id: RunId, tool_call_id: str) -> ApprovalStorageResult:
        """Delete approval for a tool call."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)

                if run_key not in self._approvals:
                    return ApprovalStorageResult.success_result(False)

                deleted = self._approvals[run_key].pop(tool_call_id, None) is not None

                # Clean up empty run maps
                if not self._approvals[run_key]:
                    del self._approvals[run_key]

            return ApprovalStorageResult.success_result(deleted)
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to delete approval: {e}")

    async def clear_run_approvals(self, run_id: RunId) -> ApprovalStorageResult:
        """Clear all approvals for a run."""
        try:
            async with self._lock:
                run_key = self._get_run_key(run_id)

                if run_key not in self._approvals:
                    return ApprovalStorageResult.success_result(0)

                count = len(self._approvals[run_key])
                del self._approvals[run_key]

            return ApprovalStorageResult.success_result(count)
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to clear run approvals: {e}")

    async def get_stats(self) -> ApprovalStorageResult:
        """Get approval statistics."""
        try:
            async with self._lock:
                total_approvals = 0
                approved_count = 0
                rejected_count = 0
                runs_with_approvals = len(self._approvals)

                for run_approvals in self._approvals.values():
                    for approval in run_approvals.values():
                        total_approvals += 1
                        if approval.approved:
                            approved_count += 1
                        else:
                            rejected_count += 1

                stats = {
                    "total_approvals": total_approvals,
                    "approved_count": approved_count,
                    "rejected_count": rejected_count,
                    "runs_with_approvals": runs_with_approvals,
                }

            return ApprovalStorageResult.success_result(stats)
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to get stats: {e}")

    async def health_check(self) -> ApprovalStorageResult:
        """Health check for the storage."""
        try:
            # Simple operation to test functionality
            await asyncio.sleep(0.001)  # Minimal async operation

            health_data = {
                "healthy": True,
                "latency_ms": 1.0,  # Approximate for in-memory
            }

            return ApprovalStorageResult.success_result(health_data)
        except Exception as e:
            health_data = {"healthy": False, "error": str(e)}
            return ApprovalStorageResult.success_result(health_data)

    async def close(self) -> ApprovalStorageResult:
        """Close/cleanup the storage."""
        try:
            async with self._lock:
                self._approvals.clear()
            return ApprovalStorageResult.success_result()
        except Exception as e:
            return ApprovalStorageResult.error_result(f"Failed to close storage: {e}")


def create_in_memory_approval_storage() -> InMemoryApprovalStorage:
    """Create an in-memory approval storage instance."""
    return InMemoryApprovalStorage()
