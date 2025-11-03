"""
A2A Task Cleanup Service for JAF

This module provides pure functional cleanup and expiration policies for A2A tasks.
It includes configurable cleanup policies, scheduling utilities, and batch processing
for maintaining optimal storage performance.
"""

import asyncio
import dataclasses
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from ...memory.types import Success
from ..types import TaskState
from .types import (
    A2AResult,
    A2ATask,
    A2ATaskProvider,
    A2ATaskQuery,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_storage_error,
)


@dataclass(frozen=True)
class A2ATaskCleanupConfig:
    """Configuration for task cleanup policies"""

    enabled: bool = True
    interval: int = 3600  # Cleanup interval in seconds (1 hour)
    max_age: Optional[int] = 7 * 24 * 60 * 60  # Max age in seconds (7 days), None to disable
    max_completed_tasks: Optional[int] = 1000  # Max completed tasks, None to disable
    max_failed_tasks: Optional[int] = 500  # Max failed tasks, None to disable
    retain_states: List[str] = None  # Task states to always retain
    batch_size: int = 100
    dry_run: bool = False

    def __post_init__(self):
        if self.retain_states is None:
            # Default states to retain if not provided
            object.__setattr__(self, "retain_states", ["working", "submitted"])


@dataclass(frozen=True)
class CleanupResult:
    """Result type for cleanup operations"""

    expired_cleaned: int
    excess_completed_cleaned: int
    excess_failed_cleaned: int
    total_cleaned: int
    errors: List[str]
    would_delete_count: int = 0


# Default cleanup configuration
default_cleanup_config = A2ATaskCleanupConfig()


def validate_cleanup_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function to validate cleanup configuration
    """
    errors: List[str] = []

    if "interval" in config and config["interval"] <= 0:
        errors.append("Cleanup interval must be greater than 0")
    if "max_age" in config and config["max_age"] is not None and config["max_age"] < 0:
        errors.append("Max age must be non-negative")
    if (
        "max_completed_tasks" in config
        and config["max_completed_tasks"] is not None
        and config["max_completed_tasks"] < 0
    ):
        errors.append("Max completed tasks must be non-negative")
    if (
        "max_failed_tasks" in config
        and config["max_failed_tasks"] is not None
        and config["max_failed_tasks"] < 0
    ):
        errors.append("Max failed tasks must be non-negative")
    if "batch_size" in config and config["batch_size"] <= 0:
        errors.append("Batch size must be greater than 0")

    return {"valid": len(errors) == 0, "errors": errors}


async def perform_task_cleanup(
    task_provider: A2ATaskProvider,
    config: A2ATaskCleanupConfig = default_cleanup_config,
    context_id: Optional[str] = None,
) -> A2AResult[CleanupResult]:
    """
    Performs task cleanup based on the provided configuration.
    """
    # Validate config first
    validation = validate_cleanup_config(dataclasses.asdict(config))
    if not validation["valid"]:
        error_msg = f"Invalid cleanup configuration: {', '.join(validation['errors'])}"
        return create_a2a_failure(
            create_a2a_task_storage_error("cleanup", "validation", None, ValueError(error_msg))
        )

    if not config.enabled:
        return create_a2a_success(CleanupResult(0, 0, 0, 0, [], 0))

    errors: List[str] = []
    tasks_to_delete: Set[str] = set()

    # 1. Age-based cleanup
    if config.max_age is not None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(seconds=config.max_age)

        # Find all tasks older than the cutoff date
        # We query all states and filter out the ones to retain
        try:
            all_old_tasks_result = await task_provider.find_tasks(
                A2ATaskQuery(
                    context_id=context_id,
                    until=cutoff_date,
                    limit=10000,  # A high limit to get a large batch
                )
            )

            if all_old_tasks_result.data:
                for task in all_old_tasks_result.data:
                    if task.status.state.value not in config.retain_states:
                        tasks_to_delete.add(task.id)
            elif all_old_tasks_result.error:
                errors.append(f"Failed to query old tasks: {all_old_tasks_result.error.message}")

        except Exception as e:
            errors.append(f"Error during age-based cleanup query: {e!s}")

    expired_cleaned = len(tasks_to_delete)

    # 2. Count-based cleanup for completed tasks
    excess_completed_cleaned = 0
    if config.max_completed_tasks is not None:
        try:
            completed_tasks_result = await task_provider.find_tasks(
                A2ATaskQuery(
                    state=TaskState.COMPLETED,
                    context_id=context_id,
                    limit=config.max_completed_tasks + 200,
                )
            )
            if completed_tasks_result.data:
                tasks = [t for t in completed_tasks_result.data if t.id not in tasks_to_delete]
                if len(tasks) > config.max_completed_tasks:
                    tasks.sort(
                        key=lambda t: t.status.timestamp
                        or datetime.min.replace(tzinfo=timezone.utc)
                    )
                    num_to_delete = len(tasks) - config.max_completed_tasks
                    for i in range(num_to_delete):
                        tasks_to_delete.add(tasks[i].id)
                        excess_completed_cleaned += 1
            elif completed_tasks_result.error:
                errors.append(
                    f"Failed to query completed tasks: {completed_tasks_result.error.message}"
                )
        except Exception as e:
            errors.append(f"Error during completed task count cleanup: {e!s}")

    # 3. Count-based cleanup for failed tasks
    excess_failed_cleaned = 0
    if config.max_failed_tasks is not None:
        try:
            failed_tasks_result = await task_provider.find_tasks(
                A2ATaskQuery(
                    state=TaskState.FAILED,
                    context_id=context_id,
                    limit=config.max_failed_tasks + 200,
                )
            )
            if failed_tasks_result.data:
                tasks = [t for t in failed_tasks_result.data if t.id not in tasks_to_delete]
                if len(tasks) > config.max_failed_tasks:
                    tasks.sort(
                        key=lambda t: t.status.timestamp
                        or datetime.min.replace(tzinfo=timezone.utc)
                    )
                    num_to_delete = len(tasks) - config.max_failed_tasks
                    for i in range(num_to_delete):
                        tasks_to_delete.add(tasks[i].id)
                        excess_failed_cleaned += 1
            elif failed_tasks_result.error:
                errors.append(f"Failed to query failed tasks: {failed_tasks_result.error.message}")
        except Exception as e:
            errors.append(f"Error during failed task count cleanup: {e!s}")

    if config.dry_run:
        return create_a2a_success(
            CleanupResult(
                expired_cleaned=expired_cleaned,
                excess_completed_cleaned=excess_completed_cleaned,
                excess_failed_cleaned=excess_failed_cleaned,
                total_cleaned=0,
                errors=errors,
                would_delete_count=len(tasks_to_delete),
            )
        )

    # 4. Perform deletion
    total_cleaned = 0
    if tasks_to_delete:
        delete_tasks = [task_provider.delete_task(task_id) for task_id in tasks_to_delete]
        results = await asyncio.gather(*delete_tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Success) and res.data:
                total_cleaned += 1
            else:
                task_id = list(tasks_to_delete)[i]
                error_msg = res.error.message if hasattr(res, "error") and res.error else str(res)
                errors.append(f"Failed to delete task {task_id}: {error_msg}")

    # Recalculate counts based on actual deletions
    # This is complex as we don't know which category succeeded.
    # The initial counts are based on intent.
    # For now, we return the intended breakdown and total actual deletions.

    final_result = CleanupResult(
        expired_cleaned=expired_cleaned,
        excess_completed_cleaned=excess_completed_cleaned,
        excess_failed_cleaned=excess_failed_cleaned,
        total_cleaned=total_cleaned,
        errors=errors,
        would_delete_count=len(tasks_to_delete),
    )

    return create_a2a_success(final_result)


class TaskCleanupScheduler:
    """
    Task cleanup scheduler for running periodic cleanup operations
    """

    def __init__(
        self,
        task_provider: A2ATaskProvider,
        config: A2ATaskCleanupConfig = default_cleanup_config,
    ):
        self.task_provider = task_provider
        self.config = config
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the cleanup scheduler"""
        if self._running or not self.config.enabled:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())

    async def stop(self) -> None:
        """Stop the cleanup scheduler"""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run_once(self) -> A2AResult[CleanupResult]:
        """Run cleanup once manually"""
        return await perform_task_cleanup(self.task_provider, self.config)

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running"""
        return self._running

    async def _run_scheduler(self) -> None:
        """Internal scheduler loop"""
        while self._running:
            try:
                await self._run_cleanup()
                await asyncio.sleep(self.config.interval)
            except asyncio.CancelledError:
                break
            except Exception as error:
                print(f"A2A task cleanup scheduler error: {error!s}")
                # Avoid rapid failure loops
                await asyncio.sleep(self.config.interval)

    async def _run_cleanup(self) -> None:
        """Run a single cleanup operation"""
        try:
            result = await perform_task_cleanup(self.task_provider, self.config)

            if result.data:
                cleanup_result = result.data
                if cleanup_result.total_cleaned > 0 or cleanup_result.errors:
                    print(
                        f"A2A task cleanup completed: {cleanup_result.total_cleaned} tasks cleaned"
                    )
                    if cleanup_result.errors:
                        print(f"A2A task cleanup errors: {', '.join(cleanup_result.errors)}")
            elif result.error:
                print(f"A2A task cleanup failed: {result.error.message}")

        except Exception as error:
            print(f"A2A task cleanup error: {error!s}")


def create_task_cleanup_scheduler(
    task_provider: A2ATaskProvider,
    config: A2ATaskCleanupConfig = default_cleanup_config,
) -> TaskCleanupScheduler:
    """
    Factory function to create a task cleanup scheduler
    """
    return TaskCleanupScheduler(task_provider, config)


def create_cleanup_config_from_env() -> A2ATaskCleanupConfig:
    """
    Helper function to create cleanup config from environment variables
    """
    retain_states_str = os.getenv("JAF_A2A_CLEANUP_RETAIN_STATES", "working,submitted")
    retain_states = [s.strip() for s in retain_states_str.split(",") if s.strip()]

    return A2ATaskCleanupConfig(
        enabled=os.getenv("JAF_A2A_CLEANUP_ENABLED", "true").lower() == "true",
        interval=int(os.getenv("JAF_A2A_CLEANUP_INTERVAL", "3600")),
        max_age=int(os.getenv("JAF_A2A_CLEANUP_MAX_AGE", str(7 * 24 * 60 * 60))),
        max_completed_tasks=int(os.getenv("JAF_A2A_CLEANUP_MAX_COMPLETED", "1000")),
        max_failed_tasks=int(os.getenv("JAF_A2A_CLEANUP_MAX_FAILED", "500")),
        retain_states=retain_states,
        batch_size=int(os.getenv("JAF_A2A_CLEANUP_BATCH_SIZE", "100")),
        dry_run=os.getenv("JAF_A2A_CLEANUP_DRY_RUN", "false").lower() == "true",
    )
