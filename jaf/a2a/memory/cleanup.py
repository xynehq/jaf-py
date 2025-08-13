"""
A2A Task Cleanup Service for JAF

This module provides pure functional cleanup and expiration policies for A2A tasks.
It includes configurable cleanup policies, scheduling utilities, and batch processing
for maintaining optimal storage performance.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .types import (
    A2AResult,
    A2ATaskProvider,
    create_a2a_failure,
    create_a2a_success,
    create_a2a_task_storage_error,
)


@dataclass(frozen=True)
class A2ATaskCleanupConfig:
    """Configuration for task cleanup policies"""
    enabled: bool = True
    interval: int = 3600  # Cleanup interval in seconds (1 hour)
    max_age: int = 7 * 24 * 60 * 60  # Maximum age of completed tasks in seconds (7 days)
    max_completed_tasks: int = 1000  # Maximum number of completed tasks to keep
    max_failed_tasks: int = 500  # Maximum number of failed tasks to keep
    retain_states: List[str] = None  # Task states to always retain
    batch_size: int = 100  # Number of tasks to process in each cleanup batch
    dry_run: bool = False  # If True, log what would be cleaned up but don't delete

    def __post_init__(self):
        if self.retain_states is None:
            object.__setattr__(self, 'retain_states', ['working', 'input-required', 'submitted'])

@dataclass(frozen=True)
class CleanupResult:
    """Result type for cleanup operations"""
    expired_cleaned: int
    excess_completed_cleaned: int
    excess_failed_cleaned: int
    total_cleaned: int
    errors: List[str]

# Default cleanup configuration
default_cleanup_config = A2ATaskCleanupConfig()

async def perform_task_cleanup(
    task_provider: A2ATaskProvider,
    config: A2ATaskCleanupConfig = default_cleanup_config
) -> A2AResult[CleanupResult]:
    """
    Pure function to perform task cleanup
    
    Args:
        task_provider: The A2A task provider to clean up
        config: Cleanup configuration
        
    Returns:
        A2AResult containing cleanup results or an error
    """
    try:
        errors: List[str] = []
        expired_cleaned = 0
        excess_completed_cleaned = 0
        excess_failed_cleaned = 0

        # Step 1: Clean up expired tasks
        if config.enabled:
            try:
                expired_result = await task_provider.cleanup_expired_tasks()
                if hasattr(expired_result, 'data') and isinstance(expired_result.data, int):
                    expired_cleaned = expired_result.data
                    if config.dry_run:
                        print(f"[DRY RUN] Would clean up {expired_cleaned} expired tasks")
                        expired_cleaned = 0  # Reset for dry run
                elif hasattr(expired_result, 'error'):
                    errors.append(f"Failed to cleanup expired tasks: {expired_result.error.message}")
                else:
                    errors.append("Failed to cleanup expired tasks: unexpected result type")
            except Exception as error:
                errors.append(f"Error during expired task cleanup: {error!s}")

        # Step 2: Clean up excess completed tasks
        if config.enabled and config.max_completed_tasks > 0:
            try:
                from .types import A2ATaskQuery
                completed_tasks_result = await task_provider.find_tasks(
                    A2ATaskQuery(
                        state="completed",
                        limit=config.max_completed_tasks + config.batch_size
                    )
                )

                if hasattr(completed_tasks_result, 'data') and isinstance(completed_tasks_result.data, list):
                    completed_tasks = completed_tasks_result.data

                    if len(completed_tasks) > config.max_completed_tasks:
                        # Sort by completion time (oldest first) and remove excess
                        sorted_tasks = sorted(
                            completed_tasks,
                            key=lambda t: t.status.timestamp or "1970-01-01T00:00:00Z"
                        )
                        tasks_to_delete = sorted_tasks[:-config.max_completed_tasks]

                        if config.dry_run:
                            print(f"[DRY RUN] Would clean up {len(tasks_to_delete)} excess completed tasks")
                        else:
                            for task in tasks_to_delete:
                                delete_result = await task_provider.delete_task(task.id)
                                if isinstance(delete_result.data, bool) and delete_result.data:
                                    excess_completed_cleaned += 1
                                else:
                                    errors.append(f"Failed to delete completed task {task.id}")
                else:
                    errors.append(f"Failed to find completed tasks: {completed_tasks_result.error.message}")
            except Exception as error:
                errors.append(f"Error during completed task cleanup: {error!s}")

        # Step 3: Clean up excess failed tasks
        if config.enabled and config.max_failed_tasks > 0:
            try:
                from .types import A2ATaskQuery
                failed_tasks_result = await task_provider.find_tasks(
                    A2ATaskQuery(
                        state="failed",
                        limit=config.max_failed_tasks + config.batch_size
                    )
                )

                if hasattr(failed_tasks_result, 'data') and isinstance(failed_tasks_result.data, list):
                    failed_tasks = failed_tasks_result.data

                    if len(failed_tasks) > config.max_failed_tasks:
                        # Sort by failure time (oldest first) and remove excess
                        sorted_tasks = sorted(
                            failed_tasks,
                            key=lambda t: t.status.timestamp or "1970-01-01T00:00:00Z"
                        )
                        tasks_to_delete = sorted_tasks[:-config.max_failed_tasks]

                        if config.dry_run:
                            print(f"[DRY RUN] Would clean up {len(tasks_to_delete)} excess failed tasks")
                        else:
                            for task in tasks_to_delete:
                                delete_result = await task_provider.delete_task(task.id)
                                if isinstance(delete_result.data, bool) and delete_result.data:
                                    excess_failed_cleaned += 1
                                else:
                                    errors.append(f"Failed to delete failed task {task.id}")
                else:
                    errors.append(f"Failed to find failed tasks: {failed_tasks_result.error.message}")
            except Exception as error:
                errors.append(f"Error during failed task cleanup: {error!s}")

        # Step 4: Clean up old tasks beyond max age
        if config.enabled and config.max_age > 0:
            try:
                cutoff_date = datetime.now() - timedelta(seconds=config.max_age)

                # Clean up old completed and failed tasks
                for state in ['completed', 'failed', 'canceled']:
                    if state in config.retain_states:
                        continue

                    from .types import A2ATaskQuery
                    old_tasks_result = await task_provider.find_tasks(
                        A2ATaskQuery(
                            state=state,
                            until=cutoff_date,
                            limit=config.batch_size
                        )
                    )

                    if hasattr(old_tasks_result, 'data') and isinstance(old_tasks_result.data, list):
                        old_tasks = old_tasks_result.data

                        if config.dry_run:
                            print(f"[DRY RUN] Would clean up {len(old_tasks)} old {state} tasks")
                        else:
                            for task in old_tasks:
                                delete_result = await task_provider.delete_task(task.id)
                                if isinstance(delete_result.data, bool) and delete_result.data:
                                    if state == 'completed':
                                        excess_completed_cleaned += 1
                                    elif state == 'failed':
                                        excess_failed_cleaned += 1
                                else:
                                    errors.append(f"Failed to delete old {state} task {task.id}")
                    else:
                        errors.append(f"Failed to find old {state} tasks: {old_tasks_result.error.message}")
            except Exception as error:
                errors.append(f"Error during old task cleanup: {error!s}")

        total_cleaned = expired_cleaned + excess_completed_cleaned + excess_failed_cleaned

        return create_a2a_success(CleanupResult(
            expired_cleaned=expired_cleaned,
            excess_completed_cleaned=excess_completed_cleaned,
            excess_failed_cleaned=excess_failed_cleaned,
            total_cleaned=total_cleaned,
            errors=errors
        ))

    except Exception as error:
        return create_a2a_failure(
            create_a2a_task_storage_error('cleanup', 'unknown', None, error)
        )

class TaskCleanupScheduler:
    """
    Task cleanup scheduler for running periodic cleanup operations
    """

    def __init__(
        self,
        task_provider: A2ATaskProvider,
        config: A2ATaskCleanupConfig = default_cleanup_config
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
        try:
            # Run initial cleanup
            await self._run_cleanup()

            # Schedule periodic cleanup
            while self._running:
                await asyncio.sleep(self.config.interval)
                if self._running:  # Check again after sleep
                    await self._run_cleanup()

        except asyncio.CancelledError:
            # Normal cancellation, exit gracefully
            pass
        except Exception as error:
            print(f"A2A task cleanup scheduler error: {error!s}")

    async def _run_cleanup(self) -> None:
        """Run a single cleanup operation"""
        try:
            result = await perform_task_cleanup(self.task_provider, self.config)

            if isinstance(result.data, CleanupResult):
                cleanup_result = result.data

                if cleanup_result.total_cleaned > 0 or cleanup_result.errors:
                    print(f"A2A task cleanup completed: {cleanup_result.total_cleaned} tasks cleaned")

                    if cleanup_result.errors:
                        print(f"A2A task cleanup errors: {', '.join(cleanup_result.errors)}")
            else:
                print(f"A2A task cleanup failed: {result.error.message}")

        except Exception as error:
            print(f"A2A task cleanup error: {error!s}")

def create_task_cleanup_scheduler(
    task_provider: A2ATaskProvider,
    config: A2ATaskCleanupConfig = default_cleanup_config
) -> TaskCleanupScheduler:
    """
    Factory function to create a task cleanup scheduler
    
    Args:
        task_provider: The A2A task provider to clean up
        config: Cleanup configuration
        
    Returns:
        TaskCleanupScheduler instance
    """
    return TaskCleanupScheduler(task_provider, config)

def validate_cleanup_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function to validate cleanup configuration
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary with 'valid' boolean and 'errors' list
    """
    errors: List[str] = []

    if 'interval' in config and config['interval'] <= 0:
        errors.append('Cleanup interval must be greater than 0')

    if 'max_age' in config and config['max_age'] <= 0:
        errors.append('Max age must be greater than 0')

    if 'max_completed_tasks' in config and config['max_completed_tasks'] < 0:
        errors.append('Max completed tasks must be non-negative')

    if 'max_failed_tasks' in config and config['max_failed_tasks'] < 0:
        errors.append('Max failed tasks must be non-negative')

    if 'batch_size' in config and config['batch_size'] <= 0:
        errors.append('Batch size must be greater than 0')

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def create_cleanup_config_from_env() -> A2ATaskCleanupConfig:
    """
    Helper function to create cleanup config from environment variables
    
    Returns:
        A2ATaskCleanupConfig created from environment variables
    """
    retain_states_str = os.getenv('JAF_A2A_CLEANUP_RETAIN_STATES', 'working,input-required,submitted')
    retain_states = retain_states_str.split(',') if retain_states_str else []

    return A2ATaskCleanupConfig(
        enabled=os.getenv('JAF_A2A_CLEANUP_ENABLED', 'true').lower() != 'false',
        interval=int(os.getenv('JAF_A2A_CLEANUP_INTERVAL', '3600')),
        max_age=int(os.getenv('JAF_A2A_CLEANUP_MAX_AGE', str(7 * 24 * 60 * 60))),  # 7 days
        max_completed_tasks=int(os.getenv('JAF_A2A_CLEANUP_MAX_COMPLETED', '1000')),
        max_failed_tasks=int(os.getenv('JAF_A2A_CLEANUP_MAX_FAILED', '500')),
        retain_states=retain_states,
        batch_size=int(os.getenv('JAF_A2A_CLEANUP_BATCH_SIZE', '100')),
        dry_run=os.getenv('JAF_A2A_CLEANUP_DRY_RUN', 'false').lower() == 'true'
    )
