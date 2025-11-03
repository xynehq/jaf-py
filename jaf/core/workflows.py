"""
Advanced workflow orchestration system for JAF framework.

This module provides sophisticated workflow capabilities including
conditional execution, parallel processing, error recovery, and
dynamic workflow adaptation based on runtime conditions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from enum import Enum
from abc import ABC, abstractmethod

from .types import Message, ContentRole, RunState, Agent, Tool
from .performance import PerformanceMonitor
from .analytics import global_analytics_engine


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Status of individual workflow steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass(frozen=True)
class WorkflowContext:
    """Context passed through workflow execution."""

    workflow_id: str
    user_context: Any
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_variable(self, key: str, value: Any) -> "WorkflowContext":
        """Create new context with additional variable."""
        new_vars = {**self.variables, key: value}
        return WorkflowContext(
            workflow_id=self.workflow_id,
            user_context=self.user_context,
            variables=new_vars,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> "WorkflowContext":
        """Create new context with additional metadata."""
        new_metadata = {**self.metadata, key: value}
        return WorkflowContext(
            workflow_id=self.workflow_id,
            user_context=self.user_context,
            variables=self.variables,
            metadata=new_metadata,
        )


@dataclass(frozen=True)
class StepResult:
    """Result of a workflow step execution."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """Check if step failed."""
        return self.status == StepStatus.FAILED


@dataclass(frozen=True)
class WorkflowResult:
    """Result of complete workflow execution."""

    workflow_id: str
    status: WorkflowStatus
    steps: List[StepResult]
    final_output: Any = None
    error: Optional[str] = None
    total_execution_time_ms: float = 0
    context: Optional[WorkflowContext] = None

    @property
    def is_success(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def failed_steps(self) -> List[StepResult]:
        """Get list of failed steps."""
        return [step for step in self.steps if step.is_failure]

    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps."""
        if not self.steps:
            return 0.0
        successful = len([s for s in self.steps if s.is_success])
        return (successful / len(self.steps)) * 100


class WorkflowStep(ABC):
    """Abstract base class for workflow steps."""

    def __init__(self, step_id: str, name: str, description: str = ""):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.retry_count = 0
        self.max_retries = 3
        self.timeout_seconds = 30
        self.conditions: List[Callable[[WorkflowContext], bool]] = []

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute the workflow step."""
        pass

    def add_condition(self, condition: Callable[[WorkflowContext], bool]) -> "WorkflowStep":
        """Add execution condition."""
        self.conditions.append(condition)
        return self

    def with_retry(self, max_retries: int) -> "WorkflowStep":
        """Configure retry behavior."""
        self.max_retries = max_retries
        return self

    def with_timeout(self, timeout_seconds: int) -> "WorkflowStep":
        """Configure timeout."""
        self.timeout_seconds = timeout_seconds
        return self

    def should_execute(self, context: WorkflowContext) -> bool:
        """Check if step should execute based on conditions."""
        return all(condition(context) for condition in self.conditions)


class AgentStep(WorkflowStep):
    """Workflow step that executes an agent."""

    def __init__(self, step_id: str, agent: Agent, message: str, name: str = ""):
        super().__init__(step_id, name or f"Agent: {agent.name}")
        self.agent = agent
        self.message = message

    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute agent step."""
        start_time = time.time()

        try:
            # Create run state for agent
            from .types import RunState, create_run_id, create_trace_id

            run_state = RunState(
                run_id=create_run_id(f"{context.workflow_id}_{self.step_id}"),
                trace_id=create_trace_id(f"{context.workflow_id}"),
                messages=[Message(role=ContentRole.USER, content=self.message)],
                current_agent_name=self.agent.name,
                context=context.user_context,
                turn_count=0,
            )

            # Get agent instructions
            instructions = self.agent.instructions(run_state)

            # Simulate agent execution (in real implementation, use JAF engine)
            await asyncio.sleep(0.1)  # Simulate processing

            execution_time = (time.time() - start_time) * 1000

            return StepResult(
                step_id=self.step_id,
                status=StepStatus.COMPLETED,
                output=f"Agent {self.agent.name} processed: {self.message}",
                execution_time_ms=execution_time,
                metadata={"agent_name": self.agent.name, "instructions": instructions},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
            )


class ToolStep(WorkflowStep):
    """Workflow step that executes a tool."""

    def __init__(self, step_id: str, tool: Tool, args: Any, name: str = ""):
        super().__init__(step_id, name or f"Tool: {tool.schema.name}")
        self.tool = tool
        self.args = args

    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute tool step."""
        start_time = time.time()

        try:
            # Execute tool
            result = await self.tool.execute(self.args, context.user_context)

            execution_time = (time.time() - start_time) * 1000

            return StepResult(
                step_id=self.step_id,
                status=StepStatus.COMPLETED,
                output=result,
                execution_time_ms=execution_time,
                metadata={"tool_name": self.tool.schema.name},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
            )


class ConditionalStep(WorkflowStep):
    """Workflow step that executes conditionally."""

    def __init__(
        self,
        step_id: str,
        condition: Callable[[WorkflowContext], bool],
        true_step: WorkflowStep,
        false_step: Optional[WorkflowStep] = None,
    ):
        super().__init__(step_id, "Conditional Step")
        self.condition = condition
        self.true_step = true_step
        self.false_step = false_step

    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute conditional step."""
        start_time = time.time()

        try:
            if self.condition(context):
                result = await self.true_step.execute(context)
            elif self.false_step:
                result = await self.false_step.execute(context)
            else:
                execution_time = (time.time() - start_time) * 1000
                return StepResult(
                    step_id=self.step_id,
                    status=StepStatus.SKIPPED,
                    output="Condition not met, no alternative step",
                    execution_time_ms=execution_time,
                )

            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=result.status,
                output=result.output,
                error=result.error,
                execution_time_ms=execution_time,
                metadata={"delegated_to": result.step_id},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
            )


class ParallelStep(WorkflowStep):
    """Workflow step that executes multiple steps in parallel."""

    def __init__(
        self, step_id: str, steps: List[WorkflowStep], wait_for_all: bool = True, name: str = ""
    ):
        super().__init__(step_id, name or "Parallel Execution")
        self.steps = steps
        self.wait_for_all = wait_for_all

    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute parallel steps."""
        start_time = time.time()

        try:
            # Execute all steps in parallel
            tasks = [step.execute(context) for step in self.steps]

            if self.wait_for_all:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Return as soon as first completes successfully
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                results = [await task for task in done]
                # Cancel pending tasks
                for task in pending:
                    task.cancel()

            execution_time = (time.time() - start_time) * 1000

            # Process results
            step_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    step_results.append(
                        StepResult(
                            step_id=f"{self.step_id}_parallel_{i}",
                            status=StepStatus.FAILED,
                            error=str(result),
                            execution_time_ms=0,
                        )
                    )
                else:
                    step_results.append(result)

            # Determine overall status
            if all(r.is_success for r in step_results):
                status = StepStatus.COMPLETED
            elif any(r.is_success for r in step_results) and not self.wait_for_all:
                status = StepStatus.COMPLETED
            else:
                status = StepStatus.FAILED

            return StepResult(
                step_id=self.step_id,
                status=status,
                output=step_results,
                execution_time_ms=execution_time,
                metadata={"parallel_results": len(step_results)},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
            )


class LoopStep(WorkflowStep):
    """Workflow step that executes in a loop."""

    def __init__(
        self,
        step_id: str,
        step: WorkflowStep,
        condition: Callable[[WorkflowContext, int], bool],
        max_iterations: int = 10,
    ):
        super().__init__(step_id, "Loop Step")
        self.step = step
        self.condition = condition
        self.max_iterations = max_iterations

    async def execute(self, context: WorkflowContext) -> StepResult:
        """Execute loop step."""
        start_time = time.time()
        results = []
        iteration = 0

        try:
            while iteration < self.max_iterations and self.condition(context, iteration):
                result = await self.step.execute(context)
                results.append(result)

                # Update context with iteration results
                context = context.with_variable(
                    f"loop_{self.step_id}_iteration_{iteration}", result.output
                )

                if result.is_failure:
                    break

                iteration += 1

            execution_time = (time.time() - start_time) * 1000

            # Determine overall status
            if results and all(r.is_success for r in results):
                status = StepStatus.COMPLETED
            elif not results:
                status = StepStatus.SKIPPED
            else:
                status = StepStatus.FAILED

            return StepResult(
                step_id=self.step_id,
                status=status,
                output=results,
                execution_time_ms=execution_time,
                metadata={"iterations": iteration, "results_count": len(results)},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StepResult(
                step_id=self.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
            )


class Workflow:
    """Main workflow orchestrator."""

    def __init__(self, workflow_id: str, name: str, description: str = ""):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []
        self.error_handlers: Dict[str, Callable[[StepResult, WorkflowContext], WorkflowStep]] = {}
        self.on_step_complete: Optional[Callable[[StepResult, WorkflowContext], None]] = None
        self.on_workflow_complete: Optional[Callable[[WorkflowResult], None]] = None
        self.performance_monitor = PerformanceMonitor()

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """Add a step to the workflow."""
        self.steps.append(step)
        return self

    def add_error_handler(
        self, step_id: str, handler: Callable[[StepResult, WorkflowContext], WorkflowStep]
    ) -> "Workflow":
        """Add error handler for specific step."""
        self.error_handlers[step_id] = handler
        return self

    def on_step_completed(
        self, callback: Callable[[StepResult, WorkflowContext], None]
    ) -> "Workflow":
        """Set callback for step completion."""
        self.on_step_complete = callback
        return self

    def on_workflow_completed(self, callback: Callable[[WorkflowResult], None]) -> "Workflow":
        """Set callback for workflow completion."""
        self.on_workflow_complete = callback
        return self

    async def execute(self, context: WorkflowContext) -> WorkflowResult:
        """Execute the complete workflow."""
        start_time = time.time()
        self.performance_monitor.start_monitoring()

        step_results: List[StepResult] = []
        current_context = context

        try:
            for step in self.steps:
                # Check if step should execute
                if not step.should_execute(current_context):
                    step_result = StepResult(
                        step_id=step.step_id,
                        status=StepStatus.SKIPPED,
                        output="Conditions not met",
                        execution_time_ms=0,
                    )
                    step_results.append(step_result)
                    continue

                # Execute step with retry logic
                step_result = await self._execute_step_with_retry(step, current_context)
                step_results.append(step_result)

                # Handle step completion
                if self.on_step_complete:
                    self.on_step_complete(step_result, current_context)

                # Update context with step result
                current_context = current_context.with_variable(
                    f"step_{step.step_id}_result", step_result.output
                )

                # Handle errors
                if step_result.is_failure and step.step_id in self.error_handlers:
                    error_handler = self.error_handlers[step.step_id]
                    recovery_step = error_handler(step_result, current_context)
                    recovery_result = await self._execute_step_with_retry(
                        recovery_step, current_context
                    )
                    step_results.append(recovery_result)

                    if recovery_result.is_failure:
                        # Recovery failed, stop workflow
                        break
                elif step_result.is_failure:
                    # No error handler, stop workflow
                    break

            # Calculate final status
            if all(r.is_success or r.status == StepStatus.SKIPPED for r in step_results):
                final_status = WorkflowStatus.COMPLETED
            else:
                final_status = WorkflowStatus.FAILED

            execution_time = (time.time() - start_time) * 1000
            performance_metrics = self.performance_monitor.stop_monitoring()

            # Get final output from last successful step
            final_output = None
            for result in reversed(step_results):
                if result.is_success and result.output is not None:
                    final_output = result.output
                    break

            workflow_result = WorkflowResult(
                workflow_id=self.workflow_id,
                status=final_status,
                steps=step_results,
                final_output=final_output,
                total_execution_time_ms=execution_time,
                context=current_context,
            )

            # Record analytics
            global_analytics_engine.record_system_metrics(
                performance_metrics, f"workflow_{self.workflow_id}"
            )

            # Handle workflow completion
            if self.on_workflow_complete:
                self.on_workflow_complete(workflow_result)

            return workflow_result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                steps=step_results,
                error=str(e),
                total_execution_time_ms=execution_time,
                context=current_context,
            )

    async def _execute_step_with_retry(
        self, step: WorkflowStep, context: WorkflowContext
    ) -> StepResult:
        """Execute step with retry logic."""
        last_result = None

        for attempt in range(step.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(step.execute(context), timeout=step.timeout_seconds)

                if result.is_success:
                    return result

                last_result = result

                if attempt < step.max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

            except asyncio.TimeoutError:
                last_result = StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILED,
                    error=f"Step timed out after {step.timeout_seconds} seconds",
                    execution_time_ms=step.timeout_seconds * 1000,
                )
            except Exception as e:
                last_result = StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    execution_time_ms=0,
                )

        return last_result or StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            error="Unknown error during execution",
            execution_time_ms=0,
        )


class WorkflowBuilder:
    """Builder for creating workflows with fluent API."""

    def __init__(self, workflow_id: str, name: str):
        self.workflow = Workflow(workflow_id, name)

    def add_agent_step(self, step_id: str, agent: Agent, message: str) -> "WorkflowBuilder":
        """Add agent execution step."""
        step = AgentStep(step_id, agent, message)
        self.workflow.add_step(step)
        return self

    def add_tool_step(self, step_id: str, tool: Tool, args: Any) -> "WorkflowBuilder":
        """Add tool execution step."""
        step = ToolStep(step_id, tool, args)
        self.workflow.add_step(step)
        return self

    def add_conditional_step(
        self,
        step_id: str,
        condition: Callable[[WorkflowContext], bool],
        true_step: WorkflowStep,
        false_step: Optional[WorkflowStep] = None,
    ) -> "WorkflowBuilder":
        """Add conditional execution step."""
        step = ConditionalStep(step_id, condition, true_step, false_step)
        self.workflow.add_step(step)
        return self

    def add_parallel_step(
        self, step_id: str, steps: List[WorkflowStep], wait_for_all: bool = True
    ) -> "WorkflowBuilder":
        """Add parallel execution step."""
        step = ParallelStep(step_id, steps, wait_for_all)
        self.workflow.add_step(step)
        return self

    def add_loop_step(
        self,
        step_id: str,
        step: WorkflowStep,
        condition: Callable[[WorkflowContext, int], bool],
        max_iterations: int = 10,
    ) -> "WorkflowBuilder":
        """Add loop execution step."""
        loop_step = LoopStep(step_id, step, condition, max_iterations)
        self.workflow.add_step(loop_step)
        return self

    def with_error_handler(
        self, step_id: str, handler: Callable[[StepResult, WorkflowContext], WorkflowStep]
    ) -> "WorkflowBuilder":
        """Add error handler."""
        self.workflow.add_error_handler(step_id, handler)
        return self

    def with_step_callback(
        self, callback: Callable[[StepResult, WorkflowContext], None]
    ) -> "WorkflowBuilder":
        """Add step completion callback."""
        self.workflow.on_step_completed(callback)
        return self

    def with_completion_callback(
        self, callback: Callable[[WorkflowResult], None]
    ) -> "WorkflowBuilder":
        """Add workflow completion callback."""
        self.workflow.on_workflow_completed(callback)
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        return self.workflow


# Convenience functions for creating workflows
def create_workflow(workflow_id: str, name: str) -> WorkflowBuilder:
    """Create a new workflow builder."""
    return WorkflowBuilder(workflow_id, name)


def create_sequential_workflow(workflow_id: str, name: str, steps: List[WorkflowStep]) -> Workflow:
    """Create a simple sequential workflow."""
    workflow = Workflow(workflow_id, name)
    for step in steps:
        workflow.add_step(step)
    return workflow


def create_parallel_workflow(
    workflow_id: str, name: str, steps: List[WorkflowStep], wait_for_all: bool = True
) -> Workflow:
    """Create a workflow that executes all steps in parallel."""
    workflow = Workflow(workflow_id, name)
    parallel_step = ParallelStep("parallel_execution", steps, wait_for_all)
    workflow.add_step(parallel_step)
    return workflow


async def execute_workflow_stream(
    workflow: Workflow, context: WorkflowContext
) -> AsyncIterator[StepResult]:
    """Execute workflow and stream step results as they complete."""
    step_results: List[StepResult] = []
    current_context = context

    for step in workflow.steps:
        if not step.should_execute(current_context):
            step_result = StepResult(
                step_id=step.step_id,
                status=StepStatus.SKIPPED,
                output="Conditions not met",
                execution_time_ms=0,
            )
            step_results.append(step_result)
            yield step_result
            continue

        step_result = await workflow._execute_step_with_retry(step, current_context)
        step_results.append(step_result)
        yield step_result

        # Update context
        current_context = current_context.with_variable(
            f"step_{step.step_id}_result", step_result.output
        )

        if step_result.is_failure:
            break
