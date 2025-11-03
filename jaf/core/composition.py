"""
Tool composition utilities for JAF framework.

This module provides powerful composition patterns for creating complex tool behaviors
from simple, reusable components. It supports pipelines, conditional execution,
parallel processing, and higher-order tool transformations.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
    Awaitable,
    Protocol,
    runtime_checkable,
)
from functools import wraps
from enum import Enum

from .types import Tool, ToolSchema, ToolSource, ToolExecuteFunction
from .tool_results import ToolResult, ToolResponse


T = TypeVar("T")
Args = TypeVar("Args")
Ctx = TypeVar("Ctx")


class CompositionStrategy(str, Enum):
    """Strategies for tool composition."""

    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    RETRY = "retry"
    CACHE = "cache"
    RATE_LIMIT = "rate_limit"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class CompositionMetadata:
    """Metadata for composed tools."""

    strategy: CompositionStrategy
    component_tools: List[str]
    composition_id: str
    created_at: float = field(default_factory=lambda: __import__("time").time())


@runtime_checkable
class ToolTransformer(Protocol):
    """Protocol for tool transformers."""

    def transform(self, tool: Tool[Any, Any]) -> Tool[Any, Any]:
        """Transform a tool into an enhanced version."""
        ...


class ToolPipeline:
    """
    Creates a pipeline of tools that execute in sequence.

    Each tool's output becomes the input to the next tool in the pipeline.
    """

    def __init__(self, *tools: Tool[Any, Any], name: str = "pipeline"):
        self.tools = list(tools)
        self.name = name
        self.pipeline_id = f"pipeline_{id(self)}"

    def create_tool(self) -> Tool[Any, Any]:
        """Create a single tool that represents the entire pipeline."""

        class PipelineTool:
            def __init__(self, pipeline: ToolPipeline):
                self.pipeline = pipeline

            @property
            def schema(self) -> ToolSchema[Any]:
                # Use the first tool's schema as the base
                first_tool = self.pipeline.tools[0]
                return ToolSchema(
                    name=f"{self.pipeline.name}_pipeline",
                    description=f"Pipeline of {len(self.pipeline.tools)} tools: {', '.join(t.schema.name for t in self.pipeline.tools)}",
                    parameters=first_tool.schema.parameters,
                )

            async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
                """Execute the pipeline sequentially."""
                current_input = args
                results = []

                for i, tool in enumerate(self.pipeline.tools):
                    try:
                        result = await tool.execute(current_input, context)
                        results.append({"tool": tool.schema.name, "step": i + 1, "result": result})

                        # Parse result for next step
                        if isinstance(result, str):
                            try:
                                parsed = json.loads(result)
                                current_input = parsed
                            except json.JSONDecodeError:
                                current_input = {"input": result}
                        else:
                            current_input = result.data if hasattr(result, "data") else result

                    except Exception as e:
                        return ToolResponse.error(
                            code="pipeline_error",
                            message=f"Pipeline failed at step {i + 1} ({tool.schema.name}): {str(e)}",
                            details={"step": i + 1, "tool": tool.schema.name, "results": results},
                        )

                return ToolResponse.success(
                    data=current_input,
                    metadata={
                        "pipeline_id": self.pipeline.pipeline_id,
                        "steps_executed": len(results),
                        "step_results": results,
                    },
                )

        return PipelineTool(self)


class ParallelToolExecution:
    """
    Executes multiple tools in parallel and combines their results.
    """

    def __init__(
        self, *tools: Tool[Any, Any], name: str = "parallel", combine_strategy: str = "merge"
    ):
        self.tools = list(tools)
        self.name = name
        self.combine_strategy = combine_strategy
        self.execution_id = f"parallel_{id(self)}"

    def create_tool(self) -> Tool[Any, Any]:
        """Create a single tool that executes all tools in parallel."""

        class ParallelTool:
            def __init__(self, executor: ParallelToolExecution):
                self.executor = executor

            @property
            def schema(self) -> ToolSchema[Any]:
                tool_names = [t.schema.name for t in self.executor.tools]
                return ToolSchema(
                    name=f"{self.executor.name}_parallel",
                    description=f"Parallel execution of {len(self.executor.tools)} tools: {', '.join(tool_names)}",
                    parameters=self.executor.tools[0].schema.parameters
                    if self.executor.tools
                    else None,
                )

            async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
                """Execute all tools in parallel."""
                if not self.executor.tools:
                    return ToolResponse.error("no_tools", "No tools to execute")

                # Execute all tools concurrently
                tasks = [
                    self._execute_tool_with_metadata(tool, args, context)
                    for tool in self.executor.tools
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                successful_results = []
                failed_results = []

                for i, result in enumerate(results):
                    tool_name = self.executor.tools[i].schema.name

                    if isinstance(result, Exception):
                        failed_results.append({"tool": tool_name, "error": str(result), "index": i})
                    else:
                        successful_results.append({"tool": tool_name, "result": result, "index": i})

                # Combine results based on strategy
                combined_data = self._combine_results(successful_results)

                return ToolResponse.success(
                    data=combined_data,
                    metadata={
                        "execution_id": self.executor.execution_id,
                        "successful_count": len(successful_results),
                        "failed_count": len(failed_results),
                        "failed_tools": failed_results,
                        "combine_strategy": self.executor.combine_strategy,
                    },
                )

            async def _execute_tool_with_metadata(
                self, tool: Tool[Any, Any], args: Any, context: Any
            ) -> Dict[str, Any]:
                """Execute a single tool and wrap result with metadata."""
                start_time = __import__("time").time()
                try:
                    result = await tool.execute(args, context)
                    execution_time = (__import__("time").time() - start_time) * 1000

                    return {
                        "success": True,
                        "data": result,
                        "execution_time_ms": execution_time,
                        "tool_name": tool.schema.name,
                    }
                except Exception as e:
                    execution_time = (__import__("time").time() - start_time) * 1000
                    return {
                        "success": False,
                        "error": str(e),
                        "execution_time_ms": execution_time,
                        "tool_name": tool.schema.name,
                    }

            def _combine_results(self, results: List[Dict[str, Any]]) -> Any:
                """Combine results based on the configured strategy."""
                if not results:
                    return None

                if self.executor.combine_strategy == "merge":
                    # Merge all results into a single dictionary
                    combined = {}
                    for result in results:
                        tool_name = result["tool"]
                        combined[tool_name] = result["result"]
                    return combined

                elif self.executor.combine_strategy == "array":
                    # Return results as an array
                    return [result["result"] for result in results]

                elif self.executor.combine_strategy == "first":
                    # Return the first successful result
                    return results[0]["result"] if results else None

                elif self.executor.combine_strategy == "best":
                    # Return the result with the best execution time
                    best_result = min(
                        results, key=lambda r: r.get("execution_time_ms", float("inf"))
                    )
                    return best_result["result"]

                else:
                    # Default to merge strategy
                    return self._combine_results(results)

        return ParallelTool(self)


class ConditionalTool:
    """
    Creates a tool that conditionally executes different tools based on input.
    """

    def __init__(
        self,
        condition: Callable[[Any], bool],
        true_tool: Tool[Any, Any],
        false_tool: Tool[Any, Any],
        name: str = "conditional",
    ):
        self.condition = condition
        self.true_tool = true_tool
        self.false_tool = false_tool
        self.name = name

    def create_tool(self) -> Tool[Any, Any]:
        """Create a conditional tool."""

        class ConditionalToolImpl:
            def __init__(self, conditional: ConditionalTool):
                self.conditional = conditional

            @property
            def schema(self) -> ToolSchema[Any]:
                return ToolSchema(
                    name=f"{self.conditional.name}_conditional",
                    description=f"Conditional execution: {self.conditional.true_tool.schema.name} or {self.conditional.false_tool.schema.name}",
                    parameters=self.conditional.true_tool.schema.parameters,
                )

            async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
                """Execute the appropriate tool based on condition."""
                try:
                    condition_result = self.conditional.condition(args)
                    selected_tool = (
                        self.conditional.true_tool
                        if condition_result
                        else self.conditional.false_tool
                    )

                    result = await selected_tool.execute(args, context)

                    # Wrap result with conditional metadata
                    if isinstance(result, str):
                        return json.dumps(
                            {
                                "result": result,
                                "condition_met": condition_result,
                                "selected_tool": selected_tool.schema.name,
                            }
                        )
                    else:
                        return ToolResponse.success(
                            data=result.data if hasattr(result, "data") else result,
                            metadata={
                                "condition_met": condition_result,
                                "selected_tool": selected_tool.schema.name,
                                "conditional_name": self.conditional.name,
                            },
                        )

                except Exception as e:
                    return ToolResponse.error(
                        code="conditional_error",
                        message=f"Conditional tool execution failed: {str(e)}",
                        details={"conditional_name": self.conditional.name},
                    )

        return ConditionalToolImpl(self)


class RetryTool:
    """
    Wraps a tool with retry logic for handling transient failures.
    """

    def __init__(
        self,
        tool: Tool[Any, Any],
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ):
        self.tool = tool
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on or (lambda e: True)  # Retry on all exceptions by default

    def create_tool(self) -> Tool[Any, Any]:
        """Create a retry-enabled tool."""

        class RetryToolImpl:
            def __init__(self, retry_tool: RetryTool):
                self.retry_tool = retry_tool

            @property
            def schema(self) -> ToolSchema[Any]:
                return ToolSchema(
                    name=f"{self.retry_tool.tool.schema.name}_retry",
                    description=f"Retry-enabled: {self.retry_tool.tool.schema.description}",
                    parameters=self.retry_tool.tool.schema.parameters,
                )

            async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
                """Execute with retry logic."""
                last_exception = None

                for attempt in range(self.retry_tool.max_retries + 1):
                    try:
                        result = await self.retry_tool.tool.execute(args, context)

                        # Success - wrap with retry metadata
                        if isinstance(result, str):
                            return json.dumps(
                                {
                                    "result": result,
                                    "attempts": attempt + 1,
                                    "max_retries": self.retry_tool.max_retries,
                                }
                            )
                        else:
                            return ToolResponse.success(
                                data=result.data if hasattr(result, "data") else result,
                                metadata={
                                    "attempts": attempt + 1,
                                    "max_retries": self.retry_tool.max_retries,
                                    "original_tool": self.retry_tool.tool.schema.name,
                                },
                            )

                    except Exception as e:
                        last_exception = e

                        # Check if we should retry this exception
                        if not self.retry_tool.retry_on(e):
                            break

                        # Don't sleep on the last attempt
                        if attempt < self.retry_tool.max_retries:
                            sleep_time = self.retry_tool.backoff_factor * (2**attempt)
                            await asyncio.sleep(sleep_time)

                # All retries exhausted
                return ToolResponse.error(
                    code="retry_exhausted",
                    message=f"Tool failed after {self.retry_tool.max_retries + 1} attempts: {str(last_exception)}",
                    details={
                        "attempts": self.retry_tool.max_retries + 1,
                        "last_error": str(last_exception),
                        "original_tool": self.retry_tool.tool.schema.name,
                    },
                )

        return RetryToolImpl(self)


class CachedTool:
    """
    Wraps a tool with caching to avoid redundant executions.
    """

    def __init__(
        self,
        tool: Tool[Any, Any],
        cache_key_fn: Optional[Callable[[Any], str]] = None,
        ttl_seconds: Optional[float] = None,
    ):
        self.tool = tool
        self.cache_key_fn = cache_key_fn or (lambda args: str(hash(str(args))))
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def create_tool(self) -> Tool[Any, Any]:
        """Create a cached tool."""

        class CachedToolImpl:
            def __init__(self, cached_tool: CachedTool):
                self.cached_tool = cached_tool

            @property
            def schema(self) -> ToolSchema[Any]:
                return ToolSchema(
                    name=f"{self.cached_tool.tool.schema.name}_cached",
                    description=f"Cached: {self.cached_tool.tool.schema.description}",
                    parameters=self.cached_tool.tool.schema.parameters,
                )

            async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
                """Execute with caching."""
                cache_key = self.cached_tool.cache_key_fn(args)
                current_time = __import__("time").time()

                # Check cache
                if cache_key in self.cached_tool.cache:
                    cache_entry = self.cached_tool.cache[cache_key]

                    # Check TTL
                    if (
                        self.cached_tool.ttl_seconds is None
                        or current_time - cache_entry["timestamp"] < self.cached_tool.ttl_seconds
                    ):
                        # Cache hit
                        cached_result = cache_entry["result"]

                        if isinstance(cached_result, str):
                            return json.dumps(
                                {
                                    "result": cached_result,
                                    "cache_hit": True,
                                    "cached_at": cache_entry["timestamp"],
                                }
                            )
                        else:
                            return ToolResponse.success(
                                data=cached_result,
                                metadata={
                                    "cache_hit": True,
                                    "cached_at": cache_entry["timestamp"],
                                    "original_tool": self.cached_tool.tool.schema.name,
                                },
                            )

                # Cache miss - execute tool
                try:
                    result = await self.cached_tool.tool.execute(args, context)

                    # Store in cache
                    self.cached_tool.cache[cache_key] = {
                        "result": result,
                        "timestamp": current_time,
                    }

                    # Return with cache metadata
                    if isinstance(result, str):
                        return json.dumps(
                            {"result": result, "cache_hit": False, "cached_at": current_time}
                        )
                    else:
                        return ToolResponse.success(
                            data=result.data if hasattr(result, "data") else result,
                            metadata={
                                "cache_hit": False,
                                "cached_at": current_time,
                                "original_tool": self.cached_tool.tool.schema.name,
                            },
                        )

                except Exception as e:
                    return ToolResponse.error(
                        code="cached_tool_error",
                        message=f"Cached tool execution failed: {str(e)}",
                        details={"original_tool": self.cached_tool.tool.schema.name},
                    )

        return CachedToolImpl(self)


# Convenience functions for creating composed tools


def create_tool_pipeline(*tools: Tool[Any, Any], name: str = "pipeline") -> Tool[Any, Any]:
    """Create a pipeline of tools that execute in sequence."""
    return ToolPipeline(*tools, name=name).create_tool()


def create_parallel_tools(
    *tools: Tool[Any, Any], name: str = "parallel", combine_strategy: str = "merge"
) -> Tool[Any, Any]:
    """Create a tool that executes multiple tools in parallel."""
    return ParallelToolExecution(*tools, name=name, combine_strategy=combine_strategy).create_tool()


def create_conditional_tool(
    condition: Callable[[Any], bool],
    true_tool: Tool[Any, Any],
    false_tool: Tool[Any, Any],
    name: str = "conditional",
) -> Tool[Any, Any]:
    """Create a tool that conditionally executes different tools."""
    return ConditionalTool(condition, true_tool, false_tool, name).create_tool()


def with_retry(
    tool: Tool[Any, Any],
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retry_on: Optional[Callable[[Exception], bool]] = None,
) -> Tool[Any, Any]:
    """Wrap a tool with retry logic."""
    return RetryTool(tool, max_retries, backoff_factor, retry_on).create_tool()


def with_cache(
    tool: Tool[Any, Any],
    cache_key_fn: Optional[Callable[[Any], str]] = None,
    ttl_seconds: Optional[float] = None,
) -> Tool[Any, Any]:
    """Wrap a tool with caching."""
    return CachedTool(tool, cache_key_fn, ttl_seconds).create_tool()


def with_timeout(tool: Tool[Any, Any], timeout_seconds: float) -> Tool[Any, Any]:
    """Wrap a tool with timeout protection."""

    class TimeoutTool:
        @property
        def schema(self) -> ToolSchema[Any]:
            return ToolSchema(
                name=f"{tool.schema.name}_timeout",
                description=f"Timeout-protected: {tool.schema.description}",
                parameters=tool.schema.parameters,
            )

        async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
            """Execute with timeout protection."""
            try:
                result = await asyncio.wait_for(
                    tool.execute(args, context), timeout=timeout_seconds
                )

                if isinstance(result, str):
                    return json.dumps(
                        {"result": result, "timeout_seconds": timeout_seconds, "timed_out": False}
                    )
                else:
                    return ToolResponse.success(
                        data=result.data if hasattr(result, "data") else result,
                        metadata={
                            "timeout_seconds": timeout_seconds,
                            "timed_out": False,
                            "original_tool": tool.schema.name,
                        },
                    )

            except asyncio.TimeoutError:
                return ToolResponse.error(
                    code="timeout_error",
                    message=f"Tool execution timed out after {timeout_seconds} seconds",
                    details={"timeout_seconds": timeout_seconds, "original_tool": tool.schema.name},
                )

    return TimeoutTool()


# Higher-order composition functions


def compose_tools(*transformers: ToolTransformer) -> Callable[[Tool[Any, Any]], Tool[Any, Any]]:
    """Compose multiple tool transformers into a single transformation."""

    def composed_transform(tool: Tool[Any, Any]) -> Tool[Any, Any]:
        result = tool
        for transformer in transformers:
            result = transformer.transform(result)
        return result

    return composed_transform


class ToolComposer:
    """
    Builder class for creating complex tool compositions.
    """

    def __init__(self, base_tool: Tool[Any, Any]):
        self.tool = base_tool
        self.composition_history: List[str] = []

    def with_retry(self, max_retries: int = 3, backoff_factor: float = 1.0) -> "ToolComposer":
        """Add retry capability."""
        self.tool = with_retry(self.tool, max_retries, backoff_factor)
        self.composition_history.append(f"retry(max={max_retries}, backoff={backoff_factor})")
        return self

    def with_cache(self, ttl_seconds: Optional[float] = None) -> "ToolComposer":
        """Add caching capability."""
        self.tool = with_cache(self.tool, ttl_seconds=ttl_seconds)
        self.composition_history.append(f"cache(ttl={ttl_seconds})")
        return self

    def with_timeout(self, timeout_seconds: float) -> "ToolComposer":
        """Add timeout protection."""
        self.tool = with_timeout(self.tool, timeout_seconds)
        self.composition_history.append(f"timeout({timeout_seconds}s)")
        return self

    def build(self) -> Tool[Any, Any]:
        """Build the final composed tool."""
        return self.tool

    def get_composition_info(self) -> Dict[str, Any]:
        """Get information about the composition."""
        return {
            "base_tool": self.tool.schema.name,
            "transformations": self.composition_history,
            "final_name": self.tool.schema.name,
        }


def compose(base_tool: Tool[Any, Any]) -> ToolComposer:
    """Start a tool composition chain."""
    return ToolComposer(base_tool)
