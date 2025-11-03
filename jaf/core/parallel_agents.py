"""
Parallel Agent Execution for JAF Framework.

This module provides functionality to execute multiple sub-agents in parallel groups,
allowing for coordinated parallel execution with configurable grouping and result aggregation.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar

from .types import (
    Agent,
    Tool,
    ToolSchema,
    ToolSource,
    RunConfig,
    RunState,
    RunResult,
    Message,
    ContentRole,
    generate_run_id,
    generate_trace_id,
)
from .agent_tool import create_agent_tool, AgentToolInput

Ctx = TypeVar("Ctx")
Out = TypeVar("Out")


@dataclass
class ParallelAgentGroup:
    """Configuration for a group of agents to be executed in parallel."""

    name: str
    agents: List[Agent[Ctx, Out]]
    shared_input: bool = True  # Whether all agents receive the same input
    result_aggregation: str = "combine"  # "combine", "first", "majority", "custom"
    custom_aggregator: Optional[Callable[[List[str]], str]] = None
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel agent execution."""

    groups: List[ParallelAgentGroup]
    inter_group_execution: str = "sequential"  # "sequential" or "parallel"
    global_timeout: Optional[float] = None
    preserve_session: bool = False


class ParallelAgentsTool:
    """Tool that executes multiple agent groups in parallel."""

    def __init__(
        self,
        config: ParallelExecutionConfig,
        tool_name: str = "execute_parallel_agents",
        tool_description: str = "Execute multiple agents in parallel groups",
    ):
        self.config = config
        self.tool_name = tool_name
        self.tool_description = tool_description

        # Create tool schema
        self.schema = ToolSchema(
            name=tool_name,
            description=tool_description,
            parameters=AgentToolInput,
            timeout=config.global_timeout,
        )
        self.source = ToolSource.NATIVE
        self.metadata = {"source": "parallel_agents", "groups": len(config.groups)}

    async def execute(self, args: AgentToolInput, context: Ctx) -> str:
        """Execute all configured agent groups."""
        try:
            if self.config.inter_group_execution == "parallel":
                # Execute all groups in parallel
                group_results = await asyncio.gather(
                    *[
                        self._execute_group(group, args.input, context)
                        for group in self.config.groups
                    ]
                )
            else:
                # Execute groups sequentially
                group_results = []
                for group in self.config.groups:
                    result = await self._execute_group(group, args.input, context)
                    group_results.append(result)

            # Combine results from all groups
            final_result = {
                "parallel_execution_results": {
                    group.name: result for group, result in zip(self.config.groups, group_results)
                },
                "execution_mode": self.config.inter_group_execution,
                "total_groups": len(self.config.groups),
            }

            return json.dumps(final_result, indent=2)

        except Exception as e:
            return json.dumps(
                {
                    "error": "parallel_execution_failed",
                    "message": f"Failed to execute parallel agents: {str(e)}",
                    "groups_attempted": len(self.config.groups),
                }
            )

    async def _execute_group(
        self, group: ParallelAgentGroup, input_text: str, context: Ctx
    ) -> Dict[str, Any]:
        """Execute a single group of agents in parallel."""
        try:
            # Create agent tools for all agents in the group
            agent_tools = []
            for agent in group.agents:
                tool = create_agent_tool(
                    agent=agent,
                    tool_name=f"run_{agent.name.lower().replace(' ', '_')}",
                    tool_description=f"Execute the {agent.name} agent",
                    timeout=group.timeout,
                    preserve_session=self.config.preserve_session,
                )
                agent_tools.append((agent.name, tool))

            # Execute all agents in the group in parallel
            if group.shared_input:
                # All agents get the same input
                tasks = [
                    tool.execute(AgentToolInput(input=input_text), context)
                    for _, tool in agent_tools
                ]
            else:
                # This could be extended to support different inputs per agent
                tasks = [
                    tool.execute(AgentToolInput(input=input_text), context)
                    for _, tool in agent_tools
                ]

            # Execute with timeout if specified
            if group.timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=group.timeout
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            agent_results = {}
            for (agent_name, _), result in zip(agent_tools, results):
                if isinstance(result, Exception):
                    agent_results[agent_name] = {
                        "error": True,
                        "message": str(result),
                        "type": type(result).__name__,
                    }
                else:
                    agent_results[agent_name] = {"success": True, "result": result}

            # Apply result aggregation
            aggregated_result = self._aggregate_results(group, agent_results)

            return {
                "group_name": group.name,
                "agent_count": len(group.agents),
                "individual_results": agent_results,
                "aggregated_result": aggregated_result,
                "execution_time_ms": None,  # Could be added with timing
            }

        except asyncio.TimeoutError:
            return {
                "group_name": group.name,
                "error": "timeout",
                "message": f"Group {group.name} execution timed out after {group.timeout} seconds",
                "agent_count": len(group.agents),
            }
        except Exception as e:
            return {
                "group_name": group.name,
                "error": "execution_failed",
                "message": str(e),
                "agent_count": len(group.agents),
            }

    def _aggregate_results(
        self, group: ParallelAgentGroup, agent_results: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]:
        """Aggregate results from parallel agent execution."""
        successful_results = [
            result["result"]
            for result in agent_results.values()
            if result.get("success") and "result" in result
        ]

        if not successful_results:
            return {"error": "no_successful_results", "message": "All agents failed"}

        if group.result_aggregation == "first":
            return successful_results[0]
        elif group.result_aggregation == "combine":
            return {"combined_results": successful_results, "result_count": len(successful_results)}
        elif group.result_aggregation == "majority":
            # Simple majority logic - could be enhanced
            if len(successful_results) >= len(group.agents) // 2 + 1:
                return successful_results[0]  # Return first as majority representative
            else:
                return {"error": "no_majority", "results": successful_results}
        elif group.result_aggregation == "custom" and group.custom_aggregator:
            try:
                return group.custom_aggregator(successful_results)
            except Exception as e:
                return {"error": "custom_aggregation_failed", "message": str(e)}
        else:
            return {"combined_results": successful_results}


def create_parallel_agents_tool(
    groups: List[ParallelAgentGroup],
    tool_name: str = "execute_parallel_agents",
    tool_description: str = "Execute multiple agents in parallel groups",
    inter_group_execution: str = "sequential",
    global_timeout: Optional[float] = None,
    preserve_session: bool = False,
) -> Tool:
    """
    Create a tool that executes multiple agent groups in parallel.

    Args:
        groups: List of parallel agent groups to execute
        tool_name: Name of the tool
        tool_description: Description of the tool
        inter_group_execution: How to execute groups ("sequential" or "parallel")
        global_timeout: Global timeout for all executions
        preserve_session: Whether to preserve session across agent calls

    Returns:
        A Tool that can execute parallel agent groups
    """
    config = ParallelExecutionConfig(
        groups=groups,
        inter_group_execution=inter_group_execution,
        global_timeout=global_timeout,
        preserve_session=preserve_session,
    )

    return ParallelAgentsTool(config, tool_name, tool_description)


def create_simple_parallel_tool(
    agents: List[Agent],
    group_name: str = "parallel_group",
    tool_name: str = "execute_parallel_agents",
    shared_input: bool = True,
    result_aggregation: str = "combine",
    timeout: Optional[float] = None,
) -> Tool:
    """
    Create a simple parallel agents tool from a list of agents.

    Args:
        agents: List of agents to execute in parallel
        group_name: Name for the parallel group
        tool_name: Name of the tool
        shared_input: Whether all agents receive the same input
        result_aggregation: How to aggregate results ("combine", "first", "majority")
        timeout: Timeout for parallel execution

    Returns:
        A Tool that executes all agents in parallel
    """
    group = ParallelAgentGroup(
        name=group_name,
        agents=agents,
        shared_input=shared_input,
        result_aggregation=result_aggregation,
        timeout=timeout,
    )

    return create_parallel_agents_tool([group], tool_name=tool_name)


# Convenience functions for common parallel execution patterns


def create_language_specialists_tool(
    language_agents: Dict[str, Agent],
    tool_name: str = "consult_language_specialists",
    timeout: Optional[float] = 300.0,
) -> Tool:
    """Create a tool that consults multiple language specialists in parallel."""
    group = ParallelAgentGroup(
        name="language_specialists",
        agents=list(language_agents.values()),
        shared_input=True,
        result_aggregation="combine",
        timeout=timeout,
        metadata={"languages": list(language_agents.keys())},
    )

    return create_parallel_agents_tool(
        [group],
        tool_name=tool_name,
        tool_description="Consult multiple language specialists in parallel",
    )


def create_domain_experts_tool(
    expert_agents: Dict[str, Agent],
    tool_name: str = "consult_domain_experts",
    result_aggregation: str = "combine",
    timeout: Optional[float] = 60.0,
) -> Tool:
    """Create a tool that consults multiple domain experts in parallel."""
    group = ParallelAgentGroup(
        name="domain_experts",
        agents=list(expert_agents.values()),
        shared_input=True,
        result_aggregation=result_aggregation,
        timeout=timeout,
        metadata={"domains": list(expert_agents.keys())},
    )

    return create_parallel_agents_tool(
        [group], tool_name=tool_name, tool_description="Consult multiple domain experts in parallel"
    )
