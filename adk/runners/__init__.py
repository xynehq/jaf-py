"""
ADK Runners Module

Intelligent multi-agent coordination and execution strategies.

This module provides sophisticated multi-agent orchestration capabilities including:
- Intelligent agent selection based on keyword matching and context
- Advanced response merging for parallel execution
- Hierarchical coordination with delegation decision extraction
- Rule-based coordination with conditional/parallel/sequential actions
"""

from .multi_agent import (
    execute_multi_agent,
    select_best_agent,
    merge_parallel_responses,
    extract_delegation_decision,
    execute_with_coordination_rules,
    extract_keywords,
)
from .agent_runner import execute_agent, run_agent
from .types import (
    MultiAgentConfig,
    AgentConfig,
    CoordinationRule,
    DelegationStrategy,
    RunnerCallbacks,
    RunnerConfig,
    LLMControlResult,
    ToolSelectionControlResult,
    ToolExecutionControlResult,
    IterationControlResult,
    IterationCompleteResult,
    SynthesisCheckResult,
    FallbackCheckResult,
)

__all__ = [
    "execute_multi_agent",
    "select_best_agent",
    "merge_parallel_responses",
    "extract_delegation_decision",
    "execute_with_coordination_rules",
    "extract_keywords",
    "execute_agent",
    "run_agent",
    "MultiAgentConfig",
    "AgentConfig",
    "CoordinationRule",
    "DelegationStrategy",
    "RunnerCallbacks",
    "RunnerConfig",
    "LLMControlResult",
    "ToolSelectionControlResult",
    "ToolExecutionControlResult",
    "IterationControlResult",
    "IterationCompleteResult",
    "SynthesisCheckResult",
    "FallbackCheckResult",
]
