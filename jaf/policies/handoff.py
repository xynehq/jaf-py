"""
Handoff policies for the JAF framework.

This module provides validation and security for agent-to-agent handoffs,
ensuring controlled and secure agent transitions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.types import Guardrail, ValidationResult


@dataclass
class HandoffPolicy:
    """Policy configuration for agent handoffs."""
    allowed_handoffs: Dict[str, List[str]]  # source_agent -> [allowed_target_agents]
    require_permission: bool = True
    validate_context: bool = True
    max_handoff_depth: int = 10

def create_handoff_guardrail(
    policy: HandoffPolicy,
    current_agent: str,
    handoff_history: Optional[List[str]] = None
) -> Guardrail:
    """
    Create a guardrail that validates agent handoffs.
    
    Args:
        policy: Handoff policy configuration
        current_agent: Name of the current agent
        handoff_history: List of previous agents in the current session
        
    Returns:
        Guardrail function for handoff validation
    """
    handoff_history = handoff_history or []

    def handoff_guardrail(handoff_data: Any) -> ValidationResult:
        # Extract target agent from handoff data
        if isinstance(handoff_data, dict):
            target_agent = handoff_data.get('handoff_to') or handoff_data.get('target_agent')
        elif isinstance(handoff_data, str):
            # Assume the string is the target agent name
            target_agent = handoff_data
        else:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid handoff data format"
            )

        if not target_agent:
            return ValidationResult(
                is_valid=False,
                error_message="No target agent specified in handoff"
            )

        # Check permission
        if policy.require_permission:
            allowed_targets = policy.allowed_handoffs.get(current_agent, [])
            if target_agent not in allowed_targets:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Agent {current_agent} is not allowed to handoff to {target_agent}"
                )

        # Check handoff depth to prevent infinite loops
        if len(handoff_history) >= policy.max_handoff_depth:
            return ValidationResult(
                is_valid=False,
                error_message=f"Maximum handoff depth ({policy.max_handoff_depth}) exceeded"
            )

        # Check for immediate circular handoffs
        if handoff_history and handoff_history[-1] == target_agent:
            return ValidationResult(
                is_valid=False,
                error_message=f"Circular handoff detected: {current_agent} -> {target_agent} -> {current_agent}"
            )

        # Validate context if required
        if policy.validate_context and isinstance(handoff_data, dict):
            context = handoff_data.get('context', {})
            if not isinstance(context, dict):
                return ValidationResult(
                    is_valid=False,
                    error_message="Handoff context must be a dictionary"
                )

        return ValidationResult(is_valid=True)

    return handoff_guardrail

def validate_handoff_permissions(
    source_agent: str,
    target_agent: str,
    allowed_handoffs: Dict[str, List[str]]
) -> ValidationResult:
    """
    Validate if a handoff is allowed between two agents.
    
    Args:
        source_agent: Name of the source agent
        target_agent: Name of the target agent
        allowed_handoffs: Dictionary mapping source agents to allowed targets
        
    Returns:
        ValidationResult indicating if the handoff is allowed
    """
    allowed_targets = allowed_handoffs.get(source_agent, [])

    if target_agent not in allowed_targets:
        return ValidationResult(
            is_valid=False,
            error_message=f"Handoff from {source_agent} to {target_agent} not permitted"
        )

    return ValidationResult(is_valid=True)

def create_role_based_handoff_policy(
    agent_roles: Dict[str, str],  # agent_name -> role
    role_permissions: Dict[str, List[str]]  # role -> [allowed_target_roles]
) -> HandoffPolicy:
    """
    Create a handoff policy based on agent roles.
    
    Args:
        agent_roles: Mapping of agent names to their roles
        role_permissions: Mapping of roles to allowed target roles
        
    Returns:
        HandoffPolicy configured for role-based permissions
    """
    allowed_handoffs = {}

    for agent_name, agent_role in agent_roles.items():
        allowed_target_roles = role_permissions.get(agent_role, [])
        allowed_targets = [
            target_agent for target_agent, target_role in agent_roles.items()
            if target_role in allowed_target_roles and target_agent != agent_name
        ]
        allowed_handoffs[agent_name] = allowed_targets

    return HandoffPolicy(
        allowed_handoffs=allowed_handoffs,
        require_permission=True,
        validate_context=True
    )

def create_workflow_handoff_policy(
    workflow_steps: List[List[str]]  # List of workflow steps, each step is a list of agent names
) -> HandoffPolicy:
    """
    Create a handoff policy based on a defined workflow.
    
    Args:
        workflow_steps: List of workflow steps, where each step contains agent names
                       that can transition to agents in the next step
        
    Returns:
        HandoffPolicy configured for workflow-based transitions
    """
    allowed_handoffs = {}

    for i, current_step in enumerate(workflow_steps):
        # Agents in current step can handoff to agents in next step
        next_step = workflow_steps[i + 1] if i + 1 < len(workflow_steps) else []

        for agent in current_step:
            # Allow handoffs within the same step and to the next step
            allowed_targets = []

            # Same step handoffs (for parallel processing)
            allowed_targets.extend([a for a in current_step if a != agent])

            # Next step handoffs
            allowed_targets.extend(next_step)

            allowed_handoffs[agent] = allowed_targets

    return HandoffPolicy(
        allowed_handoffs=allowed_handoffs,
        require_permission=True,
        validate_context=True
    )

# Predefined policies for common scenarios
def create_hierarchical_handoff_policy(
    hierarchy: Dict[str, List[str]]  # supervisor -> [subordinates]
) -> HandoffPolicy:
    """Create a policy for hierarchical agent handoffs."""
    allowed_handoffs = {}

    # Supervisors can handoff to subordinates
    for supervisor, subordinates in hierarchy.items():
        allowed_handoffs[supervisor] = subordinates.copy()

    # Subordinates can handoff back to supervisors and to siblings
    all_subordinates = set()
    for subordinates in hierarchy.values():
        all_subordinates.update(subordinates)

    for subordinate in all_subordinates:
        # Find supervisors of this subordinate
        supervisors = [sup for sup, subs in hierarchy.items() if subordinate in subs]

        # Find siblings (other subordinates of the same supervisors)
        siblings = set()
        for supervisor in supervisors:
            siblings.update(hierarchy[supervisor])
        siblings.discard(subordinate)  # Remove self

        allowed_handoffs[subordinate] = supervisors + list(siblings)

    return HandoffPolicy(
        allowed_handoffs=allowed_handoffs,
        require_permission=True,
        validate_context=True
    )

def create_open_handoff_policy(agent_names: List[str]) -> HandoffPolicy:
    """Create a policy that allows any agent to handoff to any other agent."""
    allowed_handoffs = {}

    for agent in agent_names:
        allowed_handoffs[agent] = [other for other in agent_names if other != agent]

    return HandoffPolicy(
        allowed_handoffs=allowed_handoffs,
        require_permission=False,  # Open policy
        validate_context=False
    )

def create_restricted_handoff_policy() -> HandoffPolicy:
    """Create a policy that doesn't allow any handoffs."""
    return HandoffPolicy(
        allowed_handoffs={},
        require_permission=True,
        validate_context=True,
        max_handoff_depth=0
    )
