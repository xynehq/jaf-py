"""
Pure functional Agent Card generation
Transforms JAF agents into A2A Agent Cards
"""

from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

from .types import (
    AgentCard, AgentSkill, A2AAgent, AgentCapabilities, AgentProvider
)


def generate_agent_card(
    config: Dict[str, Any],
    agents: Dict[str, A2AAgent],
    base_url: str = "http://localhost:3000"
) -> Dict[str, Any]:
    """Pure function to generate Agent Card from A2A agents"""
    return {
        "protocolVersion": "0.3.0",
        "name": config["name"],
        "description": config["description"],
        "url": f"{base_url}/a2a",
        "preferredTransport": "JSONRPC",
        "version": config["version"],
        "provider": config.get("provider"),
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": generate_skills_from_agents(agents),
        "securitySchemes": generate_security_schemes(),
        "security": generate_security_requirements()
    }


def generate_skills_from_agents(agents: Dict[str, A2AAgent]) -> List[Dict[str, Any]]:
    """Pure function to generate skills from A2A agents"""
    skills = []
    
    for agent_name, agent in agents.items():
        # Create a main skill for the agent
        main_skill = {
            "id": f"{agent_name}-main",
            "name": agent.name,
            "description": agent.description,
            "tags": ["general"] + [tool.name for tool in agent.tools],
            "examples": generate_examples_for_agent(agent),
            "inputModes": agent.supported_content_types,
            "outputModes": agent.supported_content_types
        }
        
        skills.append(main_skill)
        
        # Create individual skills for each tool
        for tool in agent.tools:
            tool_skill = {
                "id": f"{agent_name}-{tool.name}",
                "name": tool.name,
                "description": tool.description,
                "tags": [tool.name, "tool", agent_name],
                "examples": generate_examples_for_tool(tool),
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"]
            }
            
            skills.append(tool_skill)
    
    return skills


def generate_examples_for_agent(agent: A2AAgent) -> List[str]:
    """Pure function to generate examples for an agent"""
    base_examples = [
        f"Ask {agent.name} for help",
        f"What can {agent.name} do?"
    ]
    
    # Add tool-specific examples
    tool_examples = []
    for tool in agent.tools[:2]:  # Limit to first 2 tools
        tool_examples.append(f"Use {tool.name} to {tool.description.lower()}")
    
    return base_examples + tool_examples


def generate_examples_for_tool(tool: Any) -> List[str]:
    """Pure function to generate examples for a tool"""
    return [
        f"Use {tool.name}",
        tool.description,
        f"Help me with {tool.name.replace('_', ' ')}"
    ]


def generate_security_schemes() -> Dict[str, Any]:
    """Pure function to generate security schemes"""
    return {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "Bearer token authentication"
        },
        "apiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication"
        }
    }


def generate_security_requirements() -> List[Dict[str, List[str]]]:
    """Pure function to generate security requirements"""
    return [
        # No authentication required by default
        {},
        # Optional bearer auth
        {"bearerAuth": []},
        # Optional API key
        {"apiKey": []}
    ]


def generate_agent_card_for_agent(
    agent_name: str,
    agent: A2AAgent,
    config: Optional[Dict[str, Any]] = None,
    base_url: str = "http://localhost:3000"
) -> Dict[str, Any]:
    """Pure function to generate Agent Card for a specific agent"""
    agent_map = {agent_name: agent}
    
    card_config = {
        "name": (config or {}).get("name", agent.name),
        "description": (config or {}).get("description", agent.description),
        "version": (config or {}).get("version", "1.0.0"),
        "provider": (config or {}).get("provider", {
            "organization": "JAF Agent",
            "url": "https://functional-agent-framework.com"
        })
    }
    
    return generate_agent_card(card_config, agent_map, base_url)


def validate_agent_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to validate Agent Card"""
    errors = []
    
    # Required fields validation
    if not card.get("name", "").strip():
        errors.append("Agent card name is required")
    
    if not card.get("description", "").strip():
        errors.append("Agent card description is required")
    
    if not card.get("url", "").strip():
        errors.append("Agent card URL is required")
    
    if not card.get("version", "").strip():
        errors.append("Agent card version is required")
    
    if not card.get("protocolVersion", "").strip():
        errors.append("Protocol version is required")
    
    # Skills validation
    skills = card.get("skills", [])
    if not skills:
        errors.append("At least one skill is required")
    else:
        for index, skill in enumerate(skills):
            if not skill.get("id", "").strip():
                errors.append(f"Skill {index}: ID is required")
            if not skill.get("name", "").strip():
                errors.append(f"Skill {index}: Name is required")
            if not skill.get("description", "").strip():
                errors.append(f"Skill {index}: Description is required")
            if not skill.get("tags") or len(skill.get("tags", [])) == 0:
                errors.append(f"Skill {index}: At least one tag is required")
    
    # Input/Output modes validation
    if not card.get("defaultInputModes") or len(card.get("defaultInputModes", [])) == 0:
        errors.append("At least one default input mode is required")
    
    if not card.get("defaultOutputModes") or len(card.get("defaultOutputModes", [])) == 0:
        errors.append("At least one default output mode is required")
    
    # URL validation
    url = card.get("url")
    if url and not is_valid_url(url):
        errors.append("Invalid URL format")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }


def is_valid_url(url: str) -> bool:
    """Pure helper function to validate URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def create_minimal_agent_card(
    name: str,
    description: str,
    url: str = "http://localhost:3000/a2a"
) -> Dict[str, Any]:
    """Pure function to create minimal Agent Card"""
    return {
        "protocolVersion": "0.3.0",
        "name": name,
        "description": description,
        "url": url,
        "preferredTransport": "JSONRPC",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{
            "id": "general",
            "name": "General Assistant",
            "description": "General purpose assistance",
            "tags": ["general", "assistant"],
            "examples": ["How can I help you?"]
        }]
    }


def merge_agent_cards(*cards: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to merge multiple Agent Cards"""
    if not cards:
        return create_minimal_agent_card("Empty", "No agents")
    
    base_card = cards[0]
    additional_cards = cards[1:]
    
    merged_skills = [*base_card.get("skills", [])]
    for card in additional_cards:
        merged_skills.extend(card.get("skills", []))
    
    # Remove duplicate skills by ID
    unique_skills = []
    seen_ids = set()
    for skill in merged_skills:
        skill_id = skill.get("id")
        if skill_id not in seen_ids:
            unique_skills.append(skill)
            seen_ids.add(skill_id)
    
    # Merge input/output modes
    merged_input_modes = list(set([
        *base_card.get("defaultInputModes", []),
        *[mode for card in additional_cards for mode in card.get("defaultInputModes", [])]
    ]))
    
    merged_output_modes = list(set([
        *base_card.get("defaultOutputModes", []),
        *[mode for card in additional_cards for mode in card.get("defaultOutputModes", [])]
    ]))
    
    # Merge capabilities
    capabilities = base_card.get("capabilities", {})
    for card in additional_cards:
        card_capabilities = card.get("capabilities", {})
        capabilities = {
            "streaming": capabilities.get("streaming", False) or card_capabilities.get("streaming", False),
            "pushNotifications": capabilities.get("pushNotifications", False) or card_capabilities.get("pushNotifications", False),
            "stateTransitionHistory": capabilities.get("stateTransitionHistory", False) or card_capabilities.get("stateTransitionHistory", False)
        }
    
    return {
        **base_card,
        "skills": unique_skills,
        "defaultInputModes": merged_input_modes,
        "defaultOutputModes": merged_output_modes,
        "capabilities": capabilities
    }


def create_agent_card_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function to create Agent Card from configuration"""
    base_url = config.get("baseUrl", "http://localhost:3000")
    
    return {
        "protocolVersion": "0.3.0",
        "name": config["name"],
        "description": config["description"],
        "url": f"{base_url}/a2a",
        "preferredTransport": "JSONRPC",
        "version": config.get("version", "1.0.0"),
        "provider": config.get("provider", {
            "organization": "JAF Framework",
            "url": "https://functional-agent-framework.com"
        }),
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            **config.get("capabilities", {})
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": generate_skills_from_agents(config.get("agents", {}))
    }