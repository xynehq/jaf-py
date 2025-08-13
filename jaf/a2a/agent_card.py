"""
Pure functional Agent Card generation
Transforms JAF agents into A2A Agent Cards
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .types import A2AAgent


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

    def create_agent_skills(agent_name: str, agent: A2AAgent) -> List[Dict[str, Any]]:
        """Create skills for a single agent"""
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

        # Create individual skills for each tool
        tool_skills = [
            {
                "id": f"{agent_name}-{tool.name}",
                "name": tool.name,
                "description": tool.description,
                "tags": [tool.name, "tool", agent_name],
                "examples": generate_examples_for_tool(tool),
                "inputModes": ["text/plain", "application/json"],
                "outputModes": ["text/plain", "application/json"]
            }
            for tool in agent.tools
        ]

        return [main_skill, *tool_skills]

    # Use functional approach to build skills
    all_skills = [
        skill
        for agent_name, agent in agents.items()
        for skill in create_agent_skills(agent_name, agent)
    ]

    return all_skills


def generate_examples_for_agent(agent: A2AAgent) -> List[str]:
    """Pure function to generate examples for an agent"""
    base_examples = [
        f"Ask {agent.name} for help",
        f"What can {agent.name} do?"
    ]

    # Add tool-specific examples using functional approach
    tool_examples = [
        f"Use {tool.name} to {tool.description.lower()}"
        for tool in agent.tools[:2]  # Limit to first 2 tools
    ]

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

    # Validate required fields functionally
    required_field_errors = [
        error for field, error_msg in [
            (card.get("name", "").strip(), "Agent card name is required"),
            (card.get("description", "").strip(), "Agent card description is required"),
            (card.get("url", "").strip(), "Agent card URL is required"),
            (card.get("version", "").strip(), "Agent card version is required"),
            (card.get("protocolVersion", "").strip(), "Protocol version is required")
        ] if not field
        for error in [error_msg]
    ]

    # Skills validation functionally
    skills = card.get("skills", [])
    skills_errors = (
        ["At least one skill is required"] if not skills else
        [
            error
            for index, skill in enumerate(skills)
            for error in [
                f"Skill {index}: ID is required" if not skill.get("id", "").strip() else None,
                f"Skill {index}: Name is required" if not skill.get("name", "").strip() else None,
                f"Skill {index}: Description is required" if not skill.get("description", "").strip() else None,
                f"Skill {index}: At least one tag is required" if not skill.get("tags") or len(skill.get("tags", [])) == 0 else None
            ] if error is not None
        ]
    )

    # Input/Output modes validation functionally
    mode_errors = [
        error for condition, error in [
            (not card.get("defaultInputModes") or len(card.get("defaultInputModes", [])) == 0, "At least one default input mode is required"),
            (not card.get("defaultOutputModes") or len(card.get("defaultOutputModes", [])) == 0, "At least one default output mode is required")
        ] if condition
        for error in [error]
    ]

    # URL validation functionally
    url_errors = [
        "Invalid URL format"
        for url in [card.get("url")]
        if url and not is_valid_url(url)
    ]

    # Combine all errors functionally
    errors = required_field_errors + skills_errors + mode_errors + url_errors

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

    # Merge skills functionally
    all_skills = [
        skill
        for card in cards
        for skill in card.get("skills", [])
    ]

    # Remove duplicate skills by ID functionally
    seen_ids = set()
    unique_skills = [
        skill
        for skill in all_skills
        if skill.get("id") not in seen_ids and not seen_ids.add(skill.get("id"))
    ]

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
