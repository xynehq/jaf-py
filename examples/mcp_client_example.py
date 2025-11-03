"""
A client example to connect to the streamable HTTP MCP server.
"""

import asyncio
import json
import httpx


async def main():
    """
    Main function to connect to the server, list tools, and execute a tool.
    """
    base_url = "http://localhost:8000"
    agent_name = "math_agent"

    async with httpx.AsyncClient() as client:
        # 1. List tools
        print(f"--- Listing tools for agent: {agent_name} ---")
        try:
            response = await client.get(f"{base_url}/tools/{agent_name}")
            response.raise_for_status()
            tools_data = response.json()
            print(json.dumps(tools_data, indent=2))
        except httpx.HTTPStatusError as e:
            print(f"Error listing tools: {e}")
            return
        except httpx.RequestError as e:
            print(f"Error connecting to server: {e}")
            return

        # 2. Execute math tool
        print("\n--- Executing math tool ---")
        request_body = {"message": "what is 2+2?"}

        try:
            async with client.stream(
                "POST", f"{base_url}/stream", json=request_body, timeout=10
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        event_data = json.loads(line[len("data:") :])
                        print(json.dumps(event_data, indent=2))
        except httpx.HTTPStatusError as e:
            print(f"Error executing tool: {e}")
        except httpx.RequestError as e:
            print(f"Error connecting to server: {e}")

        # 3. Execute weather tool
        print("\n--- Executing weather tool ---")
        request_body = {"message": "what is the weather in London?"}

        try:
            async with client.stream(
                "POST", f"{base_url}/stream", json=request_body, timeout=10
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        event_data = json.loads(line[len("data:") :])
                        print(json.dumps(event_data, indent=2))
        except httpx.HTTPStatusError as e:
            print(f"Error executing tool: {e}")
        except httpx.RequestError as e:
            print(f"Error connecting to server: {e}")


if __name__ == "__main__":
    asyncio.run(main())
