import asyncio
from jaf.providers.mcp import create_mcp_sse_client

async def connect_and_list_tools():
    """
    Connects to a running MCP SSE server and lists the available tools.
    """
    server_url = "http://0.0.0.0:8000/juspay"
    print(f"🔌 Connecting to SSE MCP server at {server_url}...")

    # Create an MCP client for the SSE server
    mcp_client = create_mcp_sse_client(server_url)

    try:
        # Initialize the connection to the server
        await mcp_client.initialize()
        print("✅ Connection established successfully.")

        # Get the list of available tools
        available_tools = mcp_client.get_available_tools()

        if not available_tools:
            print("🔧 No tools found on the server.")
        else:
            print(f"🔧 Found {len(available_tools)} available tools:")
            for tool_name in available_tools:
                print(f"  - {tool_name}")

    except Exception as e:
        print(f"❌ Failed to connect or list tools: {e}")
        print("💡 Make sure the SSE MCP server is running and accessible at the specified URL.")

    finally:
        # Cleanly close the connection
        print("🔌 Closing connection...")
        await mcp_client.close()
        print("✅ Connection closed.")

if __name__ == "__main__":
    asyncio.run(connect_and_list_tools())
