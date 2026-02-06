import asyncio
import json
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_plot():
    server_params = StdioServerParameters(
        command="./venv/bin/python",
        args=["src/api/mcp_server.py"],
        env={**os.environ, "PYTHONPATH": "."}
    )
    
    print("🚀 Connecting to MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("⚙️ Initializing session...")
            await session.initialize()
            
            p_args = {
                "plot_type": "scatter",
                "data_source": "session",
                "x_column": "sex",
                "y_column": "patient",
                "target_column": "",
                "styling": "{}"
            }
            
            print(f"📊 Calling generate_medical_plot with args: {p_args}")
            try:
                # Set a timeout for the tool call
                result = await asyncio.wait_for(session.call_tool("generate_medical_plot", p_args), timeout=30)
                print("✅ Tool call successful!")
                print(f"Result: {result.content[0].text}")
            except asyncio.TimeoutError:
                print("❌ ERROR: Tool call timed out after 30 seconds!")
            except Exception as e:
                print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_plot())
