import asyncio
import os
import sys
import json
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest

# Add src to path
sys.path.append(os.path.abspath("."))

async def test_dispatch():
    server_params = {
        "command": "./venv/bin/python",
        "args": ["src/api/mcp_server.py"],
        "env": os.environ.copy()
    }
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            question = "can you plotting data sex columns for patient each other ?"
            print(f"Testing Smart Dispatch with query: {question}")
            
            response = await session.call_tool("smart_intent_dispatch", {
                "question": question
            })
            
            # response.content is a list of content blocks
            # We expect the first block to be text
            raw_text = response.content[0].text
            print("\nRaw JSON Response:")
            print(raw_text)
            
            try:
                data = json.loads(raw_text)
                print("\nParsed Data:")
                print(f"Tool Type: {data.get('tool')}")
                print(f"Task Count: {len(data.get('tasks', []))}")
                if data.get('tasks'):
                    for i, t in enumerate(data['tasks']):
                        print(f"Task {i+1}: {t.get('tool')} with args {t.get('args')}")
            except Exception as e:
                print(f"Error parsing response: {e}")

if __name__ == "__main__":
    asyncio.run(test_dispatch())
