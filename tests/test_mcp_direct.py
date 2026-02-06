#!/usr/bin/env python3
"""
Direct MCP tool call test - bypass Streamlit completely.
This tests if smart_intent_dispatch returns proper JSON with tasks.
"""

import asyncio
import json
import sys
import os

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

async def test_mcp_dispatch():
    """Test smart_intent_dispatch directly via MCP."""
    
    print("=" * 80)
    print("DIRECT MCP TOOL CALL TEST")
    print("=" * 80)
    
    try:
        # Import MCP client
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # Server parameters
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "src.api.mcp_server"],
            env=None
        )
        
        print("\n🔌 Connecting to MCP server...")
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                print("✅ Connected!\n")
                
                # Test prompt
                prompt = "Buatkan PCA plot warnai berdasarkan Disease"
                
                print(f"📝 Testing prompt: {prompt}")
                print("-" * 80)
                
                # Call smart_intent_dispatch
                result = await session.call_tool(
                    "smart_intent_dispatch",
                    arguments={
                        "question": prompt,
                        "patient_id_filter": None,
                        "chat_history": []
                    }
                )
                
                print("\n📦 Raw result:")
                print(result)
                
                # Parse JSON
                if hasattr(result, 'content') and result.content:
                    content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    
                    print("\n📄 Content:")
                    print(content)
                    
                    # Parse as JSON
                    try:
                        data = json.loads(content)
                        
                        print("\n✅ Parsed JSON:")
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                        
                        print(f"\n📊 Analysis:")
                        print(f"  Answer: {data.get('answer', '')[:100]}...")
                        print(f"  Tool: {data.get('tool')}")
                        print(f"  Tasks count: {len(data.get('tasks', []))}")
                        
                        if data.get('tasks'):
                            print(f"\n✅ TASKS FOUND:")
                            for i, task in enumerate(data['tasks']):
                                print(f"    {i+1}. {task.get('tool')} - {task.get('args')}")
                        else:
                            print(f"\n❌ NO TASKS!")
                            print(f"  This is why tools don't execute!")
                        
                    except json.JSONDecodeError as e:
                        print(f"\n❌ JSON parse error: {e}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_mcp_dispatch())
