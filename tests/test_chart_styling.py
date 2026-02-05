import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_styled_plots():
    """Test the new styling parameter in generate_medical_plot"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.api.mcp_server"],
        env={"PYTHONPATH": "."}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=" * 60)
            print("Testing Styled Plot Generation")
            print("=" * 60)
            
            # Test 1: Dark Theme
            print("\n1. Testing dark theme...")
            styling_dark = '{"style": {"theme": "dark", "title_size": 16, "grid": true}}'
            result = await session.call_tool(
                "generate_medical_plot",
                arguments={
                    "plot_type": "distribution",
                    "target_column": "age",
                    "styling": styling_dark
                }
            )
            print(f"   Result: {result.content[0].text[:100]}...")
            
            # Test 2: Medical Theme with Custom Title Size
            print("\n2. Testing medical theme...")
            styling_medical = '{"style": {"theme": "medical", "title_size": 18}}'
            result = await session.call_tool(
                "generate_medical_plot",
                arguments={
                    "plot_type": "pca",
                    "styling": styling_medical
                }
            )
            print(f"   Result: {result.content[0].text[:100]}...")
            
            # Test 3: Custom Colors
            print("\n3. Testing custom primary color...")
            styling_custom = '{"colors": {"primary": "#FF5733"}, "style": {"grid": false}}'
            result = await session.call_tool(
                "generate_medical_plot",
                arguments={
                    "plot_type": "distribution",
                    "target_column": "weight",
                    "styling": styling_custom
                }
            )
            print(f"   Result: {result.content[0].text[:100]}...")
            
            # Test 4: No Styling (Default)
            print("\n4. Testing default (no styling)...")
            result = await session.call_tool(
                "generate_medical_plot",
                arguments={
                    "plot_type": "distribution",
                    "target_column": "bmi"
                }
            )
            print(f"   Result: {result.content[0].text[:100]}...")
            
            print("\n" + "=" * 60)
            print("✅ All styling tests completed!")
            print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_styled_plots())
