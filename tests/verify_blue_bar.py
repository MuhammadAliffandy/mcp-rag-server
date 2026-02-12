
import sys
import os
import json

# Ensure root is in path
sys.path.append(os.getcwd())

from PineBioML.rag.engine import RAGEngine

def test_blue_bar():
    print("🎨 Testing 'Blue Bar Chart' Styling Extraction...")
    
    try:
        engine = RAGEngine()
        
        # Test Case: Specific user request
        q = "buatkan bar chart distribusi umur warnanya biru"
        print(f"\nUser Query: '{q}'")
        
        ans, tool, tasks, ctx = engine.smart_query(q)
        
        print(f"Tool Detected: {tool}")
        if not tasks:
            print("❌ FAILURE: No tasks generated.")
            return

        task = tasks[0]
        print(f"Task Tool: {task.get('tool')}")
        args = task.get('args', {})
        print(f"Args: {json.dumps(args, indent=2)}")
        
        styling = args.get('styling', {})
        
        # Check if styling is captured
        success = False
        if isinstance(styling, dict):
            # Check deep keys
            if 'colors' in styling and ('primary' in styling['colors'] and styling['colors']['primary'] in ['blue', 'biru', '#0000FF']):
                success = True
            elif 'bar_color' in styling and styling['bar_color'] in ['blue', 'biru']:
                 success = True
        elif isinstance(styling, str):
             if 'blue' in styling.lower() or 'biru' in styling.lower():
                 success = True
                 
        if success:
            print("✅ SUCCESS: 'Blue' extraction detected!")
        else:
            print("❌ FAILURE: 'Blue' color NOT detected in styling args.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_blue_bar()
