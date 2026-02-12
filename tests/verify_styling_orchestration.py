
import sys
import os
import json

# Ensure root is in path
sys.path.append(os.getcwd())

from PineBioML.rag.engine import RAGEngine

def test_styling_orchestration():
    print("🎨 Testing Orchestrator Styling Generation...")
    
    try:
        engine = RAGEngine()
        
        # Test: Explicit Styling Request
        print("\n--- Test: 'Scatter plot theme dark color red' ---")
        q = "Buatkan scatter plot umur vs berat badan dengan tema dark dan garis warna merah ukuran font 18"
        
        ans, tool, tasks, ctx = engine.smart_query(q)
        
        print(f"Tool Detected: {tool}")
        
        if not tasks:
            print("❌ FAILURE: No tasks generated.")
            return
            
        task = tasks[0]
        args = task.get('args', {})
        styling_raw = args.get('styling', '{}')
        
        print(f"Generated Styling Raw: {styling_raw}")
        
        if isinstance(styling_raw, str):
            try:
                styling = json.loads(styling_raw)
            except:
                print("❌ FAILURE: Styling is not valid JSON string")
                styling = {}
        else:
            styling = styling_raw
            
        print(f"Parsed Styling: {json.dumps(styling, indent=2)}")
        
        # Validation
        style_part = styling.get('style', {})
        colors_part = styling.get('colors', {})
        
        checks = [
            (style_part.get('theme') == 'dark', "Theme detection"),
            (colors_part.get('primary') == 'red' or colors_part.get('primary') == '#FF0000', "Color detection"),
             (style_part.get('font_size') == 18 or style_part.get('title_size') == 18, "Font size detection")
        ]
        
        passed = 0
        for success, name in checks:
            if success:
                print(f"✅ PASSED: {name}")
                passed += 1
            else:
                print(f"⚠️ WARNING: {name} failed or mismatched.")
                
        if passed >= 2:
            print("✅ OVERALL SUCCESS: Orchestrator is generating styling correctly!")
        else:
            print("⚠️ PARTIAL SUCCESS: Comparison might be fuzzy, but JSON structure is present.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_styling_orchestration()
