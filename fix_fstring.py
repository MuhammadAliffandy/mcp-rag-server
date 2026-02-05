#!/usr/bin/env python3
# Script to insert styling examples into rag_engine.py with proper escaping

with open("src/hub/rag_engine.py", "r") as f:
    lines = f.readlines()

# Find the insertion point (after line 278 which contains the closing }})
insertion_index = None
for i, line in enumerate(lines):
    if i >= 270 and '}}' in line and 'run_umap_analysis' in lines[i-3:i].__str__():
        insertion_index = i + 1
        break

if insertion_index is None:
    # Fallback: search for the "Return JSON ONLY:" line
    for i, line in enumerate(lines):
        if 'Return JSON ONLY:' in line:
            insertion_index = i
            break

if insertion_index:
    # Prepare the styling examples with properly escaped braces for f-string
    styling_examples = '''
    User: "plot distribution of age with dark theme"
    Output: {{
      "answer": "I will generate a distribution plot for the age column with a dark theme styling.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "age", "styling": "{{{\\\\"style\\\\": {{\\\\"theme\\\\": \\\\"dark\\\\"}}}}}" }} }}
      ]
    }}

    User: "visualize bmi dengan medical theme"
    Output: {{
      "answer": "Membuatkan visualisasi distribusi BMI dengan tema medical professional.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "bmi", "styling": "{{{\\\\"style\\\\": {{\\\\"theme\\\\": \\\\"medical\\\\"}}}}}" }} }}
      ]
    }}

    User: "show age histogram with large title"
    Output: {{
      "answer": "Creating age histogram with large title.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "histogram", "target_column": "age", "styling": "{{{\\\\"style\\\\": {{\\\\"title_size\\\\": 18}}}}}" }} }}
      ]
    }}

STYLING RULES:
- Themes: "dark", "medical", "colorblind", "vibrant"
- Extract styling keywords and convert to JSON
- If no styling mentioned, OMIT the "styling" parameter

'''
    
    # Insert the content
    lines.insert(insertion_index, styling_examples)
    
    # Write back
    with open("src/hub/rag_engine.py", "w") as f:
        f.writelines(lines)
    
    print(f"✅ Successfully inserted styling examples at line {insertion_index}")
else:
    print("❌ Could not find insertion point")
