#!/usr/bin/env python3
# Clean fix for rag_engine.py styling examples

with open("src/hub/rag_engine.py", "r") as f:
    lines = f.readlines()

# Find and remove lines 280-311 (the problematic styling examples)
output = []
skip_mode = False
for i, line in enumerate(lines):
    line_num = i + 1
    
    # Start skipping at line 280
    if line_num == 280 and 'User: "plot distribution' in line:
        skip_mode = True
        # Insert clean version here with NO backslashes
        clean_examples = '''    User: "plot distribution of age with dark theme"
    Output: {{
      "answer": "Generating distribution plot for age with dark theme styling.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "age", "styling": '{{"style": {{"theme": "dark"}}}}' }} }}
      ]
    }}

    User: "visualize bmi dengan medical theme"
    Output: {{
      "answer": "Membuat visualisasi BMI dengan tema medical professional.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "bmi", "styling": '{{"style": {{"theme": "medical"}}}}' }} }}
      ]
    }}

STYLING RULES:
- Themes: "dark", "medical", "colorblind", "vibrant"
- JSON format: {{"style": {{"theme": "NAME", "title_size": 14-20}}}}
- Extract keywords (dark theme, large title) and convert to JSON
- If NO styling mentioned, OMIT the parameter

'''
        output.append(clean_examples)
        continue
    
    # Stop skipping after line 311
    if skip_mode and line_num > 311:
        skip_mode = False
    
    if not skip_mode:
        output.append(line)

# Write back
with open("src/hub/rag_engine.py", "w") as f:
    f.writelines(output)

print("✅ Fixed f-string syntax error by removing backslashes")
