import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

if os.path.exists("mcp_server.py"):
    with open("mcp_server.py", "r") as f:
        lines = f.readlines()
        print(f"mcp_server.py length: {len(lines)}")
        for i, line in enumerate(lines):
            if "def smart_intent_dispatch" in line:
                print(f"Found on line {i+1}: {line.strip()}")
            if "Optional[List]" in line:
                print(f"Error string found on line {i+1}!")

if os.path.exists("rag_engine.py"):
    with open("rag_engine.py", "r") as f:
        content = f.read()
        if "🔄 Loading existing vector store" in content:
            print("rag_engine.py has OLD loading string!")
        if "🔄 Loading vector store:" in content:
            print("rag_engine.py has NEW loading string!")
