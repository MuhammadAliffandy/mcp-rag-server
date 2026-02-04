#!/usr/bin/env python3
"""Quick test script for smart_query"""

import sys
sys.path.insert(0, '/Users/muhammadaliffandy/Documents/my-project/pineBIOML')

from rag_engine import RAGEngine

# Initialize
print("Initializing RAG engine...")
engine = RAGEngine()

# Test visualization detection
print("\n--- Test 1: Visualization keyword detection ---")
question = "buat bar chart untuk id 1-5"
print(f"Question: {question}")

try:
    answer, tool, tasks = engine.smart_query(question, patient_id_filter=None, schema_context="Age, Sum Pmayo")
    print(f"Answer: {answer}")
    print(f"Tool: {tool}")
    print(f"Tasks: {tasks}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
