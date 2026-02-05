import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hub.rag_engine import RAGEngine
import json

# Test the smart_query with analysis keywords
rag = RAGEngine()

test_questions = [
    "Silahkan analysis hasil id 1 - 5",
    "analisis data pasien 1-5",
    "jalankan pls da untuk id 1, 2, 3",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print(f"{'='*60}")
    
    ans, tool, tasks = rag.smart_query(q, patient_id_filter=None, schema_context="Sample columns", chat_history=[])
    
    print(f"Answer: {ans}")
    print(f"Tool: {tool}")
    print(f"Tasks: {json.dumps(tasks, indent=2)}")
