import os
import sys
# Add src to path
sys.path.append(os.path.abspath("."))

from src.core.orchestrator import PureOrchestrator
from dotenv import load_dotenv
load_dotenv()

def debug_orchestrator():
    orchestrator = PureOrchestrator()
    
    question = "can you plotting data sex columns for patient each other ?"
    context = {
        "schema": "sex [ID: sex] (categorical), age [ID: age] (numeric), outcome [ID: outcome] (categorical)",
        "session_preview": "sex, age, outcome\nMale, 45, Dead\nFemale, 33, Alive",
        "knowledge_preview": "",
        "inventory_preview": "",
        "chat_history": []
    }
    
    print(f"Query: {question}")
    answer, tasks, full_context = orchestrator.route(question, context)
    
    print("\nOrchestrator Output:")
    print(f"Answer: {answer}")
    print(f"Tasks: {tasks}")
    
    if tasks:
        for t in tasks:
            print(f"Task Tool Name: '{t.get('tool')}'")

if __name__ == "__main__":
    debug_orchestrator()
