import asyncio
from PineBioML.rag.engine import RAGEngine
from src.api.mcp_server import smart_intent_dispatch, synthesize_medical_results

engine = RAGEngine()

queries = [
    ("1", "What is the disease severity of patient 1?"),
    ("1", "Based on patient 1's current status, should the medication be adjusted?"),
    ("10", "Based on patient 10's current status, should the medication be adjusted?")
]

async def run_tests():
    for pid, q in queries:
        print(f"\n{'='*50}\nTesting Query: {q}\n{'='*50}")
        # Orchestrator dispatch
        orchestrator_ans = smart_intent_dispatch(q, pid)
        print("--- Orchestrator Phase (Not Final synthesis) ---")
        print(str(orchestrator_ans)[:500] + "...\n")
        
        # We need to simulate app.py's flow. smart_intent_dispatch returns JSON in evaluation.
        # But wait, smart_intent_dispatch returns a string that app.py parses.
        import json
        try:
            plan = json.loads(orchestrator_ans)
            if "tasks" in plan:
                from src.api.mcp_server import query_core_rag, query_guard_rag
                outputs = ""
                for task in plan["tasks"]:
                    tool = task["tool"]
                    args = task["args"]
                    if tool == "query_core_rag":
                        outputs += query_core_rag(**args) + "\n"
                    elif tool == "query_guard_rag":
                        outputs += query_guard_rag(**args) + "\n"
                
                # Synthesize
                final_res = synthesize_medical_results(q, outputs, "Patient clinical context")
                print("\n--- Final Synthesis Phase ---")
                print(final_res)
            else:
                print("No tasks found.")
        except Exception as e:
            print("Error parsing orchestrator out:", orchestrator_ans)
            print(e)
            
if __name__ == "__main__":
    asyncio.run(run_tests())
