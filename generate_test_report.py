import asyncio
from PineBioML.rag.engine import RAGEngine
from src.api.mcp_server import smart_intent_dispatch, synthesize_medical_results, query_core_rag, query_guard_rag
import json

queries = [
    ("1", "What is the disease severity of the “patient 1”?"),
    ("1", "Based on the patient’s current status, should the medication be adjusted? continue from chat history"),
    ("10", "Based on the “patient 10”’s current status, should the medication be adjusted?")
]

async def run_tests():
    out_md = "# QA Permutation Test Results\n\n"
    
    for pid, q in queries:
        out_md += f"## Query: `{q}`\n"
        out_md += f"**Patient Context ID:** {pid}\n\n"
        print(f"Testing: {q}")
        
        try:
            orchestrator_ans = smart_intent_dispatch(q, pid)
            plan = json.loads(orchestrator_ans)
            
            outputs = ""
            if "tasks" in plan:
                for task in plan["tasks"]:
                    tool = task["tool"]
                    args = task["args"]
                    if tool == "query_core_rag":
                        outputs += query_core_rag(**args) + "\n"
                    elif tool == "query_guard_rag":
                        outputs += query_guard_rag(**args) + "\n"
                        
            final_res = synthesize_medical_results(q, outputs, "Patient clinical context")
            
            out_md += "### Actual Synthesized Output:\n"
            out_md += "```markdown\n"
            out_md += final_res + "\n"
            out_md += "```\n\n---\n\n"
            
        except Exception as e:
            out_md += f"**Error parsing query:** {e}\n\n---\n\n"
            print("Error:", e)
            
    with open("/Users/muhammadaliffandy/.gemini/antigravity/brain/f3364447-19b1-4803-a9c3-039e757bf1a9/test_permutations_report.md", "w") as f:
        f.write(out_md)

if __name__ == "__main__":
    asyncio.run(run_tests())
