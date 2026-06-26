import os
import sys
import json

# Ensure parent directory is in path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.mcp_server import execute_pinebio_ml

def test_capabilities():
    os.makedirs("ppt_and_reports", exist_ok=True)
    output_log = "ppt_and_reports/PineBioML_Capabilities_Result.md"
    
    patient_data = {
        "case_id": "Patient_1",
        "age": 36,
        "sum_pmayo": 7,
        "mes": 3,
        "indication": "ulcerative colitis"
    }
    payload_str = json.dumps(patient_data)
    
    tasks = [
        "Predict complication risk",
        "Determine remission trajectory",
        "Analyze similarity matrix for cohort matching",
        "Suggest dosage adjustments based on historical cohort outcomes"
    ]
    
    results = []
    results.append("# PineBioML Full Capability Evaluation")
    results.append("The following outputs were generated using local EXPRAG PineBioML matrix matching.\n")
    
    for task in tasks:
        results.append(f"## Task: {task}")
        try:
            print(f"Testing task: {task}...")
            res = execute_pinebio_ml(payload_str, task)
            results.append(f"**Output:**\n```text\n{res}\n```\n")
        except Exception as e:
            results.append(f"**Output:**\n```error\n{e}\n```\n")
            
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"Finished PineBioML testing. Results saved to {output_log}")

if __name__ == "__main__":
    test_capabilities()
