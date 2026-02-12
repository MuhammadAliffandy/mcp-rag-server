import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from PineBioML.rag.pipeline import EXPRAGPipeline
from PineBioML.data.loader import load_patient_data

def test_multi_source_loading():
    print("\n--- Testing Multi-Source Loading ---")
    # 1. Test Excel Loading (Existing file)
    excel_path = "./internal_docs/Test_AI for MES classification_clinical data_20251002.xlsx"
    if os.path.exists(excel_path):
        profiles = load_patient_data(excel_path)
        print(f"Excel loaded: {len(profiles)} profiles")
        if profiles:
            print(f"Sample Excel Case ID: {profiles[0].case_id}")
    
    # 2. Test CSV Loading (Create a dummy CSV)
    csv_path = "./tests/dummy_patients.csv"
    os.makedirs("./tests", exist_ok=True)
    import pandas as pd
    df = pd.DataFrame([{
        'hadm_id': 'CSV_001',
        'indication': 'IBD',
        'age': 45,
        'mayo_score': 6,
        'hbg': 12.5,
        'procedures': 'colonoscopy, biopsy'
    }])
    df.to_csv(csv_path, index=False)
    
    profiles = load_patient_data(csv_path)
    print(f"CSV loaded: {len(profiles)} profiles")
    if profiles:
        print(f"Sample CSV Case ID: {profiles[0].case_id}")
        print(f"Procedures: {profiles[0].procedures}")
    
    os.remove(csv_path)

def test_multi_axis_similarity():
    print("\n--- Testing Multi-Axis Similarity ---")
    pipeline = EXPRAGPipeline(pinecone_index="test-index")
    
    # Mock some data
    from PineBioML.data.patient import PatientProfile
    p1 = PatientProfile(case_id="P1", indication="IBD", procedures=["colonoscopy"])
    p2 = PatientProfile(case_id="P2", indication="IBD", procedures=["surgery"]) # Diagnostic same, procedure different
    p3 = PatientProfile(case_id="P3", indication="Normal", procedures=["colonoscopy"]) # Diagnostic different, procedure same
    
    pipeline.load_profiles([p2, p3])
    
    # Test similar cohort finding
    cohort = pipeline.find_similar_cohort(p1, top_k=5)
    print(f"Similar Cohort: {cohort}")

def test_high_powered_prompt():
    print("\n--- Testing High-Powered Prompt Generation ---")
    pipeline = EXPRAGPipeline(pinecone_index="test-index")
    
    current_data = {
        'patient': 'NEW_CASE',
        'age': 30,
        'indication': 'UC Flare',
        'procedures': 'colonoscopy'
    }
    
    result = pipeline.run_full_pipeline(
        current_patient_data=current_data,
        query="What is the recommended dosage for Mesalamine?",
        options=["2g daily", "4g daily", "5-ASA only", "Rectal only"],
        top_k_cohort=3
    )
    
    print("\nGenerated Prompt Snippet:")
    print("-" * 20)
    print(result['prompt'][:500] + "...")
    print("-" * 20)

if __name__ == "__main__":
    test_multi_source_loading()
    test_multi_axis_similarity()
    test_high_powered_prompt()
