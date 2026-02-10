import os
import sys
import sqlite3
import pandas as pd

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.exprag_pipeline import EXPRAGPipeline
from src.core.patient_profile import PatientProfile

def seed_test_database(db_path):
    print(f"--- Seeding Test Database: {db_path} ---")
    if os.path.exists(db_path): os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE patient_records (hadm_id TEXT PRIMARY KEY, text TEXT)")
    
    # Seed 20 patients with varying similarity
    for i in range(1, 21):
        pid = f"PAT_{i:03d}"
        text = f"Clinical Record for Patient {pid}. Diagnostic: IBD. Procedure: Colonoscopy. Outcome: Successful."
        cursor.execute("INSERT INTO patient_records (hadm_id, text) VALUES (?, ?)", (pid, text))
    
    conn.commit()
    conn.close()

def test_diversity_binning():
    print("\n--- Testing Diversity (Quintile Binning) ---")
    pipeline = EXPRAGPipeline(
        pinecone_index="test",
        excel_path="none", # manual seeding
        db_path="./tests/test_exprag.db",
        use_diversity=True
    )
    
    # Create 20 mock profiles with linear similarity
    profiles = []
    for i in range(1, 21):
        profiles.append(PatientProfile(
            case_id=f"PAT_{i:03d}",
            indication="screening",
            age=30 + i, # Similarity decreases as i increases
            procedures=["colonoscopy"]
        ))
    pipeline.load_profiles(profiles)
    
    query_p = PatientProfile(case_id="QUERY", indication="screening", age=30, procedures=["colonoscopy"])
    
    # With diversity, we should expect IDs from across the range, not just PAT_001-PAT_005
    cohort = pipeline.find_similar_cohort(query_p, top_k=5)
    print(f"Diverse Cohort (5 bins): {cohort}")
    # Check if we have someone from the 'less similar' end (e.g. PAT_016 or higher)
    has_distant = any(int(pid.split('_')[1]) > 15 for pid in cohort)
    print(f"Diversity check (contains distant cases): {'✅' if has_distant else '❌'}")

def test_separate_rag_mode():
    print("\n--- Testing Separate RAG Mode (Full Record Snippets) ---")
    pipeline = EXPRAGPipeline(
        pinecone_index="test",
        excel_path="none",
        db_path="./tests/test_exprag.db",
        rag_mode="separate"
    )
    
    # Need to load profiles for Phase 2 to work
    profiles = [PatientProfile(case_id=f"PAT_{i:03d}", indication="ibd", age=30+i, procedures=["colonoscopy"]) for i in range(1, 11)]
    pipeline.load_profiles(profiles)
    
    current_data = {'patient': 'NEW', 'age': 35, 'indication': 'flare'}
    result = pipeline.run_full_pipeline(
        current_patient_data=current_data,
        query="What is the historical outcome?",
        top_k_cohort=3
    )
    
    print("\nPrompt Experience Block (Separate Mode):")
    print("-" * 20)
    # Check if it contains the "--- Experience Patient ---" headers from separate mode
    print(result['prompt'].split("historical experience):")[1][:500])
    print("-" * 20)
    if "Experience Patient 1" in result['prompt']:
        print("✅ Separate mode formatting verified.")
    else:
        print("❌ Separate mode formatting failed.")

if __name__ == "__main__":
    os.makedirs("./tests", exist_ok=True)
    seed_test_database("./tests/test_exprag.db")
    
    try:
        test_diversity_binning()
        test_separate_rag_mode()
    finally:
        if os.path.exists("./tests/test_exprag.db"):
            os.remove("./tests/test_exprag.db")
