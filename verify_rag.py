import os
import pandas as pd
from rag_processor import DocumentProcessor
from rag_engine import RAGEngine
import matplotlib.pyplot as plt

def verify_system():
    print("🚀 Starting RAG System Verification...")
    
    # 1. Create Mock Tabular Data
    data = {
        "patient_id": ["ID 1", "ID 2", "ID 3", "ID 4", "ID 5", "ID 6"],
        "age": [25, 30, 35, 40, 45, 50],
        "score": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
        "feature_x": [0.1, 0.4, 0.2, 0.8, 0.5, 0.9]
    }
    df = pd.DataFrame(data)
    csv_path = "test_patients.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Created mock data: {csv_path}")
    
    # 2. Test Document Processor
    print("🔍 Testing Document Processor...")
    docs = DocumentProcessor.process_tabular(csv_path)
    assert len(docs) > 0, "Document processor failed to load tabular data."
    assert "patient_ids" in docs[0].metadata, "Metadata missing patient_ids."
    print(f"✅ Document Processor extracted: {docs[0].metadata['patient_ids']}")

    # 3. Test RAG Engine (Requires API Key - will skip if not set)
    if os.getenv("OPENAI_API_KEY"):
        print("🧠 Testing RAG Engine...")
        engine = RAGEngine(persist_directory="./test_chroma")
        engine.ingest_documents(docs)
        response, sources = engine.query("What is the average age of patients in the dataset?")
        print(f"✅ RAG Engine Response: {response}")
    else:
        print("⚠️ Skipping RAG Engine test (OPENAI_API_KEY not set).")

    # 4. Clean up
    if os.path.exists(csv_path):
        os.remove(csv_path)
    print("✅ Verification Complete!")

if __name__ == "__main__":
    verify_system()
