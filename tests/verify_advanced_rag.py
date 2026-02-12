import sys
import os
import json
import asyncio

# Add project root to path
sys.path.append(os.getcwd())

from PineBioML.rag.engine import RAGEngine

def test_retrieval_modes():
    print("🚀 Initializing Advanced RAG Verification...")
    engine = RAGEngine(persist_directory="./chroma_db")
    
    # Check if we have documents
    res = engine.vector_store.get()
    if not res.get("documents"):
        print("❌ No documents found in vector store. Please ingest documents first.")
        return

    question = "What are the specific clinical protocols for medical emergencies?"
    
    modes = ["vector", "sentence", "auto_merging"]
    
    for mode in modes:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        try:
            ans, sources = engine.query(question, method=mode)
            print(f"✅ Answer (snippet): {str(ans)[:200]}...")
            print(f"📚 Sources retrieved: {len(sources)}")
            for i, s in enumerate(sources[:2]):
                src_name = s.metadata.get('source', 'unknown') if hasattr(s, 'metadata') else "LlamaIndex Node"
                print(f"   [{i+1}] {src_name}")
        except Exception as e:
            print(f"❌ Mode {mode} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_retrieval_modes()
