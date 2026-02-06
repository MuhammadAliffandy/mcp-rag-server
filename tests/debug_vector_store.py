"""
Debug script to check what's actually in the vector store
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.hub.rag_engine import RAGEngine

print("=" * 80)
print("🔍 VECTOR STORE CONTENT DEBUGGER")
print("=" * 80)
print()

try:
    engine = RAGEngine()
    print("✅ RAG Engine loaded")
    
    if engine.vector_store is None:
        print("❌ Vector store is None - ChromaDB not initialized or empty")
        print()
        print("Solution: Upload data via Streamlit app first")
        sys.exit(0)
    
    print()
    
    # Check total documents
    print("📊 Searching for patient-related documents...")
    results = engine.vector_store.similarity_search("patient ID data", k=20)
    
    if not results:
        print()
        print("❌ NO DOCUMENTS FOUND IN VECTOR STORE!")
        print()
        print("Possible reasons:")
        print("1. Data belum di-upload via Streamlit app")
        print("2. ChromaDB path salah")
        print("3. Vector store kosong (fresh install)")
        print()
        print("Solution:")
        print("1. Run: streamlit run app.py")
        print("2. Upload test_patients.csv via sidebar")
        print("3. Run this script again")
    else:
        print(f"✅ Found {len(results)} documents")
        print()
        
        # Show each document
        for i, doc in enumerate(results, 1):
            print(f"{'='*80}")
            print(f"📄 Document {i}/{len(results)}")
            print(f"{'='*80}")
            print(f"Content Preview:")
            print(f"  {doc.page_content[:300]}...")
            print()
            print(f"Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print()
        
        # Check for specific patient IDs
        print("=" * 80)
        print("🔎 Checking for specific patient IDs...")
        print("=" * 80)
        
        test_ids = ["1", "ID 1", "patient 1", "Patient 1"]
        for test_id in test_ids:
            results = engine.vector_store.similarity_search(test_id, k=3)
            if results:
                print(f"✅ Found matches for '{test_id}':")
                for doc in results[:2]:
                    print(f"   - {doc.page_content[:100]}...")
            else:
                print(f"❌ No matches for '{test_id}'")
        
        print()
        print("=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)
        print()
        print("If patient data not found:")
        print("1. Check patient_id format in uploaded files")
        print("2. Upload test_patients.csv via Streamlit")
        print("3. Verify data ingestion completed successfully")
        print()
        print("If data exists but query fails:")
        print("1. Check exact_identifier_search implementation")
        print("2. Verify patient ID normalization")
        print("3. Test with exact format from vector store")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Make sure:")
    print("1. ChromaDB is initialized")
    print("2. OPENAI_API_KEY is set")
    print("3. Dependencies are installed")

print()
print("=" * 80)
