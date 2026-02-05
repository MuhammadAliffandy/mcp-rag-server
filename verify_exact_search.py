
import os
import sys
from rag_engine import RAGEngine
from langchain_core.documents import Document

def test_exact_search_logic():
    print("🧪 Testing Exact Search Logic...")
    
    # Initialize engine with temporary location
    engine = RAGEngine(persist_directory="./test_chroma_exact")
    
    # Mock data
    mock_docs = [
        Document(page_content="This is a medical record for ACCES6U86680. Patient shows normal signs.", metadata={"source": "doc1.txt", "patient_ids": "patient_1"}),
        Document(page_content="Reference code: XYZ123456. No issues found.", metadata={"source": "doc2.txt", "patient_ids": "patient_2"}),
        Document(page_content="Another line for ACCES6U86680 here.\nFollow up required.", metadata={"source": "doc1.txt", "patient_ids": "patient_1"})
    ]
    
    engine.ingest_documents(mock_docs)
    
    # Test 1: Search for specific ID
    print("🔍 Test 1: Searching for 'ACCES6U86680'...")
    res, hits = engine.exact_search("Cari kode ACCES6U86680")
    assert len(hits) == 2, f"Expected 2 hits, found {len(hits)}"
    assert "ACCES6U86680" in res
    print("✅ Test 1 Passed!")
    
    # Test 2: Search with patient filter
    print("🔍 Test 2: Searching for 'ACCES6U86680' with patient_2 filter (should fail)...")
    res, hits = engine.exact_search("ACCES6U86680", patient_id_filter="patient_2")
    assert len(hits) == 0, f"Expected 0 hits, found {len(hits)}"
    print("✅ Test 2 Passed!")

    # Test 3: Snippet extraction check
    print("🔍 Test 3: Checking snippets...")
    res, hits = engine.exact_search("XYZ123456")
    assert len(hits) == 1
    assert "```" in res, "Markdown code block (snippet) missing from response"
    print("✅ Test 3 Passed!")

    # Test 4: Simple ID extraction like "ID 8"
    print("🔍 Test 4: Searching for 'Summary for ID 8'...")
    # Ingest a doc with just ID 8
    engine.ingest_documents([Document(page_content="Data for patient 8: Healthy", metadata={"source": "p8.txt", "patient_ids": "8"})])
    res, hits = engine.exact_search("Summary for ID 8")
    assert len(hits) == 1, f"Expected 1 hit for ID 8, found {len(hits)}"
    assert "p8.txt" in res
    print("✅ Test 4 Passed!")

    print("\n🎉 All Logic Tests Passed!")

if __name__ == "__main__":
    try:
        test_exact_search_logic()
    finally:
        # Cleanup
        import shutil
        if os.path.exists("./test_chroma_exact"):
            shutil.rmtree("./test_chroma_exact")
