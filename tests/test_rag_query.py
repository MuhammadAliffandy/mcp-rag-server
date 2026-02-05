from src.hub.rag_engine import RAGEngine
import os

def test_rag():
    engine = RAGEngine()
    if not engine.qa_chain:
        print("QA Chain not initialized. Ingesting...")
        from src.hub.rag_processor import DocumentProcessor
        docs = DocumentProcessor.load_directory("internal_docs", doc_type="internal_record")
        engine.ingest_documents(docs)
    
    print("\n--- Testing RAG Query: 'Apa isi hospital protocol?' ---")
    ans, docs = engine.query("Apa isi hospital protocol?")
    print(f"Answer: {ans}")
    print(f"Source docs found: {len(docs)}")
    for d in docs:
        print(f"Ref: {d.metadata.get('source')}")

    print("\n--- Testing Knowledge Summary ---")
    summary = engine.get_knowledge_summaries()
    print(summary)

if __name__ == "__main__":
    test_rag()
