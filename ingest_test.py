from PineBioML.rag.processor import DocumentProcessor
from PineBioML.rag.engine import RAGEngine

docs = DocumentProcessor.load_directory("./internal_docs")
engine = RAGEngine()
engine.ingest_documents(docs)
print("Ingestion complete.")
