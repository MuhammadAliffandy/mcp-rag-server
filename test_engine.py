from PineBioML.rag.orchestrator import PureOrchestrator
from PineBioML.rag.engine import RAGEngine

engine = RAGEngine()
ans, sources = engine.query("Extract P001 partial mayo, endoscopic scores, lab values, and medication.", "1")
print("RAG EXTRACTED CONTEXT FOR P001:")
print(ans[:2000] if ans else "No data extracted!")
