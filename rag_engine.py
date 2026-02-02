import os
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """
    Core RAG logic using LangChain and ChromaDB.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Auto-load existing database if it exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"🔄 Loading existing vector store from {self.persist_directory}...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self._initialize_qa_chain()

    def ingest_documents(self, documents: List):
        print(f"📥 Starting ingestion of {len(documents)} document segments...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Robustly filter metadata to ensure only Chroma-compatible types are passed
        filtered_chunks = filter_complex_metadata(chunks)
        print(f"✂️  Split into {len(filtered_chunks)} chunks. Filtering metadata...")

        self.vector_store = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        print("💾 Vector store persisted successfully.")
        self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # Using 4o-mini for better medical reasoning
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        template = """
        You are an advanced medical data assistant (Medical Assistant GPT).
        Your primary goal is to help doctors understand their documents.
        
        GUIDELINES:
        1. LANGUAGE SUPPORT: Detect the user's language and ALWAYS respond in the SAME language they used (Indonesian, English, etc.).
        2. When a new document is ingested, summarize WHAT it is.
        3. Identify what columns/context are available for plotting.
        4. ADVANCED BIO-ML CAPABILITIES:
           - 'pca', 'umap', 'plsda': For high-dimensional cluster visualization.
           - 'volcano': Significance testing between groups.
           - 'clean_medical_data': Missing value imputation & outlier removal.
           - 'train_medical_model': Automates model training (RandomForest/XGBoost/SVM).
           - 'predict_patient_outcome': Predictions for new data entries.
           - 'discover_markers': Biological marker identification.
        5. If the user asks to "predict" or "train", first "clean" the data for better results.
        6. Focus on INTERNAL PATIENT DATA for patient-specific records.
        7. Use EXTERNAL MEDICAL GUIDELINES for general medical knowledge.

        Context: {context}
        Question: {question}

        Answer (Be descriptive and clinical):
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def smart_query(self, question: str, patient_id_filter: str = None, schema_context: str = None):
        """
        Uses LLM to orchestrate multiple MCP tools sequentially.
        """
        if not self.qa_chain:
            return "System not ready.", "none", []

        intent_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        schema_info = f"\nAVAILABLE MEDICAL COLUMNS: {schema_context}" if schema_context else ""
        
        system_msg = f"""
        You are a Medical Data Orchestrator. Plan a sequence of tasks based on the user's request.{schema_info}
        
        AVAILABLE TOOLS:
        - 'plot': For any chart/visualization.
          * ARGS: 'plot_type' (pca, umap, volcano, heatmap, distribution), 'target_column' (MANDATORY).
          * NOTE: Use 'distribution' for single markers mentioned by the user (e.g., 'Age', 'Gender').
        - 'clean': To prepare and prepare data.
        - 'train': To build ML models. ARGS: 'target_column'.
        - 'discover': To find biomarkers. ARGS: 'target_column'.
        - 'report': For comprehensive summary.
        - 'rag': For general text questions.

        CRITICAL Rules:
        1. If user mentions a column (e.g. 'Usia'), find the closest match in AVAILABLE MEDICAL COLUMNS and use that EXACT name as 'target_column'.
        2. NEVER use technical prefixes like 'image', 'data', or 'sp_mayo' in 'target_column' unless it is the EXACT ONLY match.
        3. If no markers are mentioned, default to 'rag'.
        4. Respond ONLY in JSON format: {{"tasks": [{{"tool": "...", "args": {{...}}}}, ...]}}
        """
        
        try:
            decision_raw = intent_llm.invoke([
                ("system", system_msg),
                ("user", f"User Request: {question}")
            ]).content
            
            import json
            decision_str = decision_raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(decision_str)
            
            tasks = decision.get("tasks", [])
            
            # If no tasks identified or it's just text info, default to RAG
            if not tasks:
                answer, sources = self.query(question, patient_id_filter)
                return answer, "rag", []
            
            return "Planning sequence...", "multi_task", tasks
            
        except Exception as e:
            answer, sources = self.query(question, patient_id_filter)
            return answer, "rag", []

    def query(self, question: str, patient_id_filter: str = None):
        """
        Standard RAG Query.
        """
        query_enriched = question
        if patient_id_filter:
            query_enriched += f" (Patient ID: {patient_id_filter})"
            
        result = self.qa_chain.invoke({"query": query_enriched})
        return result["result"], result["source_documents"]
