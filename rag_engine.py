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
        You are a Medical Data Orchestrator for PineBioML. Plan a sequence of tasks based on the user's request.{schema_info}
        
        AVAILABLE TOOLS (PineBioML Suite):
        - 'plot': Generate valid visualizations.
           - 'pca' (Default for "plot this"): Dataset overview/clustering.
           - 'heatmap': Correlation between genes/features.
           - 'volcano': Compare 2 groups (e.g. Case vs Control).
           - 'distribution': Analysis of a single variable (e.g. Age).
           - 'umap': Complex non-linear clustering.
           - 'plsda': Supervised separation.
        - 'clean': Impute missing values & remove outliers. (Suggest this before training).
        - 'train': Train ML models (RandomForest, XGBoost, SVM).
        - 'discover': Identify top biomarkers/features.
        - 'predict': Predict outcome for new patient data.
        - 'report': Generate full PDF/Image report.
        - 'describe': detailed statistical summary of data (count, missing values, columns).
        - 'rag': Answer text-based medical questions.

        CRITICAL Rules:
        1. **Language Mirroring**: DETECT the user's language (English, Indonesian, etc.) and ALWAYS respond in that SAME language.
        2. **Professional Tone**: You are an expert Medical AI. Use professional, clinical, and encouraging language. Avoid casual slang.
        3. **Comprehensive Logic**: In your 'answer', explain WHY you are choosing these tools. Be educational (e.g., "I will run PCA to reveal hidden clusters...").
        4. **Smart Plotting**: If user says "plot" without type, choose 'pca' for whole data, or 'distribution' for single column.
        5. **Data Summary**: If user asks "What is in the file?", "Describe data", or "Show summary", use the 'describe' tool.
        6. **Formatting**: When answering about data content (rows/columns), USE MARKDOWN TABLES or BULLET LISTS. Do NOT use long narrative paragraphs.
        7. **Schema First**: Use "AVAILABLE MEDICAL COLUMNS" to find exact column names for 'target_column'.
        8. **Chain of Thought**: If user wants to "predict" or "train", recommend "clean" first if data might be dirty.
        
        8. TASK STRUCTURE (Strict JSON):
           The "tasks" list MUST contain objects with EXACTLY this structure:
           {{"tool": "tool_name", "args": {{"arg1": "value1"}}}}
           
           Example: {{"tool": "plot", "args": {{"plot_type": "pca", "target_column": "Diagnosis"}}}}
        
        Final JSON Output Format:
        {{"tasks": [{{"tool": "...", "args": {{...}}}}], "answer": "Markdown formatted explanation (In User's Language)"}}
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
            direct_answer = decision.get("answer")

            # FIX: If tasks exist, prioritize them! 
            # (Previously, if 'answer' existed, it ignored tasks and just returned text)
            if tasks:
                 return direct_answer if direct_answer else "Executing planned actions...", "multi_task", tasks
            
            # If ONLY a direct answer exists (no tasks), return it as RAG response
            if direct_answer:
                return direct_answer, "rag", []
            
            # If no tasks and no direct answer, default to RAG search
            if not tasks:
                answer, sources = self.query(question, patient_id_filter)
                return answer, "rag", []
            
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

    def synthesize_results(self, question: str, tool_outputs: str):
        """
        Synthesizes technical tool outputs into a professional clinical explanation.
        """
        synth_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        
        prompt = f"""
        You are an Expert Medical Consultant.
        
        CONTEXT:
        The user asked: "{question}"
        The technical tools produced these results:
        {tool_outputs}
        
        TASK:
        Provide a concise, professional clinical summary that:
        1. Interprets the tool results for the doctor (e.g. "The PCA plot shows clear separation...").
        2. Wraps the findings into a cohesive answer.
        3. Maintains the language of the user's question (Indonesian/English).
        4. Does NOT repeat technical error logs, just the insights.
        
        Professional Response:
        """
        
        try:
            return synth_llm.invoke(prompt).content
        except Exception as e:
            return f"Error synthesizing results: {e}"
