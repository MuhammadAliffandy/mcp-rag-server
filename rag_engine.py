import os
import re
import json
import datetime
import sys
import warnings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()

def pine_logger(msg):
    try:
        with open("server_debug.log", "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] [RAG] {msg}\n")
    except:
        pass

class RAGEngine:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vector_store = None
        self.qa_chain = None
        
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            pine_logger(f"🔄 Loading vector store from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self._initialize_qa_chain()

    def ingest_documents(self, documents: list):
        filtered_docs = filter_complex_metadata(documents)
        pine_logger(f"📥 Starting ingestion of {len(filtered_docs)} segments")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(filtered_docs)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        pine_logger("💾 Saved to vector store successfully.")
        self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        template = """
You are a Medical Bio-ML Expert. Use the provided context to answer the QUESTION.
STRICT RULE: Mirror the user's language EXACTLY. If the question is in Indonesian, respond in Indonesian. If in English, respond in English.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTION: 
- Provide detailed, clinical explanations.
- If the context contains a [DEEP SUMMARY], use it to provide a high-level overview.
- Be scanable (use bullet points and bold text).
- Always maintain a professional medical tone.

ANSWER:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def detect_language(self, text: str) -> str:
        """Detects language with a focus on Mirroring what the user provides."""
        indo_keywords = ['apa', 'bagaimana', 'gimana', 'siapa', 'kenapa', 'ya', 'ga', 'tidak', 'adalah', 'yang', 'dengan']
        text_lower = text.lower()
        if any(re.search(rf'\b{kw}\b', text_lower) for kw in indo_keywords):
            return "Indonesian"
        # We can add more common patterns here, but the Orchestrator will have the ultimate mirroring rule.
        return "English"

    def _extract_target_column(self, question_lower: str, schema_context: str = None) -> str:
        if not schema_context: return "Data"
        # schema_context now contains types like "Age(numeric)"
        raw_cols = [c.strip() for c in schema_context.split(',')]
        cols = [re.sub(r'\(.*\)', '', c) for c in raw_cols] # Remove (type)
        
        translation_map = {
            'usia': 'age', 'umur': 'age', 'kelamin': 'sex', 'gender': 'sex', 
            'mayo': 'pmayo', 'darah': 'hb', 'infeksi': 'crp', 'tinja': 'fc'
        }
        
        # Expand question with translations
        expanded_query = question_lower
        for k, v in translation_map.items():
            if k in question_lower: expanded_query += f" {v}"
        
        scores = {}
        for i, c in enumerate(cols):
            c_low = c.lower().replace('_', ' ').replace('-', ' ')
            # Check for direct word overlap
            score = 0
            for word in expanded_query.split():
                if len(word) > 2 and word in c_low:
                    score += 1
            if score > 0: scores[raw_cols[i]] = score
            
        return sorted(scores.items(), key=lambda x: -x[1])[0][0] if scores else raw_cols[0]

    def smart_query(self, question: str, patient_id_filter: str = None, schema_context: str = None, chat_history: list = None):
        if not self.qa_chain: return "RAG not initialized.", "none", []
        
        lang = self.detect_language(question)
        question_lower = question.lower()
        
        # Proactive Discovery: Always include a preview of what we know
        summary_docs = self.vector_store.similarity_search("[DEEP SUMMARY]", k=5)
        context_preview = "\n---\n".join([d.page_content for d in summary_docs if "[DEEP SUMMARY]" in d.page_content])
        
        # Also check for specific relevance to the question
        relevance_docs = self.vector_store.similarity_search(question, k=3)
        relevance_preview = "\n---\n".join([d.page_content[:500] for d in relevance_docs if "[DEEP SUMMARY]" not in d.page_content])
        
        full_context = f"GLOBAL INVENTORY:\n{context_preview}\n\nSPECIFIC RELEVANCE:\n{relevance_preview}"
        
        # LLM Orchestration Prompt
        try:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
            history_str = "".join([f"{m.get('role','').upper()}: {m.get('content','')}\n" for m in (chat_history or [])[-5:]])
            
            system_msg = f"""
You are the Strategic Orchestrator for PineBioML. 
STRICT RULE 1: YOU MUST MIRROR THE USER'S LANGUAGE. 
- If user asks in Indonesian, answer in Indonesian.
- If user asks in English, answer in English.
- STRICT MIRRORING for the 'answer' field is mandatory.

STRICT RULE 2: Use CHAT HISTORY to correct previous mistakes.

Data Features: {schema_context}
File Context Samples: {full_context[:2000]}

Goal: Execute medical analysis. 

Available Tools (PineBioML Core):
- generate_medical_plot(plot_type, target_column): PCA, distribution.
- run_pls_analysis(): Supervised class separation (PLS-DA).
- run_umap_analysis(): Non-linear manifold clustering (UMAP).
- run_correlation_heatmap(): Feature relationship matrix.
- get_data_context(): Describe statistics.
- clean_medical_data(): PineBioML imputation logic.
- train_medical_model(target_column): Random Forest tuner.
- discover_markers(target_column): Volcano plot discovery.
- inspect_knowledge_base(): List ALL ingested files and their summaries.
- query_medical_rag(question): Search medical guidelines/SOPs.

ORCHESTRATION LOGIC:
1. PILIH tool yang paling spesifik untuk menjawab kebutuhan client.
2. JANGAN jalankan semua algoritma sekaligus kecuali user minta "analisis lengkap" atau "overview".
3. Gunakan 'run_pls_analysis' jika user ingin melihat pemisahan antar grup (misal: Sehat vs Sakit).
4. Gunakan 'run_umap_analysis' untuk mencari cluster data yang kompleks.
6. Gunakan 'run_correlation_heatmap' jika user ingin melihat hubungan antar variabel.
7. JIKA user bertanya "isi RAG", "knowledge base", "apa yang kamu tahu", "file apa saja", GUNAKAN 'inspect_knowledge_base'.
8. Response 'answer' MUST be in {lang} and act as a professional medical consultant.

Return JSON ONLY:
{{
  "answer": "Plan/Response in {lang}",
  "tasks": [{{ "tool": "tool_name", "args": {{...}} }}]
}}
            """
            
            res = llm.invoke([("system", system_msg), ("user", f"Question: {question}")]).content
            if "{" in res:
                clean_json = res[res.find("{"):res.rfind("}")+1]
                data = json.loads(clean_json)
                # Force language mirroring in the answer if it leaked English
                ans = data.get("answer", "Planning...")
                return ans, "multi_task", data.get("tasks", [])
        except: pass
            
        answer, sources = self.query(question, patient_id_filter)
        return answer, "rag", []

    def query(self, question: str, patient_id_filter: str = None):
        if not self.qa_chain: return "Not ready.", []
        try:
            res = self.qa_chain.invoke({"query": question})
            return res.get("result", ""), res.get("source_documents", [])
        except Exception as e: return f"Error: {e}", []

    def synthesize_results(self, question: str, tool_outputs: str):
        """Final clinical synthesis wrapping all findings with strict language mirroring."""
        try:
            lang = self.detect_language(question)
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
            
            # Universal Mirroring Instruction
            sys_msg = "You are a Senior Clinical Data Scientist. You MUST mirror the user's language perfectly."
            instr = f"Mirror the user's language (Detect {lang}). Wrap findings into a cohesive clinical narrative. Explain biological significance. IF FINDINGS ARE A FILE LIST: Explain the COLLECTIVE UTILITY of these documents."

            user_prompt = f"""
[USER REQUEST]: {question}
[TECHNICAL FINDINGS]:
{tool_outputs}

INSTRUCTIONS:
1. {instr}
2. Respond in the EXACT SAME language as the User Request. This is the Mirroring Rule.
3. Professional Markdown formatting.
            """
            return llm.invoke([("system", sys_msg), ("human", user_prompt)]).content
        except Exception as e: return f"Synthesis error: {e}"

    def get_knowledge_summaries(self):
        """Retrieves and beautifully formats knowledge base entries."""
        if not self.vector_store: return "No knowledge base loaded."
        try:
            results = self.vector_store.similarity_search("[DEEP SUMMARY]", k=30)
            seen_files = set()
            formatted_output = []
            
            for d in results:
                content = d.page_content
                if "[DEEP SUMMARY]" in content:
                    # Extract fields using regex for robustness
                    file_match = re.search(r"File:\s*([^\n\r]+)", content)
                    format_match = re.search(r"Format:\s*([^\n\r]+)", content)
                    preview_match = re.search(r"Preview:\s*(.+)", content, re.DOTALL)
                    
                    filename = file_match.group(1).strip() if file_match else "Unknown File"
                    
                    if filename not in seen_files:
                        seen_files.add(filename)
                        fmt = format_match.group(1).strip() if format_match else "Document"
                        preview = preview_match.group(1).strip()[:300] + "..." if preview_match else "No preview available."
                        
                        # Markdown Formatting
                        icon = "📄" if "txt" in fmt.lower() or "pdf" in fmt.lower() else "📊"
                        card = f"""
### {icon} {filename}
**Type**: `{fmt}`
> {preview}
"""
                        formatted_output.append(card)
            
            if not formatted_output: return "No summarized knowledge found."
            return "\n".join(formatted_output)
        except Exception as e: return f"Error listing knowledge: {e}"
