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
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "server_debug.log"), "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"[{timestamp}] [RAG] {msg}\n")
    except:
        pass

IDENT_RE = re.compile(r"\b[A-Za-z]{3,}\w*\d{3,}\b")  # Accession codes
PATIENT_ID_RE = re.compile(r"\b(?:id|patient|idx)\s*[:#]?\s*(\d+)\b", re.IGNORECASE)

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
        
        # Duplicate Prevention: Check existing sources
        existing_sources = set()
        if self.vector_store is not None:
            res = self.vector_store.get()
            for meta in res.get("metadatas", []):
                if meta.get("source"):
                    existing_sources.add(meta.get("source"))
        
        new_docs = [d for d in filtered_docs if d.metadata.get("source") not in existing_sources]
        
        if not new_docs:
            pine_logger("⏭️ All documents already exist in vector store. Skipping ingestion.")
            return

        pine_logger(f"📥 Starting ingestion of {len(new_docs)} NEW segments")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(new_docs)
        
        if self.vector_store is not None:
            pine_logger("➕ Appending to existing vector store.")
            self.vector_store.add_documents(chunks)
        else:
            pine_logger("🆕 Creating new vector store.")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        pine_logger("💾 Saved to vector store successfully.")
        self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
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
        indo_keywords = [
            'apa', 'bagaimana', 'gimana', 'siapa', 'kenapa', 'mengapa', 'kapan', 'dimana', 'mana',
            'ya', 'ga', 'tidak', 'tak', 'adalah', 'yang', 'dengan', 'untuk', 'pada', 'ke', 'dari',
            'ini', 'itu', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'udah', 'dah',
            'sudah', 'belum', 'sdh', 'blm', 'bisa', 'boleh', 'tahu', 'tau', 'ada', 'kok', 'kali',
            'banget', 'saja', 'aja'
        ]
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
        
        # 1. Get GLOBAL SUMMARIES (Inventaris Pengetahuan)
        summary_docs = self.vector_store.similarity_search("[DEEP SUMMARY]", k=10)
        
        # 2. Get CURRENT SESSION DOCUMENTS (Prioritas Utama User)
        # We try to find content from user-uploaded files
        session_docs = self.vector_store.similarity_search(
            question, k=10, 
            filter={"doc_type": {"$in": ["session_upload", "internal_patient"]}}
        )
        
        # 3. Get KNOWLEDGE BASE / SOPs (Supporting Reference)
        knowledge_docs = self.vector_store.similarity_search(
            question, k=5, 
            filter={"doc_type": "internal_record"}
        )

        # Context Formatting
        session_preview = "\n---\n".join([d.page_content[:1500] for d in session_docs if "[DEEP SUMMARY]" not in d.page_content])
        knowledge_preview = "\n---\n".join([d.page_content[:1000] for d in knowledge_docs if "[DEEP SUMMARY]" not in d.page_content])
        inventory_preview = "\n---\n".join([d.page_content for d in summary_docs if "[DEEP SUMMARY]" in d.page_content])
        
        full_context = f"""
[DATA SESI SAAT INI (USER UPLOAD)]:
{session_preview or "Tidak ada data spesifik dari user upload yang relevan."}

[PENGETAHUAN INTERNAL / SOP]:
{knowledge_preview or "Tidak ada SOP internal yang relevan ditemukan."}

[INVENTARIS FILE]:
{inventory_preview}
        """.strip()

        pine_logger(f"🧠 Smart Query Context: Found {len(session_docs)} session chunks and {len(knowledge_docs)} knowledge chunks.")
        
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

STRICT RULE 3: PINNED SESSION PRIORITY
- [DATA SESI SAAT INI] contains information the user JUST uploaded. This is the PRIMARY source for answering about "this file" or "my data".
- [PENGETAHUAN INTERNAL / SOP] contains hospital guidelines and background knowledge. Use this ONLY as a reference or if the user asks about procedures/rules.
- DO NOT confuse background SOPs with the user's specific uploaded documents.

STRICT RULE 4: CONVERSATIONAL CONTINUITY
- Use the CHAT HISTORY below to resolve ambiguous follow-up questions.
- Example: If user asked for a plot of "sex" and then says "how about age?", you should generate a similar plot but for the "age" column.

[CHAT HISTORY]:
{history_str}

Data Features: {schema_context}
File Context Samples: {full_context[:3000]}

Goal: Execute medical analysis. 

Available Tools (PineBioML Core):
- generate_medical_plot(plot_type, target_column): PCA, distribution, bar, histogram.
- run_pls_analysis(): Supervised class separation (PLS-DA).
- run_umap_analysis(): Non-linear manifold clustering (UMAP).
- run_correlation_heatmap(): Feature relationship matrix.
- get_data_context(): Describe statistics.
- clean_medical_data(): PineBioML imputation logic.
- train_medical_model(target_column): Random Forest tuner.
- discover_markers(target_column): Volcano plot discovery.
- inspect_knowledge_base(): List ALL ingested files and their summaries.
- query_medical_rag(question, patient_id_filter): Search medical guidelines/SOPs.
- exact_identifier_search(query, patient_id_filter): Perform literal/substring search for specific IDs like 'ACCES6U86680' or Patient IDs. Use this when the user mentions a specific code or wants to 'find' something very specific.

ORCHESTRATION LOGIC:
1. PILIH tool yang paling spesifik untuk menjawab kebutuhan client.
2. JANGAN jalankan semua algoritma sekaligus kecuali user minta "analisis lengkap" atau "overview".
3. Gunakan 'run_pls_analysis' jika user ingin melihat pemisahan antar grup (misal: Sehat vs Sakit).
4. Gunakan 'run_umap_analysis' untuk mencari cluster data yang kompleks.
6. Gunakan 'run_correlation_heatmap' jika user ingin melihat hubungan antar variabel.
7. JIKA diminta plotting atau analisis kolom tertentu, PASTIKAN 'target_column' yang Anda pilih ADA di dalam daftar 'Data Features' di atas. Gunakan ID yang sesuai (misal: jika user minta 'age', pilih ID 'age_at_cpy' jika tersedia).
8. ANDA WAJIB menggunakan 'query_medical_rag' untuk pertanyaan apapun yang bisa dijawab oleh dokumen yang sudah di-ingest (baik itu protokol, SOP, data medis, atau profil personal yang ada di dokumen).
9. JANGAN mencoba menjawab pertanyaan tentang fakta spesifik (misal: pendidikan seseorang, detail protokol) menggunakan pengetahuan internal Anda jika ada kemungkinan datanya ada di RAG. Gunakan RAG terlebih dahulu.
10. JIKA user mencari kode spesifik atau ID pasien (misal: "temukan ACCES..."), GUNAKAN 'exact_identifier_search'.
11. Response 'answer' MUST be in {lang} (AS DETECTED) and MUST MATCH THE INPUT LANGUAGE.
12. FINAL CHECK: If the user spoke Indonesian, the 'answer' field MUST be Indonesian. No exceptions.
13. COMMAND CONFIRMATION: If the user says 'ok', 'go ahead', 'run it', 'silahkan', or confirms a suggestion from the chat history, you MUST execute the relevant tool immediately. Do not just reply with text confirming you will do it. Set 'tool' to 'multi_task' and populate 'tasks'.
14. PATIENT ID PARSING: If the user specifies IDs in the prompt (e.g., "analyze ids 1, 2, 3" or "1-5"), YOU MUST EXTRACT them and pass them to the 'patient_ids' argument of the tools. Use the extracted IDs to override the global filter.
15. FEW-SHOT EXAMPLES (STRICTLY FOLLOW THIS PATTERN):

    User: "Silahkan analysis hasil id 1 - 5"
    Output: {{
      "answer": "Baik, saya akan menjalankan analisis PLS-DA untuk pasien ID 1-5 sesuai permintaan.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "run_pls_analysis", "args": {{ "patient_ids": "1,2,3,4,5" }} }}
      ]
    }}

    User: "Coba heatmap untuk pasien 1, 2, 3"
    Output: {{
      "answer": "Membuatkan heatmap korelasi khusus untuk pasien 1, 2, dan 3.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "run_correlation_heatmap", "args": {{ "patient_ids": "1,2,3" }} }}
      ]
    }}

    User: "Ok run it" (Context: User previously asked about UMAP)
    Output: {{
      "answer": "Menjalankan analisis UMAP sekarang.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "run_umap_analysis", "args": {{}} }}
      ]
    }}

    User: "plot distribution of age with dark theme"
    Output: {{
      "answer": "Generating distribution plot for age with dark theme styling.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "age", "styling": '{{"style": {{"theme": "dark"}}}}' }} }}
      ]
    }}

    User: "visualize bmi dengan medical theme"
    Output: {{
      "answer": "Membuat visualisasi BMI dengan tema medical professional.",
      "tool": "multi_task",
      "tasks": [
        {{ "tool": "generate_medical_plot", "args": {{ "plot_type": "distribution", "target_column": "bmi", "styling": '{{"style": {{"theme": "medical"}}}}' }} }}
      ]
    }}

STYLING RULES:
- Themes: "dark", "medical", "colorblind", "vibrant"
- JSON format: {{"style": {{"theme": "NAME", "title_size": 14-20}}}}
- Extract keywords (dark theme, large title) and convert to JSON
- If NO styling mentioned, OMIT the parameter


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
                tool_type = data.get("tool", "rag")
                tasks = data.get("tasks", [])
                
                # HEURISTIC OVERRIDE: Force execution if LLM misclassified
                q_low = question.lower()
                action_keywords = ['analisis', 'analysis', 'pls', 'umap', 'heatmap', 'plot', 'correlation', 'korelasi', 'tabel', 'tampilkan']
                
                # Enhanced ID Extraction
                range_pattern = re.search(r'(?:id|pasien|patient)\s*(\d+)\s*[-,]\s*(\d+)', q_low)
                single_pattern = re.search(r'(?:id|pasien|patient)\s*(\d+)\b', q_low)
                
                patient_ids = None
                is_single_id = False
                
                if range_pattern:
                    start_id = int(range_pattern.group(1))
                    end_id = int(range_pattern.group(2))
                    patient_ids = ",".join([str(i) for i in range(start_id, end_id + 1)])
                elif single_pattern:
                    patient_ids = single_pattern.group(1)
                    is_single_id = True
                
                # Logic: If it's a single ID and a generic "analysis" request, keep it as RAG 
                # unless a specific statistical tool (pls, umap, heatmap) is mentioned.
                if tool_type == "rag" and any(kw in q_low for kw in action_keywords):
                    # Specific Statistical Tools
                    if 'pls' in q_low:
                        tasks = [{"tool": "run_pls_analysis", "args": {"patient_ids": patient_ids} if patient_ids else {}}]
                        tool_type = "multi_task"
                    elif 'umap' in q_low:
                        tasks = [{"tool": "run_umap_analysis", "args": {"patient_ids": patient_ids} if patient_ids else {}}]
                        tool_type = "multi_task"
                    elif 'heatmap' in q_low or 'correlation' in q_low or 'korelasi' in q_low:
                        tasks = [{"tool": "run_correlation_heatmap", "args": {"patient_ids": patient_ids} if patient_ids else {}}]
                        tool_type = "multi_task"
                    elif ('analisis' in q_low or 'analysis' in q_low) and not is_single_id:
                        # Only force PLS-DA for general analysis if it involves multiple IDs/range
                        tasks = [{"tool": "run_pls_analysis", "args": {"patient_ids": patient_ids} if patient_ids else {}}]
                        tool_type = "multi_task"
                    # If it's a single ID and just says "analysis", stay as RAG (handled by query_medical_rag)

                # CORRECTIVE RULE: Prevent statistical comparison for single IDs unless explicitly asked
                if tool_type == "multi_task" and is_single_id:
                    statistical_tools = ['run_pls_analysis', 'run_umap_analysis', 'run_correlation_heatmap']
                    has_statistical = any(t.get('tool') in statistical_tools for t in tasks)
                    specific_mentions = ['pls', 'umap', 'heatmap', 'korelasi', 'correlation']
                    
                    if has_statistical and not any(sm in q_low for sm in specific_mentions):
                        # Use RAG and Search tasks instead of group statistics for a single subject
                        tool_type = "multi_task"
                        tasks = [
                            {"tool": "exact_identifier_search", "args": {"query": patient_ids}},
                            {"tool": "query_medical_rag", "args": {"question": f"Detail clinical and multi-omics analysis for patient {patient_ids}"}}
                        ]
                        if lang == "Indonesian":
                            ans = f"Baik, saya akan mengumpulkan data klinis dan catatan medis detail untuk Pasien {patient_ids} dari basis pengetahuan kami."
                        else:
                            ans = f"I will retrieve detailed clinical records and medical notes for Patient {patient_ids} from our knowledge base."
                
                return ans, tool_type, tasks, full_context
        except Exception as e: 
            pine_logger(f"❌ Orchestration error: {e}")
            pass
            
        pine_logger(f"📡 Fallback: Using raw RAG query for '{question}'")
        answer, sources = self.query(question, patient_id_filter)
        return answer, "rag", [], full_context

    def normalize_identifier(self, s: str) -> str:
        s2 = s.lower().strip()
        s2 = s2.replace("-", " ").replace("_", " ")
        m = re.search(r"\bpatient\s*(\d+)\b", s2)
        if m:
            return f"patient_{int(m.group(1))}"
        return s2

    def extract_identifier(self, q: str) -> str:
        # Try complex accession first
        m = IDENT_RE.search(q)
        if m: return re.sub(r"[^\w\-]+$", "", m.group(0).strip())
        
        # Try simple patient ID
        m = PATIENT_ID_RE.search(q)
        if m: return m.group(1) # Return just the number for simple IDs
        
        return ""

    def exact_search(self, query: str, patient_id_filter: str = None):
        """Perform literal substring search across all ingested documents."""
        if not self.vector_store:
            return "Knowledge base not initialized.", []
        
        # Get all documents from vector store
        res = self.vector_store.get()
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        
        hits = []
        ident = self.extract_identifier(query) or query.strip()
        ident_low = ident.lower()
        
        for doc_text, meta in zip(docs, metas):
            p_ids = str(meta.get("patient_ids", "")).lower()
            
            # Smart Patient Filter
            # If user explicitly filtered in sidebar OR if we extracted a simple patient ID
            active_filter = patient_id_filter or (ident if ident.isdigit() else None)
            
            if active_filter:
                clean_filter = str(active_filter).lower()
                # Check if the filter exists in the comma-separated metadata
                if clean_filter not in p_ids.split(','):
                    if f"patient_{clean_filter}" not in p_ids:
                        continue

            # Substring match in text or source
            source = str(meta.get("source", "")).lower()
            if ident_low in doc_text.lower() or ident_low in source:
                # Extract snippets for auditability with line numbers
                snippets = []
                lines = doc_text.splitlines()
                for i, ln in enumerate(lines):
                    if ident_low in ln.lower():
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        
                        # Build window with line numbers
                        window = []
                        for idx in range(start, end):
                            prefix = ">> " if idx == i else "   "
                            window.append(f"{prefix}L{idx+1}: {lines[idx]}")
                        
                        snippets.append("\n".join(window))
                
                hits.append({
                    "text": doc_text,
                    "metadata": meta,
                    "snippets": snippets[:5] # Limit snippets per hit
                })
                
            if len(hits) >= 50: # Cap results
                break
                
        if not hits:
            return f"No exact matches found for '{ident}'.", []
            
        # Format the result with snippets
        formatted_res = f"### 🔍 Exact Search Results for: `{ident}`\n"
        formatted_res += f"Found {len(hits)} matches across documents.\n\n"
        
        for h in hits:
            src = os.path.basename(h['metadata'].get('source', 'Unknown'))
            formatted_res += f"#### 📄 File: {src}\n"
            if h['snippets']:
                for s in h['snippets']:
                    formatted_res += f"```text\n{s}\n```\n"
            else:
                formatted_res += f"> {h['text'][:200]}...\n"
                
        return formatted_res, hits

    def query(self, question: str, patient_id_filter: str = None):
        if not self.qa_chain: return "Not ready.", []
        try:
            res = self.qa_chain.invoke({"query": question})
            return res.get("result", ""), res.get("source_documents", [])
        except Exception as e: return f"Error: {e}", []

    def synthesize_results(self, question: str, tool_outputs: str, rag_context: str = ""):
        """Final clinical synthesis wrapping all findings with strict language mirroring."""
        try:
            lang = self.detect_language(question)
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
            
            # Universal Mirroring Instruction
            sys_msg = "You are a Senior Clinical Data Scientist. You MUST mirror the user's language perfectly and ABSORB ALL provided context."
            instr = f"Mirror the user's language (Detect {lang}). Wrap findings into a cohesive clinical narrative. Explain biological significance. INTEGRATE EVERY RELEVANT DETAIL from the context."

            user_prompt = f"""
[SYSTEM MANDATE]:
You must provide a COMPREHENSIVE analysis. Do not ignore any clinical details provided in the [RAG CONTEXT]. If the context mentions specific thresholds, protocols, or patient history, INTEGRATE them into your final synthesis.

[USER REQUEST]: {question}

[RAG CONTEXT (CLINICAL BACKGROUND/GUIDELINES/RECORDS)]:
{rag_context or "No specific documentation context provided."}

[TECHNICAL ANALYSIS FINDINGS]:
{tool_outputs}

INSTRUCTIONS:
1. {instr}
2. INTEGRATE the [TECHNICAL ANALYSIS FINDINGS] with the [RAG CONTEXT] deeply. For example, explain how the analysis results compare to clinical norms or specific patient history mentioned in the context.
3. Be EXHAUSTIVE yet concise. Mention relevant biomarkers, medications, and clinical observations from the context.
4. Respond in the EXACT SAME language as the User Request. This is the Mirroring Rule.
5. Professional Markdown formatting.
            """
            return llm.invoke([("system", sys_msg), ("human", user_prompt)]).content
        except Exception as e: return f"Synthesis error: {e}"

    def get_knowledge_summaries(self):
        """Retrieves and beautifully formats knowledge base entries."""
        if not self.vector_store: return "No knowledge base loaded."
        try:
            results = self.vector_store.similarity_search("[DEEP SUMMARY]", k=50)
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
