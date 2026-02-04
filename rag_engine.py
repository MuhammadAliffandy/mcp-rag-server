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
        # Note: Chroma 0.4.x+ auto-persists, no need for manual persist()
        print("💾 Vector store saved successfully.")
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

    def detect_language(self, text: str) -> str:
        """
        Detect if user is speaking Indonesian or English.
        Uses distinctly Indonesian words only - avoids ambiguous terms.
        """
        # ONLY distinctly Indonesian words (not found in English)
        indo_keywords = [
            # Question words (distinctly Indo)
            'apa', 'bagaimana', 'gimana', 'kenapa', 'mengapa', 'kapan', 'dimana', 'berapa',
            # Particles (distinctly Indo)
            'dong', 'sih', 'nih', 'deh', 'lah', 'kan', 'ya',
            # Pronouns (distinctly Indo)
            'saya', 'aku', 'kamu', 'kita', 'kami', 'mereka',
            # Common verbs (distinctly Indo)
            'tunjukkan', 'tampilkan', 'lihat', 'cari', 'buat', 'buatkan', 'tolong', 'mohon', 'coba', 'mau', 'ingin',
            # Prepositions (distinctly Indo)
            'untuk', 'dari', 'dengan', 'yang', 'pada', 'dalam', 'kepada',
            # Medical (distinctly Indo)
            'pasien', 'dokter', 'diagnosa', 'laporan', 'hasil',
            # Visualization (distinctly Indo)
            'grafik', 'visualisasi', 'distribusi', 'kolom', 'tabel'
        ]
        
        text_lower = text.lower()
        count = sum(1 for kw in indo_keywords if kw in text_lower)
        
        # Single Indonesian keyword = respond in Indonesian
        return "Indonesian" if count >= 1 else "English"

    def _extract_target_column(self, question_lower: str, schema_context: str = None) -> str:
        """
        DYNAMIC column extraction: Match user's words against actual schema columns.
        Works with ANY Excel file - not limited to predefined column names.
        Uses WORD BOUNDARY matching to prevent false matches like 'age' -> 'image'.
        """
        if not schema_context:
            return "sum_pMayo"  # Fallback when no schema
        
        # Get actual columns from schema
        schema_cols = [c.strip() for c in schema_context.split(',')]
        
        # Extract meaningful words from question (remove common words)
        stop_words = ['plot', 'chart', 'visualize', 'show', 'graph', 'create', 'buat', 'tampilkan', 
                      'visualisasi', 'grafik', 'distribusi', 'distribution', 'untuk', 'for', 'the', 
                      'a', 'an', 'dari', 'of', 'pasien', 'patient', 'data', 'kolom', 'column']
        
        # Replace punctuation (including underscore) with space
        words = question_lower.replace(',', ' ').replace('-', ' ').replace('_', ' ').split()
        keywords = [w for w in words if w not in stop_words and len(w) >= 2]  # Allow 2-char words like 'hb', 'fc'
        
        # Indonesian-English term mapping (also handles common medical terms)
        indo_eng = {
            # Age related
            'usia': 'age', 'umur': 'age',
            # Gender
            'kelamin': 'sex', 'gender': 'sex',
            # Date/time
            'tanggal': 'date', 'waktu': 'date',
            # Diagnosis
            'diagnosis': 'diagnosis', 'diagnosa': 'diagnosis',
            # Blood markers
            'hemoglobin': 'hb', 'darah': 'blood',
            # Mayo score
            'mayo': 'pmayo', 'skor': 'pmayo',
            # Duration
            'durasi': 'duration', 'lama': 'duration',
            # Common medical terms (Indonesian -> English)
            'suhu': 'temperature', 'temperatur': 'temperature',
            'tekanan': 'pressure', 'tensi': 'pressure',
            'kolesterol': 'cholesterol',
            'jantung': 'heart', 'detak': 'heart',
            'berat': 'weight', 'tinggi': 'height',
            'gula': 'glucose', 'glukosa': 'glucose',
            'lemak': 'fat', 'trigliserida': 'triglyceride',
        }
        
        # Expand keywords with English equivalents
        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.append(kw)
            if kw in indo_eng:
                expanded_keywords.append(indo_eng[kw])
        
        # WORD BOUNDARY MATCHING: Split column name into words
        best_match = None
        best_score = 0
        
        import re
        for col in schema_cols:
            col_lower = col.lower()
            # Split column name into words (by underscore, space, or case change)
            col_words = re.split(r'[_\s]', col_lower)
            
            for keyword in expanded_keywords:
                # Check if keyword matches ANY word in column name (exact word match)
                if keyword in col_words:
                    score = 1.0  # Exact word match = highest score
                    if score > best_score:
                        best_score = score
                        best_match = col
                # Check if keyword exactly equals the column name (for short cols like 'hb', 'crp')
                elif keyword == col_lower:
                    score = 1.0
                    if score > best_score:
                        best_score = score
                        best_match = col
                # Also check if keyword is a prefix of any word (e.g., "may" matches "mayo")
                elif any(word.startswith(keyword) for word in col_words):
                    score = 0.8
                    if score > best_score:
                        best_score = score
                        best_match = col
        
        if best_match:
            return best_match
        
        # Default: Return first numeric-looking column
        for col in schema_cols:
            col_lower = col.lower()
            # Skip metadata columns
            if any(skip in col_lower for skip in ['patient', 'id', 'date', 'name', 'source', 'image', 'unnamed', 'note']):
                continue
            return col
        
        return schema_cols[0] if schema_cols else "Sum Pmayo"

    def smart_query(self, question: str, patient_id_filter: str = None, schema_context: str = None):
        """
        Uses LLM to orchestrate multiple PineBioML tools sequentially with natural conversation.
        """
        if not self.qa_chain:
            return "System not ready.", "none", []

        # LANGUAGE DETECTION
        detected_language = self.detect_language(question)
        
        # HARD-CODED VISUALIZATION DETECTOR (Bypass LLM for chart requests)
        viz_keywords = ['chart', 'plot', 'visualize', 'visualis', 'graph', 'grafik', 'diagram', 'histogram', 'bar chart', 'plotting', 'distribusi', 'distribution']
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in viz_keywords):
            # FORCE PLOT TOOL - Don't even ask LLM
            
            # Auto-extract patient IDs if present
            if not patient_id_filter:
                patient_id_filter = self._extract_patient_ids_from_question(question)
            
            # SMART COLUMN EXTRACTION: Parse target column from question
            target_col = self._extract_target_column(question_lower, schema_context)
            
            # Build response in detected language with natural explanation
            if detected_language == "Indonesian":
                answer = f"""Baik! Saya akan buat visualisasi distribusi **{target_col}**{' untuk pasien ID ' + patient_id_filter if patient_id_filter else ''}.

📊 Grafik ini akan menunjukkan sebaran nilai {target_col} dalam dataset Anda, sehingga Anda bisa melihat pola dan outlier yang ada."""
            else:
                answer = f"""Alright! I'll create a **{target_col}** distribution visualization{' for patient ID ' + patient_id_filter if patient_id_filter else ''}.

📊 This chart will show the distribution of {target_col} values in your dataset, helping you identify patterns and outliers."""
            
            # Build task
            task_args = {"plot_type": "distribution", "target_column": target_col}
            if patient_id_filter:
                task_args["patient_ids"] = patient_id_filter
            
            tasks = [{"tool": "plot", "args": task_args}]
            
            # Return in the SAME format as LLM decision (for app.py compatibility)
            return answer, "multi_task", tasks
        
        # HARD-CODED DESCRIBE DETECTOR (Bypass LLM for "what's in file" requests)
        describe_keywords = ['isi file', 'cek file', 'cek data', 'apa isinya', 'lihat data', 'describe', 'summary data', 'what\'s in', 'whats in', 'show columns', 'data summary', 'file contents', 'isi data']
        
        if any(kw in question_lower for kw in describe_keywords):
            # FORCE DESCRIBE TOOL - Don't even ask LLM
            if detected_language == "Indonesian":
                answer = """Baik! Saya akan tampilkan **ringkasan statistik** dari data tabular Anda.

📋 Ini akan mencakup: jumlah pasien, kolom yang tersedia, nilai rata-rata, missing values, dan informasi penting lainnya."""
            else:
                answer = """Alright! I'll show you a **statistical summary** of your tabular data.

📋 This will include: patient count, available columns, mean values, missing data, and other key information."""
            
            tasks = [{"tool": "describe", "args": {}}]
            return answer, "multi_task", tasks
    
        # LLM Logic (Restored connection)

        intent_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        
        schema_info = f"\n📋 AVAILABLE MEDICAL COLUMNS: {schema_context}" if schema_context else ""
        
        system_msg = f"""
You are a **Medical AI Assistant** for PineBioML. Your job is to help doctors and researchers analyze medical data in a clear, professional, and empathetic way.

🌍 **CRITICAL: LANGUAGE MIRRORING**
The user is speaking in **{detected_language}**. You MUST respond ENTIRELY in **{detected_language}**.
- Indonesian example: "Baik, saya akan analisis data pasien Anda..."
- English example: "Alright, I'll analyze your patient data..."

{schema_info}

🛠️ **AVAILABLE PINEBIOML TOOLS**:

1. **plot** - Generate visualizations from TABULAR data (Excel/CSV)
   - `pca`: Overview clustering (default for "plot the data")
   - `heatmap`: Correlation between features
   - `volcano`: Compare 2 groups (e.g., Case vs Control)
   - `distribution`: Single variable analysis (Age, BMI, Sex, etc.)
   - `umap`: Complex non-linear patterns
   - `plsda`: Supervised separation by label

2. **clean** - Handle missing values & outliers in tabular data

3. **train** - Train ML models on tabular data (RandomForest, XGBoost, SVM)

4. **discover** - Find top biomarkers/features from tabular data

5. **predict** - Predict outcome for new patients using trained model

6. **report** - Generate comprehensive PDF report

7. **describe** - Statistical summary of TABULAR data (counts, missing values, columns)

8. **rag** - Answer questions from DOCUMENTS (PDF, DOCX, TXT) using vector search

---

📜 **YOUR PERSONALITY & TONE**:
- **Helpful Colleague**: Speak like a supportive medical colleague, NOT a robot
- **Empathetic**: If user seems concerned, be reassuring ("Let me help you understand...")
- **Clear**: Avoid jargon - explain WHY you're doing something
- **Educational**: Briefly explain clinical relevance

---

✅ **RESPONSE RULES**:

1. **No JSON in user-facing text**: Use natural sentences, NOT `{{"tool": "plot"}}`

2. **Explain reasoning**: 
   - ❌ Bad: "I will run PCA."
   - ✅ Good (Indonesian): "Saya akan analisis PCA untuk melihat apakah ada pola tersembunyi antar pasien..."
   - ✅ Good (English): "I'll run a PCA analysis to reveal hidden clusters between patients..."

3. **Use markdown formatting**:
   - Tables for data summaries
   - Bullet lists for options
   - **Bold** for emphasis

4. **Context-aware**:
   - If user asks for patient ID X → Use RAG to retrieve that patient's documents
   - If data looks messy → Recommend cleaning first
   - If user wants prediction but no model → Suggest training first

5. **Column name matching**: Use the AVAILABLE MEDICAL COLUMNS list to find exact names (case-insensitive search OK)

6. **CRITICAL: KEYWORD FORCING**:
   - If user says "plot", "chart", "visualize", "graph", "distribution", "bar chart", "histogram" → **ALWAYS use plot tool**
   - Indonesian: "visualisasikan", "grafik", "diagram", "plotting" → **ALWAYS use plot tool**
   - **NEVER use RAG for visualization requests**
   - Example: "bar chart for ID 1-5" → plot tool (distribution), NOT rag

---

🎯 **DECISION TREE: WHEN TO USE WHAT?**

**Use RAG tool ONLY for:**
- Questions about DOCUMENTS (PDFs, Word, Text files)
- "What's in the medical guidelines?"
- "Summary of patient report for ID 5"
- "Show me SOP for treatment X"
- "What does the colonoscopy report say?"

**Use PineBioML tools (plot, describe, train) for:**
- Questions about TABULAR DATA (Excel, CSV with numbers)
- "Show age distribution"
- "Correlation between Mayo score and Age"
- "Train a model to predict outcome"
- "Clean the data"

**Use BOTH (chain them) when:**
- User asks: "Get patient 5's Mayo score and plot the trend"
  → Step 1: Use RAG to find Mayo score for patient 5
  → Step 2: Use plot (distribution) to visualize
- User asks: "Compare patients 1-5 based on the uploaded report"
  → Step 1: RAG to extract comparison data
  → Step 2: Describe or plot the comparison

---

📤 **OUTPUT FORMAT** (Internal JSON - not shown to user):

{{
  "tasks": [
    {{"tool": "plot", "args": {{"plot_type": "distribution", "target_column": "Age"}}}},
    {{"tool": "describe", "args": {{}}}}
  ],
  "answer": "**Natural explanation in {detected_language}** using markdown formatting"
}}

---

🎯 **EXAMPLES**:

**Example 1 - Pure Tool (Tabular Analysis)**:
User (Indonesian): "Tunjukkan distribusi umur pasien"
Your answer: "Baik! Saya akan visualisasikan sebaran **usia pasien** dari data Excel. Ini akan membantu kita melihat rentang usia yang paling umum."
Your tasks: [{{"tool": "plot", "args": {{"plot_type": "distribution", "target_column": "Age"}}}}]

**Example 2 - Pure Tool (Data Summary)**:
User (English): "What's in this tabular data?"
Your answer: "Let me give you a statistical summary of the uploaded Excel file - I'll show patient counts, columns, and missing values."
Your tasks: [{{"tool": "describe", "args": {{}}}}]

**Example 3 - Pure RAG (Document Query)**:
User (Indonesian): "Apa isi laporan kolonoskopi pasien 5?"
Your answer: "Baik, saya akan cari laporan kolonoskopi untuk **Pasien ID 5** dari dokumen yang diunggah."
Your tasks: [{{"tool": "rag", "args": {{"question": "Colonoscopy report findings for patient 5", "patient_id_filter": "5"}}}}]

**Example 4 - Chaining RAG + Tool**:
User (English): "Get Mayo scores for patients 1-5 from the reports and plot them"
Your answer: "I'll first extract the Mayo scores from the medical reports for patients 1-5, then create a visualization to compare them."
Your tasks: [
  {{"tool": "rag", "args": {{"question": "Extract Mayo scores for patients 1-5", "patient_id_filter": "1-5"}}}},
  {{"tool": "plot", "args": {{"plot_type": "distribution", "target_column": "Mayo Score"}}}}
]

---

NOW, respond to the user's question below in **{detected_language}** with a natural, helpful tone!
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
        Standard RAG Query with optional patient ID filtering.
        Supports single ID, ranges (1-5), and comma-separated (1,2,3).
        """
        if not self.vector_store:
            return "System not initialized.", []
        
        if patient_id_filter:
            # PARSE PATIENT IDs: Support ranges and comma-separated
            patient_ids = self._parse_patient_ids(patient_id_filter)
            
            # DYNAMIC K: More patients = more documents needed
            k_value = max(10, len(patient_ids) * 5)  # At least 5 docs per patient
            
            try:
                # Build filter: Match any of the patient IDs
                # Note: ChromaDB $in operator for multiple values
                if len(patient_ids) == 1:
                    filter_query = {"patient_ids": {"$contains": patient_ids[0]}}
                else:
                    # For multiple IDs, we'll do manual filtering after retrieval
                    # because ChromaDB's filter syntax is limited
                    filter_query = None
                
                # Retrieve documents
                if filter_query:
                    filtered_retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": k_value, "filter": filter_query}
                    )
                    temp_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
                        retriever=filtered_retriever,
                        return_source_documents=True
                    )
                    result = temp_chain.invoke({"query": question})
                else:
                    # Retrieve more docs and filter manually
                    result = self.qa_chain.invoke({"query": question})
                
                # MANUAL FILTERING: Filter source docs by patient IDs
                filtered_sources = [
                    doc for doc in result.get("source_documents", [])
                    if any(pid in doc.metadata.get("patient_ids", "") for pid in patient_ids)
                ]
                
                if not filtered_sources:
                    missing = ", ".join(patient_ids)
                    return f"No data found for Patient ID(s): {missing}.", []
                
                # Re-generate answer using only filtered sources
                context = "\n\n".join([doc.page_content for doc in filtered_sources[:k_value]])
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                
                prompt = f"""You are a medical data assistant. Answer the question based ONLY on the following patient data:

PATIENT DATA:
{context}

QUESTION: {question}

Provide a comprehensive answer using the patient data above. If multiple patients are mentioned, summarize each one clearly."""
                
                answer = llm.invoke([("human", prompt)]).content
                return answer, filtered_sources
                
            except Exception as e:
                # Fallback: Enrich query with patient context
                patient_list = ", ".join(patient_ids)
                query_enriched = f"{question} (specific to Patient ID(s): {patient_list})"
                result = self.qa_chain.invoke({"query": query_enriched})
                return result["result"], result.get("source_documents", [])
        else:
            # Standard query without filtering
            result = self.qa_chain.invoke({"query": question})
            return result["result"], result.get("source_documents", [])
    
    def _parse_patient_ids(self, patient_id_filter: str) -> list:
        """
        Parse patient ID string into list of individual IDs.
        Supports: "1", "1-5", "1,2,3", "1-3,5,7-9"
        """
        import re
        patient_ids = []
        
        # Split by comma first
        parts = patient_id_filter.replace(" ", "").split(",")
        
        for part in parts:
            # Check if it's a range (e.g., "1-5")
            if "-" in part:
                try:
                    start, end = part.split("-")
                    patient_ids.extend([str(i) for i in range(int(start), int(end) + 1)])
                except ValueError:
                    # Not a valid range, treat as single ID
                    patient_ids.append(part)
            else:
                # Single ID
                patient_ids.append(part)
        
        return list(set(patient_ids))  # Remove duplicates
    
    def _extract_patient_ids_from_question(self, question: str):
        """
        Extract patient IDs from natural language question.
        Examples:
        - "summary for patient 5" → "5"
        - "show data for ID 1-5" → "1-5"
        - "compare patients 1, 3, 5" → "1,3,5"
        """
        import re
        
        # Pattern 1: Range (ID 1-5, patient 1-5)
        range_match = re.search(r'(?:id|patient|pasien)\s*(\d+)\s*-\s*(\d+)', question, re.IGNORECASE)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        
        # Pattern 2: Comma-separated (ID 1,2,3 or patient 1, 2, 3)
        multi_match = re.findall(r'(?:id|patient|pasien)\s*(\d+(?:\s*,\s*\d+)+)', question, re.IGNORECASE)
        if multi_match:
            return multi_match[0].replace(" ", "")
        
        # Pattern 3: Single ID
        single_match = re.search(r'(?:id|patient|pasien)\s*(\d+)', question, re.IGNORECASE)
        if single_match:
            return single_match.group(1)
        
        return None

    def synthesize_results(self, question: str, tool_outputs: str):
        """
        Synthesizes technical tool outputs into a professional, natural clinical explanation.
        Detects user language and mirrors it in the response.
        """
        synth_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        
        # LANGUAGE DETECTION
        detected_language = self.detect_language(question)
        
        # ERROR DETECTION: Check if the outputs contain error messages
        is_error = any(indicator in tool_outputs for indicator in ["Error:", "❌", "⚠️", "Cannot", "failed"])
        
        if is_error:
            # ERROR GUIDANCE MODE (Natural & Empathetic)
            prompt = f"""
You are a **helpful Medical AI Assistant** providing troubleshooting guidance.

**CRITICAL**: Respond in **{detected_language}**.

USER ASKED: "{question}"

SYSTEM ISSUE:
{tool_outputs}

YOUR JOB:
1. Acknowledge the issue empathetically (don't just say "Error occurred")
2. Explain WHY it happened in simple terms
3. Give clear, actionable steps to fix it
4. Be reassuring and supportive

TONE:
- ❌ Bad: "Error: Insufficient data."
- ✅ Good (Indonesian): "Hmm, sepertinya data yang diunggah tidak cukup lengkap untuk analisis ini. Ini bisa terjadi karena..."
- ✅ Good (English): "Hmm, it looks like the uploaded data doesn't have enough information for this analysis. This might happen because..."

OUTPUT FORMAT: Use markdown, bold key terms, and numbered steps if giving instructions.
"""
        else:
            # SUCCESS SYNTHESIS MODE (Clear & Clinical)
            prompt = f"""
You are a **Medical AI Assistant** explaining analysis results to doctors/researchers.

**CRITICAL**: Respond in **{detected_language}**.

USER ASKED: "{question}"

ANALYSIS RESULTS:
{tool_outputs}

YOUR JOB:
1. Summarize what was done (in plain language, not technical jargon)
2. Highlight key findings or insights
3. Explain clinical relevance (WHY does this matter?)
4. Suggest next steps if applicable

TONE:
- Natural, like a helpful colleague
- Use "I" language ("I analyzed...", "I found...")
- Avoid robot-speak ("System executed plot tool...")

EXAMPLES:
**Indonesian**: "Baik! Saya sudah analisis data Anda. Berikut temuan utama: ..."
**English**: "Great! I've analyzed your data. Here are the key findings: ..."

OUTPUT FORMAT: Use markdown, bullet lists, and **bold** for emphasis. Make it easy to scan.
"""
        
        try:
            return synth_llm.invoke([("human", prompt)]).content
        except Exception as e:
            return f"Error synthesizing results: {e}"

