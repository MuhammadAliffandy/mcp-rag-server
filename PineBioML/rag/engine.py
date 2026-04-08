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
        """
        Smart query routing using Pure LLM Orchestrator (ZERO HARDCODING).
        
        This method delegates to the PureOrchestrator for agentic tool selection
        without any hardcoded heuristics or keyword matching.
        
        Exception: A deterministic pre-router guard catches external guideline keywords
        and bypasses the LLM to guarantee correct routing to query_external_guidelines.
        """
        if not self.qa_chain: 
            return "RAG not initialized.", "none", [], ""
        
        pine_logger(f"🧠 Smart Query: '{question[:100]}...'")

        # ─── GUARD RAG PRE-ROUTER ─────────────────────────────────────────────
        # Deterministic: if the question mentions known external guideline authorities
        # OR clinical protocol keywords, bypass LLM and route directly to
        # query_external_guidelines. This prevents the LLM from misrouting to
        # internal query_medical_rag which has no external guideline content.
        EXTERNAL_GUIDELINE_TRIGGERS = [
            # Named guideline bodies
            "acg", "ecco", "who guidelines", "nice guidelines", "esc guidelines",
            "aha guidelines", "acc guidelines", "idsa", "ada guidelines",
            "asco guidelines", "esmo", "nccn", "gold guidelines", "ats guidelines",
            "ers guidelines", "kdigo", "eular", "aan guidelines", "acog",
            "bsg guidelines", "wgo guidelines", "sccm", "esicm",
            # Clinical action phrases
            "per acg", "per ecco", "per who", "per nice", "per esc", "per idsa",
            "per ada", "per asco", "per gold", "per kdigo", "per eular",
            "based on acg", "based on ecco", "based on guidelines",
            "according to acg", "according to ecco", "according to who",
            "according to guidelines", "according to protocol",
            "recommended escalation", "escalation therapy", "rescue therapy",
            "international guidelines", "clinical guidelines for",
            "guideline recommendation", "panduan tatalaksana",  # Indonesian
            "rekomendasi guideline", "berdasarkan guideline",   # Indonesian
        ]
        q_lower = question.lower()
        is_external_guideline = any(trigger in q_lower for trigger in EXTERNAL_GUIDELINE_TRIGGERS)
        
        if is_external_guideline:
            pine_logger(f"🌐 Guard RAG Pre-Router: Detected external guideline keyword — bypassing LLM, routing to query_external_guidelines")
            answer = "I will fetch the latest external medical guidelines relevant to your question."
            if self.detect_language(question) == "Indonesian":
                answer = "Saya akan mengambil panduan medis terbaru dari sumber eksternal yang relevan dengan pertanyaan Anda."
            
            # Extract patient context from question text
            from PineBioML.rag.external_guidelines import extract_patient_context
            extracted_ctx = extract_patient_context(question)
            
            tasks = [{
                "tool": "query_guard_rag",
                "args": {
                    "query_intent": question
                }
            }]
            return answer, "multi_task", tasks, ""
        # ─── END GUARD RAG PRE-ROUTER ─────────────────────────────────────────

        pine_logger(f"🧠 Smart Query (LLM route): '{question[:100]}...'")
        
        # 1. Multi-tier RAG Retrieval
        try:
            # Get GLOBAL SUMMARIES (File Inventory)
            summary_docs = self.vector_store.similarity_search("[DEEP SUMMARY]", k=10)
            
            # Fetch more candidates to allow for Python-side filtering
            fetch_k = 50 if patient_id_filter else 10
            
            # ─── COMPREHENSIVE PATIENT DATA EXTRACTION ────────────────────
            # When a patient ID is specified, fetch ALL tabular rows for that
            # patient across ALL sheets (UC_baseline, UC_cpy, UC_lab, etc.)
            # This ensures the orchestrator always has complete clinical data.
            comprehensive_patient_data = ""
            if patient_id_filter:
                try:
                    clean_pid = str(patient_id_filter).strip().lower()
                    all_tab_docs = self.vector_store.get(
                        where={"type": "tabular_row"},
                        include=["documents", "metadatas"]
                    )
                    
                    rows_by_sheet = {}
                    if all_tab_docs and all_tab_docs.get("documents"):
                        for doc_text, meta in zip(all_tab_docs["documents"], all_tab_docs["metadatas"]):
                            p_ids = str(meta.get("patient_ids", "")).lower().strip()
                            id_variants = [clean_pid, f"{clean_pid}.0", f"patient_{clean_pid}", f"id {clean_pid}", f"id{clean_pid}"]
                            if any(v == p_ids or v in p_ids.split(',') for v in id_variants):
                                sheet = meta.get("sheet_name", "Unknown")
                                if sheet not in rows_by_sheet:
                                    rows_by_sheet[sheet] = []
                                rows_by_sheet[sheet].append(doc_text)
                    
                    if rows_by_sheet:
                        parts = [f"\n=== COMPREHENSIVE PATIENT {patient_id_filter} DATA ==="]
                        for sheet_name, rows in rows_by_sheet.items():
                            parts.append(f"\n--- SHEET: {sheet_name} ---")
                            for row in rows:
                                parts.append(row)
                        comprehensive_patient_data = "\n".join(parts)
                        pine_logger(f"📊 Comprehensive extraction: {sum(len(r) for r in rows_by_sheet.values())} rows across sheets: {list(rows_by_sheet.keys())}")
                except Exception as comp_err:
                    pine_logger(f"⚠️ Comprehensive extraction error: {comp_err}")
            # ─── END COMPREHENSIVE EXTRACTION ─────────────────────────────
            
            # Get SESSION DOCUMENTS (User Uploads - Priority)
            session_docs_raw = self.vector_store.similarity_search(
                question, k=fetch_k, 
                filter={"doc_type": {"$in": ["session_upload", "internal_patient"]}}
            )
            
            # Get KNOWLEDGE BASE (SOPs/Guidelines - Reference)
            knowledge_docs_raw = self.vector_store.similarity_search(
                question, k=fetch_k, 
                filter={"doc_type": "internal_record"}
            )
            
            session_docs = []
            knowledge_docs = []
            
            # Custom Python-side filtering for patient IDs
            if patient_id_filter:
                clean_filter = str(patient_id_filter).lower().strip()
                
                # Filter session docs
                for d in session_docs_raw:
                    p_ids = str(d.metadata.get("patient_ids", "")).lower()
                    if clean_filter in p_ids.split(',') or f"patient_{clean_filter}" in p_ids or f"id {clean_filter}" in p_ids:
                        session_docs.append(d)
                
                # Filter knowledge docs 
                for d in knowledge_docs_raw:
                    p_ids = str(d.metadata.get("patient_ids", "")).lower()
                    if clean_filter in p_ids.split(',') or f"patient_{clean_filter}" in p_ids or f"id {clean_filter}" in p_ids:
                        knowledge_docs.append(d)
                        
                pine_logger(f"🔍 Filtered to {len(session_docs)} session docs and {len(knowledge_docs)} knowledge docs for Patient {clean_filter}")
            else:
                session_docs = session_docs_raw
                knowledge_docs = knowledge_docs_raw
            
            # Format context previews
            # Prepend comprehensive patient data to session preview so it's always visible
            session_preview = ""
            if comprehensive_patient_data:
                session_preview = comprehensive_patient_data + "\n\n---\n\n"
            
            session_preview += "\n---\n".join([
                d.page_content[:1500] 
                for d in session_docs 
                if "[DEEP SUMMARY]" not in d.page_content
            ])
            
            knowledge_preview = "\n---\n".join([
                d.page_content[:1000] 
                for d in knowledge_docs 
                if "[DEEP SUMMARY]" not in d.page_content
            ])
            
            inventory_preview = "\n---\n".join([
                d.page_content 
                for d in summary_docs 
                if "[DEEP SUMMARY]" in d.page_content
            ])
            
            pine_logger(f"📚 Retrieved: {len(session_docs)} session, {len(knowledge_docs)} knowledge, {len(summary_docs)} summary docs")
            
        except Exception as e:
            pine_logger(f"⚠️ Retrieval error: {e}")
            session_preview = ""
            knowledge_preview = ""
            inventory_preview = ""
        
        # 2. Delegate to Pure Orchestrator (NO HARDCODING)
        try:
            from PineBioML.rag.orchestrator import PureOrchestrator
            
            orchestrator = PureOrchestrator()
            
            # Smart Truncation Helper (avoids cutting mid-sentence)
            def smart_truncate(text: str, max_chars: int) -> str:
                """Truncate at sentence boundary to preserve context integrity."""
                if len(text) <= max_chars:
                    return text
                
                # Truncate at max
                truncated = text[:max_chars]
                
                # Find last sentence boundary
                last_boundary = max(
                    truncated.rfind('. '),
                    truncated.rfind('.\n'),
                    truncated.rfind('\n\n'),
                    truncated.rfind('| ')  # Table row end
                )
                
                # If we found a boundary in the last 20% of truncated text, use it
                if last_boundary > max_chars * 0.8:
                    return truncated[:last_boundary + 1]
                
                # Otherwise, cut at word boundary
                last_space = truncated.rfind(' ')
                if last_space > max_chars * 0.9:
                    return truncated[:last_space] + "..."
                
                return truncated + "..."
            
            # Build context dictionary with SMART truncation (sentence-boundary aware)
            # 1 token ~= 4 chars. 128k tokens ~= 500k chars.
            # We limit specific sections to keep total prompt under ~50k tokens (200k chars)
            
            safe_schema = smart_truncate(schema_context or "", 20000)      # ~5k tokens
            safe_session = smart_truncate(session_preview, 50000)         # ~12k tokens
            safe_knowledge = smart_truncate(knowledge_preview, 30000)     # ~7.5k tokens
            safe_inventory = smart_truncate(inventory_preview, 20000)     # ~5k tokens
            
            context = {
                "schema": safe_schema,
                "session_preview": safe_session,
                "knowledge_preview": safe_knowledge,
                "inventory_preview": safe_inventory,
                "chat_history": chat_history or []
            }
            
            # Route using pure LLM reasoning
            answer, tasks, full_context = orchestrator.route(question, context)
            
            pine_logger(f"✅ Orchestrator decision: {len(tasks)} tasks")
            
            # Convert to expected format
            tool_type = "multi_task" if tasks else "rag"
            
            return answer, tool_type, tasks, full_context
            
        except Exception as e:
            pine_logger(f"❌ Orchestration error: {e}")
            import traceback
            pine_logger(traceback.format_exc())
            
            # Fallback to direct RAG query
            pine_logger(f"📡 Fallback: Using raw RAG query")
            answer, sources = self.query(question, patient_id_filter)
            return answer, "rag", [], ""

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
        
        # Extract numeric ID if present for flexible matching
        # "patient 1" → "1", "ID 1" → "1", "1" → "1"
        numeric_id = None
        id_match = re.search(r'\b(\d+)\b', ident)
        if id_match:
            numeric_id = id_match.group(1)
        
        for doc_text, meta in zip(docs, metas):
            p_ids = str(meta.get("patient_ids", ""))
            p_ids_low = p_ids.lower()
            
            # Smart Patient Filter with flexible matching
            # ONLY apply filter if user explicitly filtered in sidebar
            # Do NOT filter based on extracted ID from query - let flexible matching handle it
            if patient_id_filter:
                clean_filter = str(patient_id_filter).lower()
                # Check if the filter exists in the comma-separated metadata
                if clean_filter not in p_ids_low.split(','):
                    if f"patient_{clean_filter}" not in p_ids_low:
                        continue
        
            # IMPROVED: Flexible Patient ID matching
            # Match "patient 1" with "ID 1", "id 1", "patient 1", etc.
            should_include = False
            
            if numeric_id:
                # Try multiple format variations
                variations = [
                    numeric_id,  # "1"
                    f"id {numeric_id}",  # "id 1"
                    f"id{numeric_id}",  # "id1"
                    f"patient {numeric_id}",  # "patient 1"
                    f"patient{numeric_id}",  # "patient1"
                ]
                
                # Check if any variation exists in patient_ids metadata
                for var in variations:
                    if var in p_ids_low:
                        should_include = True
                        break
                
                # Also check in document text if not found in metadata
                if not should_include:
                    for var in variations:
                        if var in doc_text.lower():
                            should_include = True
                            break
            
            # Fallback: Substring match in text or source
            if not should_include:
                source = str(meta.get("source", "")).lower()
                if ident_low in doc_text.lower() or ident_low in source:
                    should_include = True
            
            if not should_include:
                continue

            # Document matches! Extract snippets for auditability with line numbers
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
                        line_content = lines[idx]
                        if len(line_content) > 300:
                            line_content = line_content[:300] + "..."
                        window.append(f"{prefix}L{idx+1}: {line_content}")
                    
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

    def query(self, question: str, patient_id_filter: str = None, method: str = "vector"):
        """
        Base query method with support for advanced medical RAG methods.
        - vector: Standard LangChain retrieval.
        - sentence: LlamaIndex Sentence Window retrieval.
        - auto_merging: LlamaIndex Hierarchical merging retrieval.
        """
        if method in ["sentence", "auto_merging"]:
            try:
                from .advanced import AdvancedRAGTool
                # Get all docs from vector store to load into LlamaIndex
                # (In production, we would persist LlamaIndex directly, but for now we bridge)
                res = self.vector_store.get()
                docs_to_load = []
                for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
                    from llama_index.core import Document
                    docs_to_load.append(Document(text=text, metadata=meta))
                
                adv_rag = AdvancedRAGTool()
                adv_rag.documents = docs_to_load
                
                if method == "sentence":
                    pine_logger("🔭 Using Sentence Window Retrieval")
                    ans, nodes = adv_rag.query_sentence_window(question)
                else:
                    pine_logger("🌳 Using Auto-Merging Retrieval")
                    ans, nodes = adv_rag.query_auto_merging(question)
                    
                return ans, nodes
            except Exception as e:
                pine_logger(f"⚠️ Advanced RAG failed: {e}. Falling back to standard vector.")
                import traceback
                pine_logger(traceback.format_exc())

        if not self.qa_chain: return "Not ready.", []
        try:
            res = self.qa_chain.invoke({"query": question})
            return res.get("result", ""), res.get("source_documents", [])
        except Exception as e: return f"Error: {e}", []

    def synthesize_results(self, question: str, tool_outputs: str, rag_context: str = ""):
        """Final clinical synthesis wrapping all findings with strict ColonoSense formatting and tiered evidence."""
        try:
            lang = self.detect_language(question)
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
            
            # Python-side intent detection for Category 1 & 2 triggers
            q_lower = question.lower()
            cat_id = None
            if any(k in q_lower for k in ["severity", "classification", "severity of patient"]):
                cat_id = "Q1.1"
            elif any(k in q_lower for k in ["remission", "target", "remission status"]):
                cat_id = "Q1.2"
            elif any(k in q_lower for k in ["adjust", "medication", "change medication"]) and any(k in q_lower for k in ["should", "need", "dosage", "adjust"]):
                cat_id = "Q2.2"

            from PineBioML.prompts.synthesis import get_synthesis_prompt
            prompt = get_synthesis_prompt(lang, question, rag_context, tool_outputs, category_id=cat_id)
            
            return llm.invoke([("system", "You are ColonoSense, a clinical decision support AI specializing in inflammatory bowel disease (IBD)."), ("human", prompt)]).content
        except Exception as e: return f"Synthesis error: {e}"

    def has_doc_type(self, doc_type: str) -> bool:
        """Checks if any documents of the given doc_type exist in the vector store."""
        if not self.vector_store: return False
        try:
            res = self.vector_store.get(where={"doc_type": doc_type}, limit=1)
            return len(res.get("ids", [])) > 0
        except:
            return False

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
