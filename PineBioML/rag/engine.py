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
        """
        if not self.qa_chain: 
            return "RAG not initialized.", "none", [], ""
        
        pine_logger(f"🧠 Smart Query: '{question[:100]}...'")
        
        # 1. Multi-tier RAG Retrieval
        try:
            # Get GLOBAL SUMMARIES (File Inventory)
            summary_docs = self.vector_store.similarity_search("[DEEP SUMMARY]", k=10)
            
            # Get SESSION DOCUMENTS (User Uploads - Priority)
            session_docs = self.vector_store.similarity_search(
                question, k=10, 
                filter={"doc_type": {"$in": ["session_upload", "internal_patient"]}}
            )
            
            # Get KNOWLEDGE BASE (SOPs/Guidelines - Reference)
            knowledge_docs = self.vector_store.similarity_search(
                question, k=5, 
                filter={"doc_type": "internal_record"}
            )
            
            # Format context previews
            session_preview = "\n---\n".join([
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
            
            # Build context dictionary with STRICT truncation to avoid 128k token limit
            # 1 token ~= 4 chars. 128k tokens ~= 500k chars.
            # We limit specific sections to keep total prompt under ~50k tokens (200k chars)
            
            safe_schema = (schema_context or "")[:20000] # Limit schema to ~5k tokens
            safe_session = session_preview[:50000] # Limit data preview to ~12k tokens
            safe_knowledge = knowledge_preview[:30000] # Limit knowledge to ~7.5k tokens
            safe_inventory = inventory_preview[:20000] # Limit inventory to ~5k tokens
            
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
{tool_outputs[:50000] if tool_outputs else "No findings."}

[INSTRUCTIONS & CONSTRAINTS]:
1. {instr}
2. **NO REPETITION**: The user has ALREADY seen the [TECHNICAL ANALYSIS FINDINGS]. **DO NOT** repeat tables, lists of numbers, or raw statistics.
3. **INTERPRETATION ONLY**: Focus entirely on the **clinical implications** of the findings. What does the data *mean* for the patient?
    - Instead of: "The mean CRP is 5.2 vs 1.2" (Redundant)
    - Say: "The significantly elevated CRP in the Biologics group suggests active inflammation despite treatment." (Insight)
4. **INTEGRATE RAG CONTEXT**: Use the [RAG CONTEXT] to explain *why* these findings matter based on similar cases or guidelines.
5. Respond in the EXACT SAME language as the User Request.
6. **FORMATTING**: Use **Professional Markdown** (bullet points, bold key terms) for readability.
7. Short and concise. Do not summarize what was just shown. Start directly with the insight.
            """
            return llm.invoke([("system", sys_msg), ("human", user_prompt)]).content
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
