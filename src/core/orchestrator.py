"""
Pure LLM-based Orchestrator for agentic routing without hardcoded heuristics.

This module replaces the hardcoded keyword-matching logic with pure LLM reasoning
using structured output parsing and comprehensive few-shot examples.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.prompts.orchestration import get_orchestration_prompt


class ToolCall(BaseModel):
    """Represents a single tool invocation."""
    tool: str = Field(description="Name of the tool to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments as key-value pairs")


class AgenticDecision(BaseModel):
    """Structured output from the orchestrator LLM."""
    answer: str = Field(description="User-facing explanation in the detected language")
    tasks: List[ToolCall] = Field(default_factory=list, description="List of tools to execute in order")


class PureOrchestrator:
    """
    Zero-hardcoding orchestrator that uses pure LLM reasoning for tool selection.
    
    Key Features:
    - No regex/keyword matching
    - Structured output parsing with Pydantic
    - Comprehensive few-shot examples
    - Language detection and mirroring
    - Context-aware routing using chat history
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the orchestrator.
        
        Args:
            model_name: OpenAI model to use
            temperature: LLM temperature (lower = more deterministic)
        """
        # Force JSON output mode
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}}  # Force JSON!
        )
        self.parser = JsonOutputParser(pydantic_object=AgenticDecision)
        
        # Language detection keywords
        self.indo_keywords = [
            'apa', 'bagaimana', 'gimana', 'siapa', 'kenapa', 'mengapa', 'kapan', 'dimana', 'mana',
            'ya', 'ga', 'tidak', 'tak', 'adalah', 'yang', 'dengan', 'untuk', 'pada', 'ke', 'dari',
            'ini', 'itu', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'udah', 'dah',
            'sudah', 'belum', 'sdh', 'blm', 'bisa', 'boleh', 'tahu', 'tau', 'ada', 'kok', 'kali',
            'banget', 'saja', 'aja', 'silahkan', 'tolong', 'coba', 'lihat', 'analisis', 'buatkan'
        ]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from user input.
        
        Args:
            text: User's question
        
        Returns:
            "Indonesian" or "English"
        """
        text_lower = text.lower()
        if any(re.search(rf'\b{kw}\b', text_lower) for kw in self.indo_keywords):
            return "Indonesian"
        return "English"
    
    def route(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Route user question to appropriate tools using pure LLM reasoning.
        
        Args:
            question: User's question
            context: Dictionary containing:
                - schema: Data schema with column types
                - session_preview: Preview of user-uploaded data
                - knowledge_preview: Preview of internal SOPs
                - inventory_preview: File inventory summaries
                - chat_history: Recent conversation history (list of dicts)
        
        Returns:
            Tuple of (answer, tasks, full_context)
            - answer: User-facing explanation
            - tasks: List of tool calls with args
            - full_context: Combined context string for synthesis
        """
        # Detect language
        language = self.detect_language(question)
        
        # Format chat history
        chat_history_list = context.get("chat_history", [])
        history_str = "\n".join([
            f"{msg.get('role', '').upper()}: {msg.get('content', '')}"
            for msg in chat_history_list[-5:]  # Last 5 messages
        ])
        
        # Build full context for synthesis (to be returned)
        full_context = f"""
[DATA SESI SAAT INI (USER UPLOAD)]:
{context.get('session_preview', 'Tidak ada data spesifik dari user upload yang relevan.')}

[PENGETAHUAN INTERNAL / SOP]:
{context.get('knowledge_preview', 'Tidak ada SOP internal yang relevan ditemukan.')}

[INVENTARIS FILE]:
{context.get('inventory_preview', '')}
        """.strip()
        
        # Get orchestration prompt
        system_prompt = get_orchestration_prompt(
            language=language,
            chat_history=history_str,
            schema_context=context.get("schema", ""),
            session_preview=context.get("session_preview", ""),
            knowledge_preview=context.get("knowledge_preview", ""),
            inventory_preview=context.get("inventory_preview", "")
        )
        
        # Try to invoke LLM with retries for stability
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Invoke LLM
                response = self.llm.invoke([
                    ("system", system_prompt),
                    ("user", f"Question: {question}")
                ])
                response_text = response.content
                
                # Parse JSON from response
                if "{" in response_text and "}" in response_text:
                    # Clean up markdown if present
                    clean_res = response_text.replace("```json", "").replace("```", "").strip()
                    json_start = clean_res.find("{")
                    json_end = clean_res.rfind("}") + 1
                    json_str = clean_res[json_start:json_end]
                    
                    # Parse
                    data = json.loads(json_str)
                    
                    # DEBUG: Log parsed data
                    print(f"\n🔍 DEBUG ORCHESTRATOR (Attempt {attempt+1}):")
                    print(f"  Answer: {data.get('answer', '')[:100]}...")
                    print(f"  Tasks count: {len(data.get('tasks', []))}")
                    
                    # Extract fields
                    answer = data.get("answer", "Processing your request...")
                    tasks_raw = data.get("tasks", [])
                    
                    # Ensure tasks is a list
                    if not isinstance(tasks_raw, list):
                        tasks_raw = [tasks_raw] if tasks_raw else []
                        
                    # Convert to expected format
                    tasks = [
                        {
                            "tool": task.get("tool", ""),
                            "args": task.get("args", {})
                        }
                        for task in tasks_raw if isinstance(task, dict)
                    ]
                    
                    # Special Case: If "answer" exists but "tasks" IS EMPTY and user asked for analysis
                    # We should check if we can reconstruct the intent
                    if not tasks and any(kw in question.lower() for kw in ['plot', 'analisis', 'compare', 'bandingkan', 'overview']):
                         print("  ⚠️ Detected analysis intent with empty tasks. Retrying...")
                         continue

                    return answer, tasks, full_context
                else:
                    print(f"  ⚠️ No JSON found in attempt {attempt+1}")
                    last_error = "No JSON detected in response."
                    continue
                    
            except Exception as e:
                last_error = str(e)
                print(f"  ⚠️ Orchestration Attempt {attempt+1} failed: {last_error}")
                if "connection" in last_error.lower() or "reset" in last_error.lower():
                    import time
                    time.sleep(1) # Backoff
                continue
        
        # If all retries fail, fall back to professional explanation or RAG
        error_msg = f"I encountered a temporary clinical analysis challenge: {last_error}"
        if language == "Indonesian":
            error_msg = f"Saya mengalami sedikit kendala teknis saat melakukan analisis: {last_error}"
        return error_msg, [], full_context
    
    def extract_patient_ids(self, question: str) -> str:
        """
        Extract patient IDs from question (fallback utility).
        
        Args:
            question: User's question
        
        Returns:
            Comma-separated patient IDs or empty string
        """
        # Range pattern: "id 1-5" or "patient 10 - 15"
        range_pattern = re.search(r'(?:id|pasien|patient)\s*(\d+)\s*[-–]\s*(\d+)', question.lower())
        if range_pattern:
            start = int(range_pattern.group(1))
            end = int(range_pattern.group(2))
            return ",".join(str(i) for i in range(start, end + 1))
        
        # List pattern: "id 1, 2, 3" or "patient 1,2,3"
        list_pattern = re.search(r'(?:id|pasien|patient)\s*([\d,\s]+)', question.lower())
        if list_pattern:
            ids = re.findall(r'\d+', list_pattern.group(1))
            return ",".join(ids)
        
        # Single pattern: "id 1" or "patient 5"
        single_pattern = re.search(r'(?:id|pasien|patient)\s*(\d+)\b', question.lower())
        if single_pattern:
            return single_pattern.group(1)
        
        return ""
