"""
Prompt Engine for EXPRAG Workflow
Provides structured templates and context formatting for clinical QA.
"""

from typing import List, Dict, Any, Optional

class EXPRAGPromptEngine:
    """
    Orchestrates specialized clinical prompts with the strict 
    REASON/ANSWER format from EXPRAG research.
    """
    
    CLINICAL_QA_TEMPLATE = """
Answer a question for a target patient, who has background:
{background}

Question: {question}

{options_block}

You have some EHR data from other most similar patients as that patient (historical experience):
{experience_context}

Your solution should have only 2 parts: "REASON" and "ANSWER", and start from "REASON".
The "REASON" part should be less than 50 words and be your reasoning process about why you make that choice instead of just copying it.
The "ANSWER" part should be the option letters only.

"REASON": [Your clinical reasoning]
"ANSWER": [Option letter(s)]
""".strip()

    SUMMARY_TEMPLATE = """
Please retrieve all relevant information as a summary from the clinical record below.
Focus on findings that would help answer the specific question.

Question: {question}

Clinical Record:
{record_text}

Summary:
""".strip()

    @staticmethod
    def format_background(profile_dict: Dict[str, Any]) -> str:
        """Format a patient profile into a clinical background string."""
        parts = []
        if profile_dict.get('age'): parts.append(f"{profile_dict['age']}-year-old")
        if profile_dict.get('indication'): parts.append(f"with {profile_dict['indication']}")
        
        base = "Patient " + " ".join(parts)
        metrics = []
        if profile_dict.get('sum_pmayo'): metrics.append(f"pMayo score of {profile_dict['sum_pmayo']}")
        if profile_dict.get('hb'): metrics.append(f"Hb level of {profile_dict['hb']}")
        
        if metrics:
            return base + ", presenting with " + " and ".join(metrics) + "."
        return base + "."

    @staticmethod
    def format_options(options: List[str]) -> str:
        """Format multiple choice options (A, B, C...)."""
        if not options: return ""
        formatted = ["Choices:"]
        for i, opt in enumerate(options):
            letter = chr(65 + i) # A, B, C...
            formatted.append(f"{letter}: {opt}")
        return "\n".join(formatted)

    def build_qa_prompt(
        self, 
        profile_dict: Dict[str, Any], 
        question: str, 
        experience_context: str,
        options: Optional[List[str]] = None
    ) -> str:
        """Construct the final high-powered EXPRAG prompt."""
        background = self.format_background(profile_dict)
        options_block = self.format_options(options) if options else ""
        
        return self.CLINICAL_QA_TEMPLATE.format(
            background=background,
            question=question,
            options_block=options_block,
            experience_context=experience_context
        )

    def build_experience_block(self, cohort_summaries: List[Dict[str, Any]]) -> str:
        """
        Format the dervied experience from similar patients.
        Supports both 'combine' (raw chunks) and 'separate' (summaries).
        """
        formatted = []
        for i, item in enumerate(cohort_summaries):
            pid = item.get('case_id', 'Unknown')
            text = item.get('text', '')
            formatted.append(f"--- Experience Patient {i+1} (ID: {pid}) ---\n{text}")
        
        return "\n\n".join(formatted)

    def extract_reason_answer(self, llm_response: str) -> Dict[str, str]:
        """
        Parse the strict EXPRAG format:
        "REASON": ...
        "ANSWER": ...
        """
        reason_match = re.search(r'"REASON":\s*(.*?)(?="ANSWER"|$)', llm_response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'"ANSWER":\s*(.*)', llm_response, re.IGNORECASE)
        
        return {
            "reason": reason_match.group(1).strip() if reason_match else "No reason found.",
            "answer": answer_match.group(1).strip() if answer_match else "No answer found."
        }

# Add missing import for re
import re
