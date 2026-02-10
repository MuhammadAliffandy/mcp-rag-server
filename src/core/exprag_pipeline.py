from typing import List, Dict, Any, Optional
import pandas as pd
from src.core.patient_profile import PatientProfile
from src.core.similarity import get_cohort_ids
from src.hub.pinecone_connector import PineconeRetriever
from src.core.prompt_engine import EXPRAGPromptEngine
from src.core.clinical_loader import load_patient_data
from src.hub.database import EHRDatabase


class EXPRAGPipeline:
    """
    Complete EXPRAG workflow orchestrator (Advanced 2.0).
    
    Implements:
    - Multi-Axis Similarity
    - Quintile Diversity Binning
    - SQL/CSV Record Persistence
    - Dual RAG Modes (Combine/Separate)
    """
    
    def __init__(
        self, 
        pinecone_index: str = "medical-records", 
        excel_path: str = "./internal_docs/Test_AI for MES classification_clinical data_20251002.xlsx",
        db_path: str = "./internal_docs/mimic_iv_records.db",
        rag_mode: str = "combine",
        use_diversity: bool = True
    ):
        self.pinecone = PineconeRetriever(index_name=pinecone_index)
        self.database: EHRDatabase = EHRDatabase(db_path)
        self.profiles: List[PatientProfile] = []
        self.prompt_engine = EXPRAGPromptEngine()
        
        # Ablation / Configuration Flags
        self.rag_mode = rag_mode
        self.use_diversity = use_diversity
        
        # Auto-load profile metadata if file exists
        import os
        if os.path.exists(excel_path):
            self.load_profiles(load_patient_data(excel_path))
    
    def load_profiles(self, profiles: List[PatientProfile]):
        """Load profile metadata for similarity comparison."""
        self.profiles = profiles
        print(f"📊 Experience Base: {len(profiles)} patient profiles registered.")

    def find_similar_cohort(
        self,
        input_profile: PatientProfile,
        top_k: int = 10
    ) -> List[str]:
        """Phase 2: Find similar patients with optional Diversity Binning."""
        return get_cohort_ids(
            input_profile,
            self.profiles,
            top_k=top_k,
            use_diversity=self.use_diversity
        )
    
    def get_structured_context(
        self, 
        query: str, 
        cohort_ids: List[str], 
        top_k_chunks: int = 3
    ) -> str:
        """
        Phase 3: Scoped Retrieval & Formatting.
        - combine: Standard RAG across all cohort chunks.
        - separate: Fetches full records from EHRDatabase for a patient-by-patient overview.
        """
        if self.rag_mode == "separate":
            # Fetch full patient summaries from EHRDatabase
            summaries = []
            for pid in cohort_ids:
                record = self.database.get_full_record(pid)
                if record:
                    summaries.append({'case_id': pid, 'text': record[:2000] + "..."}) # Snippet
            return self.prompt_engine.build_experience_block(summaries)
        
        # Combine mode: Scoped Vector Search
        raw_results = self.pinecone.retrieve_scoped_context(
            query=query,
            cohort_ids=cohort_ids,
            top_k=top_k_chunks
        )
        
        if not raw_results:
            # Fallback to EHRDatabase full records if Pinecone fails
            fallback_summaries = []
            for pid in cohort_ids:
                record = self.database.get_full_record(pid)
                if record:
                    fallback_summaries.append({'case_id': pid, 'text': record[:1000]})
            return self.prompt_engine.build_experience_block(fallback_summaries)
            
        return self.prompt_engine.build_experience_block(raw_results)

    def run_full_pipeline(
        self,
        current_patient_data: Dict[str, Any],
        query: str,
        options: Optional[List[str]] = None,
        top_k_cohort: int = 10
    ) -> Dict[str, Any]:
        """Full Advanced EXPRAG execution."""
        print(f"🚀 Running FULL EXPRAG (Diversity: {self.use_diversity}, Mode: {self.rag_mode})")
        
        # 1. Profile Generation
        current_profile = self.create_profile_from_dict(current_patient_data)
        
        # 2. Diversity Ranking (Phase 2)
        cohort_ids = self.find_similar_cohort(current_profile, top_k=top_k_cohort)
        
        # 3. Scoped Context (Phase 3)
        context = self.get_structured_context(query, cohort_ids)
        
        # 4. Prompt Synthesis (Phase 4)
        final_prompt = self.prompt_engine.build_qa_prompt(
            profile_dict=current_profile.to_comparison_dict(),
            question=query,
            experience_context=context,
            options=options
        )
        
        return {
            'prompt': final_prompt,
            'cohort_ids': cohort_ids,
            'profile': current_profile,
            'mode': self.rag_mode,
            'context': context
        }

    def execute_clinical_qa(
        self,
        current_patient_data: Dict[str, Any],
        query: str,
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full EXPRAG pipeline and performs the final LLM reasoning.
        Returns structured result with REASON and ANSWER.
        """
        from langchain_openai import ChatOpenAI
        
        # 1. Generate the overpowered prompt
        pipeline_res = self.run_full_pipeline(current_patient_data, query, options)
        full_prompt = pipeline_res['prompt']
        
        # 2. Call LLM (Mirroring EXPRAG strict temperature)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        response = llm.invoke(full_prompt).content
        
        # 3. Parse formatted output
        parsed = self.prompt_engine.extract_reason_answer(response)
        
        return {
            "reason": parsed['reason'],
            "answer": parsed['answer'],
            "full_response": response,
            "cohort_ids": pipeline_res['cohort_ids'],
            "profile": pipeline_res['profile']
        }

    def create_profile_from_dict(self, data: Dict[str, Any]) -> PatientProfile:
        """Phase 1: Robust structure mapping."""
        def find_val(keys):
            for k in keys:
                if k in data and not pd.isna(data[k]): return data[k]
            return None

        # Normalize list fields
        def to_list(val):
            if isinstance(val, str):
                return [v.strip() for v in val.split(',') if v.strip()]
            return val if isinstance(val, list) else []

        return PatientProfile(
            case_id=str(data.get('case_id', data.get('patient', 'unknown'))),
            indication=str(data.get('indication', 'screening')).lower(),
            age=find_val(['age', 'age_at_cpy', 'usia']),
            sum_pmayo=find_val(['sum_pmayo', 'mayo', 'mayo_score']),
            dz_location=find_val(['dz_location', 'location', 'lokasi']),
            hb=find_val(['hb', 'hemoglobin', 'hbg']),
            rectal_bleed=find_val(['rectal_bleed', 'bleeding']),
            polyp_history=to_list(data.get('polyp_history', [])),
            procedures=to_list(data.get('procedures', ['colonoscopy']))
        )
