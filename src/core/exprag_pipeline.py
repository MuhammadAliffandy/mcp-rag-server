"""
EXPRAG Pipeline - Complete 3-Phase Implementation

This module orchestrates the full EXPRAG workflow:
1. Data Structuring: Convert unstructured data to PatientProfile
2. Coarse Ranking: Find similar patients using Jaccard similarity
3. Scoped Retrieval: Query Pinecone with metadata filter on cohort
"""

from typing import List, Dict, Any, Optional
from src.core.patient_profile import PatientProfile
from src.core.similarity import rank_cohort, get_cohort_ids
from src.hub.pinecone_connector import PineconeRetriever


from src.core.clinical_loader import load_clinical_excel

class EXPRAGPipeline:
    """
    Complete EXPRAG workflow orchestrator.
    
    This pipeline implements the 3-phase experience retrieval architecture:
    - Phase 1: Structured comparison base
    - Phase 2: Efficient coarse filtering
    - Phase 3: Precise vector search within cohort
    """
    
    def __init__(self, pinecone_index: str = "medical-records", excel_path: str = "./internal_docs/Test_AI for MES classification_clinical data_20251002.xlsx"):
        """
        Initialize EXPRAG pipeline.
        
        Args:
            pinecone_index: Name of Pinecone index for Phase 3
            excel_path: Path to the experience database (Excel)
        """
        self.pinecone = PineconeRetriever(index_name=pinecone_index)
        self.database: List[PatientProfile] = []
        
        # Auto-load database if file exists
        import os
        if os.path.exists(excel_path):
            self.load_patient_database(load_clinical_excel(excel_path))
    
    def load_patient_database(self, profiles: List[PatientProfile]):
        """
        Load historical patient profiles for comparison.
        """
        self.database = profiles
        print(f"📊 Loaded {len(profiles)} patient profiles into Experience Index")

    def get_hybrid_context(
        self,
        current_patient_data: Dict[str, Any],
        query: str,
        top_k_cohort: int = 5
    ) -> Dict[str, Any]:
        """
        Consolidated method for Hybrid RAG retrieval.
        Retrieves both Peer Experience (Internal) and Knowledge (External).
        """
        # 1. Internal Experience Stream
        profile = self.create_profile_from_dict(current_patient_data)
        cohort_ids = self.find_similar_cohort(profile, top_k=top_k_cohort)
        internal_context = self.retrieve_context(query, cohort_ids)
        
        # 2. External Knowledge Stream (Placeholder - would call RAGEngine)
        # For now, we return a structured dictionary that the Orchestrator can use
        
        return {
            'patient_profile': profile,
            'cohort_ids': cohort_ids,
            'internal_experience': internal_context,
            'query': query
        }

    def create_profile_from_dict(self, data: Dict[str, Any]) -> PatientProfile:
        """
        Phase 1: Convert unstructured data to PatientProfile.
        Maps messy keys to structured clinical metrics.
        """
        # Normalize polyp_history from string to list if needed
        polyp_history = data.get('polyp_history', [])
        if isinstance(polyp_history, str):
            polyp_history = [p.strip() for p in polyp_history.split(',') if p.strip()]
        
        # Normalize procedures from string to list if needed
        procedures = data.get('procedures', [])
        if isinstance(procedures, str):
            procedures = [p.strip() for p in procedures.split(',') if p.strip()]
        
        # Clinical Mapping (Fuzzy identification of keys)
        def find_val(keys):
            for k in keys:
                if k in data: return data[k]
            return None

        profile = PatientProfile(
            case_id=str(data.get('case_id', data.get('patient', 'unknown'))),
            indication=data.get('indication', 'screening'),
            age=find_val(['age', 'age_at_cpy', 'usia']),
            sum_pmayo=find_val(['sum_pmayo', 'mayo', 'mayo_score']),
            dz_location=find_val(['dz_location', 'location', 'lokasi']),
            hb=find_val(['hb', 'hemoglobin', 'hbg']),
            rectal_bleed=find_val(['rectal_bleed', 'bleeding']),
            polyp_history=polyp_history,
            procedures=procedures
        )
        
        return profile
    
    def find_similar_cohort(
        self,
        input_profile: PatientProfile,
        top_k: int = 10,
        min_threshold: float = 0.1
    ) -> List[str]:
        """
        Phase 2: Find Top-K similar patients using Jaccard similarity.
        
        Args:
            input_profile: The current patient's profile
            top_k: Number of similar cases to retrieve
            min_threshold: Minimum similarity threshold
        
        Returns:
            List of case_ids for similar patients
            
        Example:
            >>> cohort_ids = pipeline.find_similar_cohort(current_patient, top_k=5)
            >>> # Returns: ["analysis_id_1", "analysis_id_2", ...]
        """
        if not self.database:
            print("⚠️ Patient database is empty. Load profiles first.")
            return []
        
        cohort_ids = get_cohort_ids(
            input_profile,
            self.database,
            top_k=top_k,
            min_threshold=min_threshold
        )
        
        print(f"🎯 Phase 2 Complete: Found {len(cohort_ids)} similar cases")
        return cohort_ids
    
    def retrieve_context(
        self,
        query: str,
        cohort_ids: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Retrieve relevant context (Pinecone or In-Memory Fallback).
        
        If Pinecone is unavailable, falls back to returning the raw patient 
        data from the in-memory database for the given cohort IDs.
        """
        # Try Pinecone first
        results = self.pinecone.retrieve_scoped_context(
            query=query,
            cohort_ids=cohort_ids,
            top_k=top_k
        )
        
        # Fallback: If Pinecone returned nothing, use in-memory database
        if not results and self.database:
            print("⚠️ Pinecone unavailable, using in-memory patient data fallback.")
            for profile in self.database:
                if profile.case_id in cohort_ids:
                    # Convert profile to structured text context
                    profile_text = (
                        f"Patient {profile.case_id}: "
                        f"Age={profile.age or 'N/A'}, "
                        f"pMayo={profile.sum_pmayo or 'N/A'}, "
                        f"Hb={profile.hb or 'N/A'}, "
                        f"Location={profile.dz_location or 'N/A'}, "
                        f"Bleeding={profile.rectal_bleed or 'N/A'}"
                    )
                    results.append({
                        'case_id': profile.case_id,
                        'score': 1.0,  # Max score for exact ID match
                        'text': profile_text
                    })
            print(f"📚 In-Memory Fallback: Retrieved data for {len(results)} patients.")
        
        print(f"📚 Phase 3 Complete: Retrieved {len(results)} relevant chunks")
        return results

    
    def run_full_pipeline(
        self,
        current_patient_data: Dict[str, Any],
        query: str,
        top_k_cohort: int = 10,
        top_k_chunks: int = 5,
        min_similarity: float = 0.1
    ) -> Dict[str, Any]:
        """
        Execute complete EXPRAG workflow end-to-end.
        
        Args:
            current_patient_data: Unstructured data about current patient
            query: User's question
            top_k_cohort: How many similar patients to find (Phase 2)
            top_k_chunks: How many chunks to retrieve (Phase 3)
            min_similarity: Minimum similarity threshold
        
        Returns:
            Dictionary with results from all 3 phases
            
        Example:
            >>> result = pipeline.run_full_pipeline(
            ...     current_patient_data={
            ...         "case_id": "new_001",
            ...         "indication": "IBD",
            ...         "polyp_history": "tubular",
            ...         "procedures": "colonoscopy"
            ...     },
            ...     query="What is the typical treatment protocol?"
            ... )
            >>> print(result['retrieved_chunks'])
        """
        print("=" * 60)
        print("🚀 Starting EXPRAG Pipeline")
        print("=" * 60)
        
        # Phase 1: Structure the data
        print("\n📋 Phase 1: Data Structuring")
        current_profile = self.create_profile_from_dict(current_patient_data)
        print(f"   ✅ Created profile for: {current_profile.case_id}")
        
        # Phase 2: Coarse ranking
        print("\n🔍 Phase 2: Coarse Ranking")
        cohort_ids = self.find_similar_cohort(
            current_profile,
            top_k=top_k_cohort,
            min_threshold=min_similarity
        )
        
        if not cohort_ids:
            print("   ⚠️ No similar patients found")
            return {
                'profile': current_profile,
                'cohort_ids': [],
                'retrieved_chunks': []
            }
        
        # Phase 3: Scoped retrieval
        print("\n📡 Phase 3: Scoped Vector Retrieval")
        retrieved_chunks = self.retrieve_context(
            query=query,
            cohort_ids=cohort_ids,
            top_k=top_k_chunks
        )
        
        print("\n" + "=" * 60)
        print("✅ EXPRAG Pipeline Complete")
        print("=" * 60)
        
        return {
            'profile': current_profile,
            'cohort_ids': cohort_ids,
            'retrieved_chunks': retrieved_chunks,
            'query': query
        }
