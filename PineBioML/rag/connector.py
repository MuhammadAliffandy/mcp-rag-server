"""
Pinecone Connector for EXPRAG Workflow (Phase 3)

This module handles scoped vector retrieval using Pinecone metadata filtering.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class PineconeRetriever:
    """
    Handles Pinecone connection and scoped retrieval with metadata filtering.
    
    This connector enables the final phase of EXPRAG: querying the vector DB
    with a pre-filtered cohort of similar patients.
    """
    
    def __init__(self, index_name: str = "medical-records"):
        """
        Initialize Pinecone retriever.
        
        Args:
            index_name: Name of the Pinecone index to query
        """
        self.index_name = index_name
        self.index = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize connection to Pinecone.
        
        Requires environment variables:
        - PINECONE_API_KEY
        - PINECONE_ENV (e.g., "us-west1-gcp")
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENV", "us-west1-gcp")
            
            if not api_key:
                print("⚠️ PINECONE_API_KEY not found in environment. Pinecone retrieval disabled.")
                return False
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists, create if not
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"📌 Creating Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=environment
                    )
                )
            
            # Connect to index
            self.index = pc.Index(self.index_name)
            self._initialized = True
            
            print(f"✅ Pinecone retriever initialized: {self.index_name}")
            return True
            
        except ImportError:
            print("⚠️ Pinecone library not installed. Run: pip install pinecone-client")
            return False
        except Exception as e:
            print(f"❌ Pinecone initialization error: {e}")
            return False
    
    def retrieve_scoped_context(
        self,
        query: str,
        cohort_ids: List[str],
        top_k: int = 5,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context using scoped metadata filtering.
        
        This is the core EXPRAG Phase 3 function. It queries Pinecone with
        a filter that restricts results to only the pre-selected cohort IDs.
        
        Args:
            query: User's natural language question
            cohort_ids: List of case_ids from Phase 2 ranking
            top_k: Number of relevant chunks to retrieve
            namespace: Optional Pinecone namespace
        
        Returns:
            List of retrieved documents with metadata
            
        Example:
            >>> cohort = ["analysis_id_1", "analysis_id_2"]
            >>> results = retriever.retrieve_scoped_context(
            ...     "What were the CRP levels?",
            ...     cohort_ids=cohort
            ... )
            >>> # Only searches within the specified cohort
        
        Implementation Note:
            Uses Pinecone's metadata filtering syntax:
            filter={'case_id': {'$in': cohort_ids}}
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        if not cohort_ids:
            print("⚠️ No cohort IDs provided. Returning empty results.")
            return []
        
        try:
            # Get query embedding (using OpenAI)
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            query_vector = embeddings.embed_query(query)
            
            # Construct metadata filter for cohort
            # CRITICAL: This is the EXPRAG scoped retrieval filter
            filter_dict = {
                'case_id': {'$in': cohort_ids}
            }
            
            print(f"🔍 Querying Pinecone with filter: {filter_dict}")
            
            # Query Pinecone with filter
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                namespace=namespace
            )
            
            # Parse results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'text': match.get('metadata', {}).get('text', ''),
                    'case_id': match.get('metadata', {}).get('case_id'),
                    'metadata': match.get('metadata', {})
                })
            
            print(f"✅ Retrieved {len(matches)} scoped results from {len(cohort_ids)} cohort cases")
            return matches
            
        except Exception as e:
            print(f"❌ Scoped retrieval error: {e}")
            return []
    
    def upsert_patient_record(
        self,
        case_id: str,
        text_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> bool:
        """
        Upsert patient record chunks into Pinecone.
        
        This function is for data ingestion. Each chunk gets tagged with
        the case_id metadata for later filtered retrieval.
        
        Args:
            case_id: Unique patient case identifier
            text_chunks: List of text chunks from patient record
            metadata: Additional metadata to attach
            namespace: Optional Pinecone namespace
        
        Returns:
            True if successful
        """
        if not self._initialized:
            if not self.initialize():
                return False
        
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
            
            vectors = []
            base_metadata = metadata or {}
            base_metadata['case_id'] = case_id
            
            for i, chunk in enumerate(text_chunks):
                vector = embeddings.embed_query(chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata['text'] = chunk
                chunk_metadata['chunk_index'] = i
                
                vectors.append({
                    'id': f"{case_id}_chunk_{i}",
                    'values': vector,
                    'metadata': chunk_metadata
                })
            
            # Batch upsert
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"✅ Upserted {len(vectors)} chunks for case {case_id}")
            return True
            
        except Exception as e:
            print(f"❌ Upsert error: {e}")
            return False
