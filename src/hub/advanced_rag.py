import os
from typing import List, Optional, Tuple
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

class AdvancedRAGTool:
    """
    Advanced Medical RAG Engine using LlamaIndex.
    Implements Sentence Window and Auto-Merging retrieval for medical precision.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", top_k: int = 6):
        self.top_k = top_k
        self.llm = OpenAI(model=model_name, temperature=0)
        
        # Default to OpenAI embeddings, but support local HuggingFace if requested
        self.embed_model = OpenAIEmbedding()
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.documents = []
        self.index = None

    def load_from_directory(self, directory_path: str):
        """Loads all documents from a directory."""
        if os.path.exists(directory_path):
            self.documents = SimpleDirectoryReader(directory_path).load_data()

    def load_from_files(self, file_paths: List[str]):
        """Loads documents from a list of specific file paths."""
        valid_paths = [p for p in file_paths if os.path.exists(p)]
        if valid_paths:
            self.documents = SimpleDirectoryReader(input_files=valid_paths).load_data()

    def query_sentence_window(self, query: str) -> Tuple[str, List[Any]]:
        """
        Retrieves context using Sentence Window technique.
        Injects neighboring sentences for better clinical context.
        """
        if not self.documents:
            return "No documents loaded for retrieval.", []

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        nodes = node_parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        
        response = query_engine.query(query)
        return str(response), response.source_nodes

    def query_auto_merging(self, query: str) -> Tuple[str, List[Any]]:
        """
        Retrieves context using Auto-Merging (Hierarchical) technique.
        Merges granular chunks into larger contexts if similarity is high.
        """
        if not self.documents:
            return "No documents loaded for retrieval.", []

        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        nodes = node_parser.get_nodes_from_documents(self.documents)
        leaf_nodes = get_leaf_nodes(nodes)
        
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        
        self.index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        
        base_retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
        
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine.from_args(retriever)
        
        response = query_engine.query(query)
        return str(response), response.source_nodes

    def run_standard_rag(self, query: str) -> Tuple[str, List[Any]]:
        """Standard Vector retrieval for comparison."""
        if not self.documents:
            return "No documents loaded.", []
            
        if not self.index:
            self.index = VectorStoreIndex.from_documents(self.documents)
            
        query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)
        response = query_engine.query(query)
        return str(response), response.source_nodes
