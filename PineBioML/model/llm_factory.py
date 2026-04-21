import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """
    Centralized factory for instantiating the LLM. 
    Supports gracefully falling back to a local Ollama instance 
    if the OpenAI quota is exhausted, configured via .env.
    """
    # Ensure env variables are loaded
    load_dotenv()

    # Determine provider (default to openai)
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "ollama":
        # Fallback to local Ollama via OpenAI compatibility layer
        local_model = os.getenv("OLLAMA_MODEL", "llama3:8b").strip()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").strip()
        
        return ChatOpenAI(
            model_name=local_model,
            temperature=temperature,
            api_key="ollama", # placeholder needed for the client
            base_url=base_url
        )
    else:
        # Default behavior: OpenAI standard models
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )

def get_embeddings():
    """
    Centralized factory for instantiating Embeddings.
    Defaults to OpenAIEmbeddings, but falls back to OllamaEmbeddings if configured.
    """
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        local_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
        # Note: OllamaEmbeddings uses base_url instead of openai's target.
        # It defaults to http://localhost:11434 but can be changed.
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace("/v1", "")
        return OllamaEmbeddings(model=local_model, base_url=base_url)
    else:
        return OpenAIEmbeddings()
