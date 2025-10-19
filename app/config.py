import os
from dotenv import load_dotenv

load_dotenv()

# Ollama model name (local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# FAISS index path
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_index")

# Embedding model (sentence-transformers)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama server URL (default)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
