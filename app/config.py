
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Chunking settings
    CHUNK_SIZE = 500          # characters per chunk
    CHUNK_OVERLAP = 50        # overlap taaki context na tute

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast + lightweight, free

    # Vector store
    VECTOR_DB_PATH = "data/vector_db/faiss_index"
    UPLOAD_DIR = "data/uploads"

    # LLM
    GROQ_MODEL = "llama-3.1-8b-instant"

settings = Settings()