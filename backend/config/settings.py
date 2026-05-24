from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    llm_model: str = Field(default="llama-3.1-8b-instant", env="LLM_MODEL")

    # Gemini Integration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    embedding_model: str = Field(default="models/gemini-embedding-001", env="EMBEDDING_MODEL_NAME")

    

    # Vector Store
    chroma_db_path: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="rag_documents", env="CHROMA_COLLECTION_NAME")

    # File Storage
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    allowed_extensions: str = Field(default=".pdf,.docx,.txt", env="ALLOWED_EXTENSIONS")

    # Chunking
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")

    # Retrieval
    top_k_results: int = Field(default=3, env="TOP_K_RESULTS")

    # Backend
    backend_url: str = Field(default="http://localhost:8000", env="BACKEND_URL")

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    def ensure_dirs(self):
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = os.path.join(ROOT_DIR, ".env")
        env_file_encoding = 'utf-8'
        extra = "ignore"


settings = Settings()
settings.ensure_dirs()
