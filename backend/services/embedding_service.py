import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self):
        """Initialize the high-performance Gemini foundational embedding model."""

        logger.info(f"Loading Google Gemini Embedding Model: {settings.embedding_model}")
        print("\n" + "." * 25)
        print(" INITIALIZING GEMINI EMBEDDING SERVICE")
        print("." * 25)
        
       
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model, 
            google_api_key=settings.gemini_api_key,
            
        )
        print(f" Gemini Embeddings ({settings.embedding_model}) ready.")
        print("-" * 25 + "\n")

    def get_embedder(self):
        """Returns the configured embedding instance for LangChain."""
        return self.embeddings


embeddings_service = EmbeddingsService()