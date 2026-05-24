from langchain_groq import ChatGroq
from backend.config.settings import settings

class LLMService:
    def __init__(self):
        # Initialize Groq with zero temperature for factual consistency
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=0.3, 
            max_retries=2,
        )

    def get_llm(self) -> ChatGroq:
        """Returns the configured ChatGroq instance."""
        return self.llm

llm_service = LLMService()