


import time
import logging
from typing import List, Tuple, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

from backend.services.llm_service import llm_service 
from backend.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.llm = llm_service.get_llm()
        
        # LangChain's native in-memory store for isolated session tracking
        self.sessions: Dict[str, InMemoryChatMessageHistory] = {}

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are an expert, insightful AI assistant specializing in document analysis. "
            "Your goal is to answer the user's question accurately based on the provided Context.\n\n"
            
            "Guidelines:\n"
            "1. Use flexible, semantic reasoning. Recognize that related concepts, abbreviations, "
            "and tenses refer to the same entity (e.g., understand that 'Intern', 'Internship', and 'Interning' are deeply related).\n"
            "2. Rely on the Context for your factual ground truth. Do not invent completely new facts or external histories outside of it.\n"
            "3. If the Context truly does not discuss or imply the answer to the user's question, state: also refer to context or chat hsitory to ans qustions asked earlier "
            "'I don't have enough information to answer that based on the provided documents.'\n\n"
            
            "Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:

        """Helper to fetch or initialize an isolated session memory store."""

        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]

    def _get_langchain_history(self, session_id: str) -> List[Any]:

        """Fetches LangChain history and applies token trimming optimization."""

        history_store = self._get_session_history(session_id)
        lc_messages = history_store.messages
        
        # Trim history to the last ~4000 tokens so we don't exceed Groq's limits
        if lc_messages:
            try:
                lc_messages = trim_messages(
                    lc_messages,
                    max_tokens=4000, 
                    strategy="last",
                    token_counter=self.llm,
                    start_on="human",
                    include_system=False
                )
            except Exception as e:
                logger.error(f"Failed to trim messages: {e}. Falling back to untrimmed history.")
                
        return lc_messages

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Returns the raw history dictionaries for your router endpoint.
        Maintains your exact upstream dictionary format contract.
        """
        history_store = self._get_session_history(session_id)
        raw_history = []
        
       
        for msg in history_store.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            raw_history.append({
                "role": role,
                "content": msg.content,
                "timestamp": time.time()  # Mocked to keep schema intact
            })
        return raw_history

    def chat(self, question: str, session_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Handles retrieval, history formatting, LLM generation, and memory updates."""
        
      
        print("\n" + "═" * 60)
        print(f" NEW CHAT REQUEST [Session: {session_id}]")
        print("═" * 60)
        print(f" User Question:\n   {question}\n")

        # 1. Retrieve & optimize active Chat History
        chat_history = self._get_langchain_history(session_id)

        # 2. Retrieve Documents using your VectorStoreService
        logger.info(f"Retrieving context for session {session_id} and query: '{question}'")
        docs = vector_store_service.retrieve(query=question)
        
        # Print precise context information directly to the terminal
        print(" CONTEXT RETRIEVED FROM CHROMA DB:")
        print("─" * 60)
        if not docs:
            print("    [No relevant context documents found in ChromaDB]")
        else:
            for i, doc in enumerate(docs, 1):
                source_name = doc.metadata.get("source", "Unknown Source")
                print(f"    [Doc {i}] Source: {source_name}")
                # Print the full raw content of each retrieved chunk
                print(f"   Content:\n   {doc.page_content.strip()}")
                print("   " + "┈" * 40)
        print("─" * 60 + "\n")

        # Format the retrieved documents for the prompt and for the API response
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs]) if docs else "No context found."
        
        # Build the source citations safely from metadata
        sources = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown Source")
            sources.append({
                "source": source_name,
                "content": doc.page_content[:200] + "..." 
            })

        # 3. Build and Invoke the Chain
        chain = self.qa_prompt | self.llm
        
        logger.info("Invoking Groq LLM chain...")
        response = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": question
        })

        answer = response.content

        # Print the LLM response output to terminal
        print(" GROQ LLM RESPONSE:")
        print("─" * 60)
        print(answer.strip())
        print("═" * 60 + "\n")

        # 4. Save the interaction natively to memory using LangChain instances
        history_store = self._get_session_history(session_id)
        history_store.add_user_message(question)
        history_store.add_ai_message(answer)

        return answer, sources

# Instantiate a singleton to import into your router
rag_service = RAGService()