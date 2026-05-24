import logging
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from backend.config.settings import settings
from backend.services.embedding_service import embeddings_service

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        """Initialize ChromaDB with local persistence and optimized distance metrics."""
        print("=" * 50)
        print(" INITIALIZING VectorStoreService (GEMINI POWERED)...")
        print("=" * 50)
        print(f" ChromaDB path: {settings.chroma_db_path}")
        print(f" Collection name: {settings.chroma_collection_name}")
        print("-" * 50)
        
        logger.info(f"Initializing ChromaDB at {settings.chroma_db_path}...")
        print(" Creating ChromaDB instance...")
        
        # Enforcing Cosine matching spacing directly on instantiation
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings_service.get_embedder(),
            persist_directory=settings.chroma_db_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        print(" ChromaDB initialized successfully with Cosine Similarity!")
        print("=" * 50 + "\n")

    def add_documents(self, chunks: List[Document]) -> int:
        """
        Embed and store document chunks in ChromaDB.
        Returns the number of chunks successfully stored.


        """

        clean_chunks = [c for c in chunks if c.page_content.strip()]
    
        if len(clean_chunks) != len(chunks):
            print(f" Warning: Dropped {len(chunks) - len(clean_chunks)} empty chunks")
        print("\n" + "=" * 50)
        print(" ADD_DOCUMENTS function called")
        print("=" * 50)
        print(f" Received {len(chunks)} chunks to store")
        
        try:
            logger.info(f"Storing {len(chunks)} chunks in ChromaDB...")
            print(" Running Gemini processing & caching embeddings in ChromaDB...")
            
            ids = self.vector_store.add_documents(documents=chunks)
            
            print(f" Successfully stored {len(ids)} chunks")
            print(f" Generated IDs: {ids[:5]}{'...' if len(ids) > 5 else ''}") 
            print(" Document added to ChromaDB")
            logger.info(f"Successfully stored {len(ids)} chunks.")
            
            print("=" * 50 + "\n")
            return len(ids)
        
        except Exception as e:
            print(f" ERROR: Failed to store documents: {e}")
            logger.error(f"Failed to store documents in ChromaDB: {e}")
            print("=" * 50 + "\n")
            raise

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Perform a similarity search to find chunks relevant to the query.
        """
        print("\n" + "=" * 50)
        print(" RETRIEVE function called")
        print("=" * 50)
        print(f" Query: '{query}'")
        
        k = top_k or settings.top_k_results
        print(f" Top K value: {k}")
        
        logger.info(f"Searching for top {k} matches for query: '{query}'")
        print(" Performing Gemini vector space similarity search...")
        
        try:
            # We track math distance alongside text content extraction
            results_with_scores = self.vector_store.similarity_search_with_score(query=query, k=k)
            print(f" Found {len(results_with_scores)} relevant documents")
            
            results = []
            for i, (doc, score) in enumerate(results_with_scores, 1):
                content_preview = doc.page_content[:90].replace('\n', ' ') + "..."
                # Score tracking: closer to 0.0 means an exact semantic match
                print(f"    Result {i} [Distance Score: {score:.4f}]: {content_preview}")
                results.append(doc)
            
            print("=" * 50 + "\n")
            return results
            
        except Exception as e:
            print(f" ERROR: Search failed: {e}")
            logger.error(f"Search failed: {e}")
            print("=" * 50 + "\n")
            return []

    def get_retriever(self):
        """
        Returns a LangChain Retriever object.
        """
        print("\n" + "=" * 50)
        print(" GET_RETRIEVER function called")
        print("=" * 50)
        print(" Creating retriever from vector store...")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k_results}
        )
        
        print(f" Retriever created with top_k={settings.top_k_results}")
        print("=" * 50 + "\n")
        
        return retriever

print("\n Creating VectorStoreService singleton...\n")
vector_store_service = VectorStoreService()
print(" VectorStoreService is ready to use!\n")