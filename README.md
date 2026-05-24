# 📚 RAG Document Chatbot

An RAG chatbot engine featuring  document ingestion via **Docling**, optimized vector space persistence with **ChromaDB**, **Google Gemini** semantic embeddings, and ultra-fast generation logic powered by **Groq (Llama 3.1)**. Conversational states are managed dynamically per user session using LangChain's native memory structures.

---

## 🛠️ Architecture & Data Pipeline Blueprint

The pipeline executes sequentially across four dedicated stages to convert raw documents into context-aware conversational insights:

```
[ Raw Document ] (.pdf, .docx, .txt)
       │
       ▼
[ Docling Parser ] ──► Extracts layout-aware, semantically rich text & structures
       │
       ▼
[ Chunking & Clean ] ──► Dynamic text splitting (Size: 500, Overlap: 50)
       │
       ▼
[ Gemini Embedding ] ──► Computes dense vectors via models/text-embedding-004
       │
       ▼
[ ChromaDB Cache ] ──► Persists vectors locally using Cosine Similarity metrics
       │
       ▼
[ Context Retrieval ] ──► Extracts top-K relevant chunks matching user queries
       │
       ▼
[ Groq LLM Inference ] ──► Combines user query, context, and InMemory history to reply
```

**Semantic Ingestion & Extraction:** Raw unstructured data inputs are handled by Docling. Instead of stripping documents down to flat strings, Docling preserves rich structural layouts, table matrices, headers, and relative text hierarchies.

**Chunking & Serialization:** Text structures are standardized, scrubbed, and passed through programmatic text splitters to produce overlapping text chunks designed to prevent mid-sentence semantic clipping.

**Gemini Vector Mapping:** Chunks are systematically vectorized via the `models/text-embedding-004` (or `gemini-embedding-2`) foundational model array, mapping textual data into multidimensional coordinate points.

**ChromaDB Target Storage:** Generated coordinate vectors are written to disk inside a local `chroma_db` catalog set strictly to use Cosine Similarity (`hnsw:space: cosine`) distance metrics.

**Stateful Chat Generation:** Incoming user questions initiate a vector search against ChromaDB. The top matching text chunk is isolated, formatted along with the session's historical timeline managed via `InMemoryChatMessageHistory`, and dispatched to the Groq Llama-3.1 infrastructure layer for zero-shot synthesis.

---

## 🚀 Getting Started & Installation

### 1. Clone the Repository

Open a terminal workspace on your local machine and pull down the codebase:

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Configure Your Virtual Environment

Initialize a fresh isolated environment profile and install all downstream library dependencies listed in your system manifest:

**On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```



### 3. Setup Your Environment Secrets (`.env`)

Create a new file named `.env` in the absolute root directory of the project and drop your private API credential pairs inside. Ensure there are no surrounding whitespace blocks on either side of the configuration keys:

```env


# LLM Config (Groq Cloud Engine)
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant

# Embeddings Config (Google Gemini Provider)
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL_NAME=models/text-embedding-004
```

> ⚠️ **Security Warning:** Never commit your `.env` file containing functional cryptographic secrets to public GitHub repositories. Keep it listed safely within your `.gitignore` configuration rules.

---

## 🏎️ Running the Application

This system uses a decoupled client-server architecture. Run both services concurrently inside separate terminal processes.

### Step A: Initialize the FastAPI Execution Engine (Backend)

Navigate to the root project workspace directory and execute the core application server over local loopback interfaces using Uvicorn:

```bash
uvicorn backend.main:app --reload
```

- Backend available at: `http://localhost:8000`
- Interactive API docs at: `http://localhost:8000/docs`

### Step B: Launch the Flask Presentation Layer (Frontend)

Open a secondary terminal session, navigate to the client folder, and start the frontend proxy:

```bash
# Navigate to the client UI folder
cd app/frontend

# Boot the web application interface
python app.py
```

- Frontend available at: `http://localhost:5000`

---

## 📬 API Route Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Validates, splits, embeds, and saves file payloads to the local vector index. |
| `POST` | `/api/v1/chat` | Evaluates user input against a specific `session_id` to yield answers and source citations. |
| `GET` | `/api/v1/history/{id}` | Fetches all prior user-assistant interactions for a specific session. |
