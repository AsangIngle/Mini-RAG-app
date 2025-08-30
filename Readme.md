# RAG with Gemini + Qdrant + Cohere Reranker

This project is a **Retrieval-Augmented Generation (RAG)** application built as part of the Predusk AI/ML Intern assessment.  
It combines **Google Gemini** for answer generation, **Qdrant** for vector storage, and **Cohere** for document re-ranking.



##  Features
1.Upload a PDF file, process its text into embeddings.
2.Store embeddings in **Qdrant** (vector database).
3.Retrieve and **re-rank** the most relevant chunks for a query.
4.Generate grounded answers using **Gemini** with inline citations `[1], [2]`.
5.Display retrieved sources for transparency.
6.Show response latency and estimated token usage.



##  Architecture

```mermaid
flowchart TD
    A[User Uploads PDF] --> B[Text Extraction + Chunking]
    B --> C[SentenceTransformer Embeddings (384-d)]
    C --> D[Qdrant Vector DB]
    E[User Query] --> F[Embedding]
    F --> D
    D --> G[Top-k Retrieved Chunks]
    G --> H[Cohere Reranker]
    H --> I[Gemini LLM]
    I --> J[Answer with Citations + Sources]

How it Works

1.Upload a PDF – Text is extracted and split into chunks (1000 chars, 150 overlap).
2.Embedding – Chunks encoded using all-MiniLM-L6-v2 (384-dim).
3.Storage – Embeddings stored in Qdrant.
4.Query – User submits a question.
5.Retrieve + Rerank – Top 10 chunks retrieved → reranked by Cohere → top 3 chosen.
6.Answer – Gemini generates a final answer with inline citations.
7.UI – Streamlit shows answer, sources, response time, and estimated tokens.

Installation

# Clone the repo
git clone https://github.com/your-username/rag-gemini-qdrant.git
cd rag-gemini-qdrant

# Create virtual environment & install deps
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt


Environment Variables

api_key=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
Also provide .env.example with the same keys but empty values.

Usage

Run the Streamlit app:
  streamlit run backend.py

1.Upload a PDF file.
2.Ask a question in the text box.
3.See the answer with citations, sources, latency, and token estimates.

Remarks

Currently supports only PDFs.
Embedding model: all-MiniLM-L6-v2 (384-d).
Retrieval: top-k=10, reranker: Cohere (rerank-english-v3.0), final top-3 passed to Gemini.
Token usage is estimated (1 token ≈ 4 characters).
Free-tier APIs may hit limits → if so, fallback/error messages shown.
Tradeoff: Gemini Flash used for speed & cost efficiency (instead of Pro).

Acceptance Criteria Checklist
Live hosted on Streamlit Cloud.
Query -> retrieved chunks -> reranked -> Gemini answer with citations.
Latency + token estimates displayed.
README with diagram, params, providers, and remarks.

Tradeoffs / Limitations
1.Dataset size: Currently limited to 100 samples for faster testing. Scaling to larger datasets may increase runtime and memory requirements.
2.Model choice: Used a simpler baseline model (e.g., Random Forest / CNN-lite) instead of a heavy model to balance speed vs. accuracy. Accuracy may not be state-of-the-art.
3.Preprocessing: Minimal preprocessing applied. More advanced text/image cleaning could improve results.
4.Deployment: Code runs locally in Colab/VS Code but is not yet production-ready (e.g., no API endpoint, no Dockerization).
5.Hardware dependency: Performance may vary depending on GPU/CPU availability.
6.Generalization: Results tested only on dataset X. May not generalize well to other datasets without fine-tuning.
