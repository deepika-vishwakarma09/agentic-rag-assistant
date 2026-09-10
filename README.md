# Agentic RAG Assistant

An intelligent assistant that dynamically routes questions between **document retrieval** and **live web search**, powered by a LangGraph agent, hybrid search, and self-correcting answer generation.

Upload a PDF and ask questions — the system decides on its own whether to answer from the document or search the web, then verifies its own answer before responding.

---

## Why this project

Most RAG (Retrieval-Augmented Generation) tutorials stop at "embed and retrieve." This project goes further by combining several production-grade techniques into a single agentic pipeline:

- **Hybrid retrieval** (semantic + keyword search) instead of relying on embeddings alone
- **Cross-encoder reranking** for higher precision on the final context
- **Agentic routing** — the system decides *where* to look for an answer, not just how
- **Self-correction** — the agent checks its own answer against the retrieved context and retries if it detects a hallucination
- **RAGAS evaluation** — the pipeline is measured with real metrics (faithfulness, relevancy, precision), not just eyeballed
- **Observability** — every query is traced (route taken, latency, correction status) for debugging

---

## Architecture

```
                         ┌─────────────────────┐
                         │   User Question      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Router Agent         │
                         │   (LangGraph + LLM)    │
                         │  "document" or "web"?  │
                         └──────────┬───────────┘
                    ┌───────────────┴────────────────┐
                    │                                  │
         ┌──────────▼──────────┐          ┌───────────▼───────────┐
         │  Document Retrieval   │          │    Web Search           │
         │  ┌─────────────────┐ │          │    (Tavily API)         │
         │  │ BM25 (keyword)   │ │          └───────────┬───────────┘
         │  │ FAISS (semantic) │ │                      │
         │  │      ↓ merge     │ │                      │
         │  │ Cross-Encoder    │ │                      │
         │  │   Reranker       │ │                      │
         │  └─────────────────┘ │                      │
         └──────────┬──────────┘                      │
                    └───────────────┬──────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Answer Generation     │
                         │  (Groq LLaMA 3.1)      │
                         │  + Conversation Memory │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Self-Correction       │
                         │  Verify → Retry if     │
                         │  hallucination detected│
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Answer + Sources      │
                         │  (traced & logged)     │
                         └───────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Semantic search | FAISS + Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Keyword search | BM25 (`rank-bm25`) |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Web search | Tavily API |
| Evaluation | RAGAS |
| PDF parsing | PyMuPDF |
| Frontend | Streamlit |
| Backend | FastAPI-ready modular structure |

---

## Features

- **PDF upload & ingestion** — parses, chunks, and embeds any PDF document
- **Hybrid search** — combines BM25 keyword matching with FAISS semantic search
- **Cross-encoder reranking** — a second-stage precision pass on retrieved chunks
- **Agentic routing** — LangGraph decides document vs. live web search per query
- **Live web search fallback** — via Tavily, for questions outside the document
- **Conversation memory** — understands follow-up questions ("what about that?")
- **Self-correction** — detects and fixes hallucinated answers automatically
- **RAGAS evaluation** — measured faithfulness, relevancy, and context precision
- **Observability** — every query is logged with route, timing, and correction status
- **Streamlit chat UI** — with source citations and route indicators

---

## Evaluation results

Measured using [RAGAS](https://github.com/explodinggradients/ragas) on a sample test set:

| Metric | Score |
|---|---|
| Faithfulness | 0.75 |
| Answer Relevancy | 0.82 |
| Context Precision | 1.00 |

*Faithfulness measures whether answers are grounded in retrieved context (no hallucination). Context Precision measures whether retrieved chunks were actually relevant.*

---

## Project structure

```
agentic-rag-assistant/
├── app/
│   ├── ingestion/       # PDF parsing, chunking, embeddings
│   ├── retrieval/       # FAISS, BM25, reranking
│   ├── agents/          # LangGraph router, web search, self-correction
│   ├── generation/      # LLM client, prompts
│   ├── memory/          # Conversation history
│   ├── evaluation/      # RAGAS evaluation pipeline
│   ├── observability/   # Query tracing
│   └── config.py
├── frontend/
│   └── streamlit_app.py # Chat UI
├── data/                # Vector DB + traces (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/deepika-vishwakarma09/agentic-rag-assistant.git
   cd agentic-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add API keys**

   Copy `.env.example` to `.env` and fill in:
   ```
   GROQ_API_KEY=your_key_here      # free at console.groq.com
   TAVILY_API_KEY=your_key_here    # free tier at tavily.com
   ```

4. **Run the app**
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

5. **Run evaluation** (optional)
   ```bash
   python -m app.evaluation.ragas_eval
   ```

---

## What I learned building this

- How to combine sparse (BM25) and dense (embedding) retrieval, and why neither alone is enough
- Why a second reranking stage meaningfully improves precision over single-stage retrieval
- How to design an agent's control flow as a graph (LangGraph) rather than a linear chain
- How to measure a RAG system quantitatively (RAGAS) instead of relying on manual spot-checks
- Why self-correction loops are a practical, low-cost way to catch hallucinations before they reach the user

---

## Future work

- Multi-turn evaluation (measuring quality across full conversations, not just single turns)
- Swap local JSON tracing for a proper observability backend (Langfuse/LangSmith)
- Support for multiple simultaneous documents with per-document filtering
- Docker containerization for one-command deployment

---

## Author

**Deepika Vishwakarma** — M.Sc. AI/ML, IIIT Lucknow
[GitHub](https://github.com/deepika-vishwakarma09) 
