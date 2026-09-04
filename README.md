# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and Sentence Transformers. It ingests PDF documents, chunks and embeds them, stores them in a local vector database, and answers queries by retrieving relevant context and passing it to an LLM (OpenAI, Groq, or Anthropic).

## How it works

```
PDFs -> Documents -> Chunks -> Embeddings -> Vector Store (Chroma)
                                                    |
Query -> Query Embedding -> Semantic Search --------
                                                    |
                                        Context + Query -> LLM -> Answer
```

### 1. Ingestion Pipeline
- **Load** — all PDFs in `data/pdfs/` are loaded via `PyPDFLoader`.
- **Chunk** — documents are split into overlapping chunks with `RecursiveCharacterTextSplitter` (default: 500 chars, 50 overlap).
- **Embed** — chunks are embedded using `sentence-transformers` (`all-MiniLM-L6-v2` by default).
- **Store** — embeddings and chunks are persisted to a local ChromaDB collection (`data/vector_store/`).

### 2. Retrieval Pipeline
- A query is embedded with the same embedding model.
- ChromaDB performs a semantic (cosine similarity) search to return the top-k most relevant chunks, optionally filtered by a similarity score threshold.

### 3. Generation Pipeline
- Retrieved chunks are combined into a context block and passed to an LLM along with the original query.
- Interchangeable LLM backends are supported via LangChain:
  - **OpenAI** (`langchain-openai`)
  - **Groq** (`langchain-groq`)
  - **Anthropic Claude** (`langchain-anthropic`)

## Project structure

```
.
├── data/
│   ├── pdfs/            # Source PDF documents
│   └── vector_store/    # Persisted Chroma vector store
├── RAG_pipeline.ipynb   # Main notebook
└── README.md
```

## Setup

### Requirements
- Python 3.10+
- Jupyter Notebook / JupyterLab

### Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters \
            langchain-openai langchain-groq langchain-anthropic \
            sentence-transformers chromadb scikit-learn pypdf
```

### Configure API keys

**Do not hardcode API keys in the notebook.** Set them as environment variables instead:

```bash
export OPENAI_API_KEY="your-openai-key"
export GROQ_API_KEY="your-groq-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Then in the notebook, read them from the environment, e.g.:

```python
import os
llm = ChatAnthropic(
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
    max_tokens=1024
)
```

### Add your data
Place PDF files in `data/pdfs/`.

## Usage

Open `RAG_pipeline.ipynb` and run the cells in order:

1. Run the **Ingestion Pipeline** section to load, chunk, embed, and store your PDFs.
2. Run the **Retrieval Pipeline** section to test semantic search:
   ```python
   rag_retriever.retrieve("What is RAG?")
   ```
3. Run the **Generation Pipeline** section for your LLM provider of choice to get a full RAG answer:
   ```python
   answer = generate_output("What is RAG?", rag_retriever, llm)
   print(answer)
   ```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `model_name` (embeddings) | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `top_k` | 3–5 | Number of chunks retrieved per query |
| `score_threshold` | 0.0 | Minimum similarity score for retrieved chunks |

## Notes

- The vector store persists locally under `data/vector_store/` — delete this folder to reset the index.
- Swap LLM providers by instantiating a different `llm` object (`ChatOpenAI`, `ChatGroq`, or `ChatAnthropic`); `generate_output` works with any of them.

## License

Add a license of your choice (e.g. MIT).
