# DocMind — RAG Chat Over Your Documents

Ask questions about your own files — PDFs, Word docs, spreadsheets, CSVs, JSON — using local embeddings for retrieval and a Groq-hosted LLM for the actual answers. Runs entirely on your machine except for the LLM call itself.

## How it works

1. Your files are loaded and split into chunks (`src/data_loader.py`, `src/embedding.py`)
2. Chunks are embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in a FAISS index (`src/vectorstore.py`)
3. When you ask a question, the most relevant chunks are retrieved and passed as context to a Groq-hosted LLM, which generates the answer (`src/search.py`)
4. Everything is wrapped in a Streamlit chat interface (`app.py`)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.example .env
```

Edit `.env` and paste your key from [console.groq.com](https://console.groq.com). This file is git-ignored — never commit it.

### 3. Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## How to use

1. Upload one or more files in the **Knowledge Base** panel
2. Click **Build knowledge base** (only needed once per set of files, or again if you add new ones)
3. Ask your question in the chat box pinned to the bottom of the screen

## Supported file types

PDF, TXT, CSV, Excel (`.xlsx`), Word (`.docx`), JSON

Note: structured spreadsheet data (rows/columns meant to be filtered, summed, or joined) isn't a great fit for this retrieval approach — it works best for narrative or free-text content. See the FAQ below.

## Project structure

```
rag-ready/
├── app.py                 # Streamlit UI
├── src/
│   ├── data_loader.py      # Loads PDFs, Word, Excel, CSV, JSON, TXT
│   ├── embedding.py        # Chunking + embedding pipeline
│   ├── vectorstore.py      # FAISS index build/save/load/query
│   └── search.py           # Retrieval + LLM answer generation
├── requirements.txt
├── .env.example
└── .gitignore
```

## Models used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — runs locally, no API key needed
- **LLM**: `openai/gpt-oss-20b` via Groq (default; configurable via the `llm_model` argument in `RAGSearch`). Groq periodically deprecates models — check [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) if you hit a `model_decommissioned` error.

## FAQ

**Can I point this at Excel data instead of documents?**
Yes for narrative-style sheets (reports, notes). For structured/tabular queries ("sum column C", "filter by region"), a dataframe or SQL-agent approach is a better fit than embeddings-based retrieval.

**Can this read from a live database (e.g. Firebase)?**
Not out of the box — `data_loader.py` currently reads local files. Swapping in a Firestore/Realtime DB fetch as an additional loader is straightforward for text-heavy collections; structured records are better served by a query-generation approach than by embedding rows.

## License

Add your license of choice here.