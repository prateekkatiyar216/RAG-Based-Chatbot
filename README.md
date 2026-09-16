# PDF RAG Chat — Ready to Run

Ask questions about any PDF using local embeddings + Groq LLM.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
```bash
cp .env.example .env
# Edit .env and paste your key from https://console.groq.com
```

### 3. Run the app
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How to use

1. Upload one or more PDFs in the sidebar
2. Click **Index Documents** (only needed once per set of files)
3. Type your question in the chat box

## What's fixed vs the original

| Issue | Fix |
|---|---|
| Hardcoded empty Groq API key | Reads from `.env` via `GROQ_API_KEY` |
| No UI | Streamlit chat interface |
| Scanned PDF warning | Added clear error message |
| Import path bugs | Consistent `src.*` imports throughout |
| No feedback during indexing | Spinner + success message |

## Models used

- **Embeddings**: `all-MiniLM-L6-v2` (runs locally, no API needed)
- **LLM**: `gemma2-9b-it` via Groq (free tier available)

## Supported file types

PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON
