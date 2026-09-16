from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Scanning: {data_path}")
    documents = []

    loaders = [
        ("**/*.pdf",  lambda p: PyPDFLoader(str(p))),
        ("**/*.txt",  lambda p: TextLoader(str(p), encoding="utf-8")),
        ("**/*.csv",  lambda p: CSVLoader(str(p))),
        ("**/*.xlsx", lambda p: UnstructuredExcelLoader(str(p))),
        ("**/*.docx", lambda p: Docx2txtLoader(str(p))),
        ("**/*.json", lambda p: JSONLoader(str(p), jq_schema=".", text_content=False)),
    ]

    for pattern, loader_fn in loaders:
        files = list(data_path.glob(pattern))
        for f in files:
            try:
                loaded = loader_fn(f).load()
                documents.extend(loaded)
                print(f"[INFO] Loaded {len(loaded)} doc(s) from {f.name}")
            except Exception as e:
                print(f"[WARN] Skipped {f.name}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents
