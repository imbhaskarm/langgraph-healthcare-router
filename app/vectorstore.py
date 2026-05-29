from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "healthcare_faiss_db"


def load_vectorstore() -> FAISS:
    """
    Load the pre-built FAISS index from disk.

    Raises FileNotFoundError with a clear message if the index has not been
    built yet, so the user knows exactly what to run first.
    """
    if not Path(VECTORSTORE_PATH).exists():
        raise FileNotFoundError(
            f"Vectorstore not found at '{VECTORSTORE_PATH}/'.\n"
            "Run: python build_vectorstore.py"
        )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # ⚠️ Updated from deprecated Chroma → FAISS (latest as of 2025)
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we built this index ourselves
    )
