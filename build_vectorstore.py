import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAQ_PATH       = Path("healthcare_faq.json")
VECTORSTORE_PATH = "healthcare_faiss_db"


def build():
    """
    Load healthcare_faq.json, convert to LangChain Documents,
    build a FAISS index, and persist it to disk.

    Run this ONCE before starting the app:
        python build_vectorstore.py

    The saved index is loaded at runtime by vectorstore.py.
    """
    print("Loading FAQ data...")
    with open(FAQ_PATH, "r") as f:
        raw = json.load(f)

    # Original JSON uses 'medical_records' as department name but the LLM
    # classifier is prompted to return 'records'. Normalise here so the
    # metadata filter always matches.
    docs = [
        Document(
            page_content=item["question"] + "\n" + item["answer"],
            metadata={"category": item["department"].replace("medical_records", "records")},
        )
        for item in raw
    ]
    print(f"  {len(docs)} documents prepared.")

    print("Building FAISS index (downloading embeddings model on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"  Vectorstore saved to '{VECTORSTORE_PATH}/'")
    print("Done. You can now run: python main.py")


if __name__ == "__main__":
    build()
