import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .config import VECTOR_DB_PATH, EMBEDDING_MODEL


def index_documents(data_dir="data/sample_docs", chunk_size=1000, chunk_overlap=150):
    all_docs = []

    files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    if not files:
        print(f"⚠️ No files found in {data_dir}")
        return

    print(f"📂 Indexing documents from: {data_dir}")
    for filename in sorted(files):
        file_path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[-1].lower()

        if ext == ".pdf":
            print(f"📄 Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for i, page in enumerate(pages):
                page.metadata.update({"source": filename, "page": i + 1})
            all_docs.extend(pages)

        elif ext in [".txt", ".md"]:
            print(f"📝 Loading text: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({"source": filename})
            all_docs.extend(docs)

        else:
            print(f"⚠️ Skipping unsupported file type: {filename}")

    if not all_docs:
        print("⚠️ No valid documents were loaded. Exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)
    print(f"🧩 Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(VECTOR_DB_PATH):
        print(f"⚠️ Existing FAISS index found at {VECTOR_DB_PATH}. It will be overwritten.")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    sources = sorted(set([d.metadata.get("source") for d in all_docs]))
    print(f"📚 Indexed {len(sources)} files → {len(chunks)} chunks total.")
    print(f"💾 FAISS index saved to {VECTOR_DB_PATH}")


if __name__ == "__main__":
    start = time.time()
    index_documents()
    print(f"⏱️ Done in {time.time() - start:.2f} seconds")