import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import os

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
ALLOWED_FILES = {
    "who_diabetes_diagnosis.pdf",
    "idf_diabetes_management.pdf",
}


def load_pdfs(folder="data/guidelines/"):
    texts = []
    if not os.path.isdir(folder):
        return texts

    for file in os.listdir(folder):
        if file in ALLOWED_FILES:
            reader = PyPDF2.PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append({"text": text, "source": file})
    return texts


def chunk_text(texts, chunk_size=150, overlap=30):
    chunks = []
    for doc in texts:
        words = doc["text"].split()
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunks.append({
                "content": " ".join(words[i:i + chunk_size]),
                "source": doc["source"],
            })
    return chunks


def create_index(chunks):
    if not chunks:
        return None, []

    texts = [c["content"] for c in chunks]
    embeddings = embed_model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks


def search(query, index, chunks, k=5):
    if index is None or not chunks:
        return []

    q_vec = embed_model.encode([query])
    _, idx = index.search(np.array(q_vec, dtype=np.float32), min(k, len(chunks)))
    return [chunks[i] for i in idx[0]]


def rag_pipeline(query, folder="data/guidelines/"):
    texts = load_pdfs(folder)
    chunks = chunk_text(texts)
    index, chunks = create_index(chunks)

    results = search(query, index, chunks)
    return results
