import os
import json
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
META_PATH = INDEX_DIR / "metadata.json"
INDEX_PATH = INDEX_DIR / "faiss.index"

CHUNK_SIZE = 700  # chars
CHUNK_OVERLAP = 120  # chars
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    texts: List[str] = []
    metadatas: List[dict] = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        full_text = read_pdf(pdf)
        if not full_text.strip():
            continue
        chunks = chunk_text(full_text)
        for idx, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({
                "source": pdf.name,
                "chunk_id": idx,
            })

    if not texts:
        print("No PDFs found or empty extracts in data/. Place PDFs then retry.")
        return

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"metadatas": metadatas, "texts": texts}, f, ensure_ascii=False)

    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    build()
