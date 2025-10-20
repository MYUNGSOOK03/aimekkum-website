import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "index" / "faiss.index"
META_PATH = BASE_DIR / "index" / "metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_index = None
_meta = None
_model = None


def load_index():
    global _index, _meta, _model
    if _index is None or _meta is None or _model is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return None, None, None
        _index = faiss.read_index(str(INDEX_PATH))
        with META_PATH.open("r", encoding="utf-8") as f:
            _meta = json.load(f)
        _model = SentenceTransformer(EMBED_MODEL)
    return _index, _meta, _model


class AskReq(BaseModel):
    query: str
    top_k: Optional[int] = 4


class Chunk(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float


class AskRes(BaseModel):
    answers: List[str]
    chunks: List[Chunk]


app = FastAPI(title="Policy QA API", version="0.1.0")

# CORS for local dev (adjust origins as needed)
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ask", response_model=AskRes)
async def ask(req: AskReq):
    index, meta, model = load_index()
    if index is None:
        raise HTTPException(status_code=409, detail="Index not built. Upload PDFs and build index first.")

    q_emb = model.encode([req.query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    k = max(1, min(int(req.top_k or 4), 10))
    D, I = index.search(q_emb, k)

    chunks = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        info = meta["metadatas"][idx]
        text = meta["texts"][idx]
        chunks.append(Chunk(source=info["source"], chunk_id=info["chunk_id"], text=text, score=float(score)))

    # simple bullet extraction
    bullets: List[str] = []
    for c in chunks:
        for line in c.text.splitlines():
            s = line.strip()
            if 12 <= len(s) <= 180 and s[-1] in ".):":
                bullets.append(s)
                break
    bullets = bullets[:5]
    if not bullets:
        bullets = ["더 구체적으로 질문해 주세요. (조항 번호/서류/예외 등)"]

    return AskRes(answers=bullets, chunks=chunks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
