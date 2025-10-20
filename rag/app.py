import json
from pathlib import Path

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = BASE_DIR / "index" / "faiss.index"
META_PATH = BASE_DIR / "index" / "metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return index, meta, model


def main():
    st.set_page_config(page_title="보험 약관 챗봇 (로컬)", page_icon="🤖", layout="wide")

    st.title("🤖 보험 약관 챗봇")
    st.caption("PDF 올리고 질문하면 관련 조항과 함께 답변합니다 (로컬 RAG 데모)")

    with st.expander("1) 약관 PDF 올리기 / 교체하기", expanded=True):
        st.write("- 여기에서 PDF를 업로드하면 `rag/data/` 폴더에 저장됩니다.")
        uploaded = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
        saved_files = []
        if uploaded:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                dest = DATA_DIR / f.name
                dest.write_bytes(f.getbuffer())
                saved_files.append(dest.name)
            st.success(f"{len(saved_files)}개 파일 저장: {', '.join(saved_files[:3])}{' …' if len(saved_files) > 3 else ''}")

        if st.button("인덱스 다시 만들기"):
            import subprocess, sys
            cmd = [sys.executable, str(BASE_DIR / "build_index.py")]
            with st.spinner("인덱스 생성 중…"):
                proc = subprocess.run(cmd, capture_output=True, text=True)
            st.code(proc.stdout or "(no stdout)")
            if proc.stderr:
                st.error(proc.stderr)
            # 캐시된 인덱스 새로고침
            st.cache_resource.clear()

    index, meta, model = load_index()
    if index is None:
        st.warning("인덱스가 없습니다. 먼저 PDF를 넣고 '인덱스 다시 만들기'를 눌러주세요.")
        return

    query = st.text_input("질문을 입력하세요", placeholder="예) 입원비 청구 시 필요한 서류는?")
    top_k = st.slider("참고할 조각 수 (Top-K)", 1, 8, 4)

    if query:
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")
        D, I = index.search(q_emb, top_k)
        hits = I[0].tolist()

        st.subheader("🔎 근거 조각")
        for rank, idx in enumerate(hits, start=1):
            chunk = meta["texts"][idx]
            info = meta["metadatas"][idx]
            with st.expander(f"#{rank} {info['source']} (chunk {info['chunk_id']})", expanded=rank==1):
                st.write(chunk)

        # 간단한 규칙/요약 기반 답안 (LLM 없이)
        st.subheader("📝 요약 답변")
        st.write("아래는 유사 조각의 핵심 문장을 뽑아 간단히 요약한 내용입니다. 보다 자연스러운 문장/해석은 클라우드 LLM 연동 시 가능해집니다.")
        bullet_points = []
        for idx in hits:
            text = meta["texts"][idx]
            for line in text.splitlines():
                line = line.strip()
                if len(line) > 0 and 10 <= len(line) <= 160 and line[-1] in ".):":
                    bullet_points.append(line)
                    break
        bullet_points = bullet_points[:5]
        if bullet_points:
            for bp in bullet_points:
                st.markdown(f"- {bp}")
        else:
            st.info("명확한 요약 포인트를 찾지 못했습니다. 질문을 더 구체화해보세요.")

        st.caption("⚠️ 법률/보험 등 중요한 결정은 반드시 원문 약관과 전문가 상담으로 확인하세요.")


if __name__ == "__main__":
    main()
