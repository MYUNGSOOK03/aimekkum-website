# 보험 약관 챗봇 (로컬 RAG)

이 폴더는 로컬에서 약관 PDF를 올리고 검색/요약해 답변하는 간단한 RAG(검색+생성) 데모입니다.

## 빠른 시작

1) PDF 넣기
- `rag/data/` 폴더에 약관 PDF 파일들을 복사합니다.

2) 환경 설치 (Windows / cmd)
- 프로젝트 루트에서 아래 명령 실행:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r rag\requirements.txt
```

3) 인덱스 만들기 (첫 1회)
```
python rag\build_index.py
```

4) 챗봇 실행
```
streamlit run rag\app.py --server.port 7860
```

열기: http://localhost:7860

## 구성
- `data/` 원문 PDF 보관
- `index/` FAISS 벡터 인덱스 저장
- `build_index.py` PDF → 텍스트 추출 → 청크 → 임베딩 → 인덱스 저장
- `app.py` 질문 입력 → 유사 청크 검색 → 답변 및 근거 표시

## 참고
- 오프라인에서도 동작 가능 (모델은 로컬 임베딩 사용)
- 생성 응답은 간단한 규칙/요약에 그치며, 고급 생성은 클라우드 모델 연동 필요
