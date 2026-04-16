# Advanced RAG (Hybrid Search + Re-ranking) 구현 계획

## 1. 개요
기존의 벡터 검색(Dense) 방식에 키워드 검색(Sparse, BM25)을 결합하고, 검색된 후보군을 Re-ranker로 재정렬하여 답변의 정확도를 극대화합니다.

## 2. 주요 단계 및 구성 요소

### 2-1. 라이브러리 추가 (requirements.txt)
* `rank_bm25`: 메모리 기반 BM25 키워드 검색 엔진
* `flashrank`: 가볍고 성능이 우수한 로컬 Re-ranker (API 비용 없음)

### 2-2. 설정 확장 (src/core/config.py)
* `HYBRID_WEIGHT_BM25`: 0.5 (키워드 검색 가중치)
* `HYBRID_WEIGHT_VECTOR`: 0.5 (벡터 검색 가중치)
* `RERANK_TOP_K`: 5 (최종 LLM에 전달할 문서 개수)

### 2-3. 서비스 레이어 개선 (src/domains/rag/service.py)
1. **BM25Retriever 초기화**: ChromaDB에서 전체 문서를 로드하여 BM25 인덱스 생성 (메모리 기반).
2. **Hybrid Search (EnsembleRetriever)**:
   - `Chroma.as_retriever(k=20)` + `BM25Retriever.from_documents(k=20)` 결합.
3. **Re-ranking (ContextualCompressionRetriever)**:
   - `FlashrankRerank`를 사용하여 하이브리드 검색 결과 중 가장 관련성 높은 상위 문서를 재선별.
4. **`get_advanced_answer` 구현**: 정교한 컨텍스트 기반 답변 생성 로직 추가.

### 2-4. API 엔드포인트 추가 (src/domains/rag/router.py)
* `POST /api/v1/rag/advanced-query`: 개선된 엔진 단일 질문 테스트용.
* `POST /api/v1/rag/advanced`: 골든 데이터셋 평가용 (결과를 `data/advanced/{index}` 폴더에 자동 저장).

## 3. 기대 효과
* **검색 누락 방지**: 벡터 검색이 놓치는 고유 명사나 특정 키워드를 BM25가 보완.
* **할루시네이션 감소**: Re-ranker가 질문과 가장 밀접한 문서만 선별하여 LLM에 전달.
* **성능 지표 확보**: Basic RAG와 Advanced RAG의 평가 결과를 비교하여 개선 효과 증명.
