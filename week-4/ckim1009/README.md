
## 개요

3주차에서 구축한 RAG Indexing 파이프라인에 Generation을 연결하여 end-to-end RAG를 완성하고, Naive RAG의 검색 한계를 넘어서는 Advanced RAG 기법을 학습합니다. 이론 과제에서는 Hybrid Search, Re-ranking, 메타데이터 필터링/컨텍스트 압축을 조사하고, 실습에서는 다년도 문서(2025/2026)를 다루는 RAG 시스템을 직접 구축합니다.

### Advanced RAG란?

RAG의 검색 성능을 한층 더 끌어올리기 위해 등장한 것이 Advanced RAG입니다. Gao et al.(2024)의 분류에 따르면 RAG는 다음과 같은 세 단계로 진화해왔습니다.

```
Naive RAG ──→ Advanced RAG ──→ Modular RAG
기본 검색 + 생성    검색 전/후 처리 추가    모듈 자유 조합
```

| 단계 | 핵심 특징 | 한계 |
|------|----------|------|
| **Naive RAG** | 질문을 벡터화하여 유사 문서를 검색하고 LLM에 전달하는 단순 파이프라인 | 검색 정밀도/재현율 부족, 키워드 매칭 실패, 도메인 용어 처리 취약 |
| **Advanced RAG** | 검색 **전**(Pre-Retrieval)에 쿼리 확장·재작성·HyDE 등을 적용하고, 검색 **후**(Post-Retrieval)에 재순위화·압축·필터링으로 결과를 정제 | Naive RAG 대비 복잡도 증가, 추가 모델/API 비용 발생 |
| **Modular RAG** | 검색·메모리·융합·라우팅·스케줄링 등 독립 모듈을 태스크에 맞게 자유롭게 조합 | 설계 복잡도 높음, 모듈 간 인터페이스 설계 필요 |

## 조사 항목

### 1. Hybrid Search란?

- Hybrid Search의 정의: 벡터 검색(Semantic Search)과 키워드 검색(BM25)을 결합하는 방식
- 벡터 검색만으로 부족한 이유:
    - 도메인 특화 용어, 고유명사의 경우 벡터 유사도만으로 매칭이 실패할 수 있음
    - 짧은 키워드 위주의 질의의 경우 벡터 검색의 정확도
- BM25 알고리즘의 핵심 원리:
    - BM25(Best Matching 25)는 TF-IDF의 발전된 형태
    - TF(Term Frequency) 포화도: 특정 단어가 문서에 아무리 많이 나와도 그 중요도가 무한히 커지지 않도록 로그 스케일로 제한
    - 문서 길이 정규화: 문서의 길이가 길수록 단어 출현 빈도의 가치를 상대적으로 낮게 평가하여 점수를 부여

#### 벡터 검색과 BM25 검색의 장단점 비교

| 검색 방식 | 강점 | 약점 |
|----------|------|------|
| 벡터 검색 (Semantic) | 동의어/유의어 파악, 문맥 이해, 다국어 처리 가능 | 고유명사 매칭에 취약 |
| BM25 (Keyword) | 정확한 키워드 매칭, 도메인 독립적, 계산 효율성 | 문맥 이해 불가, 키워드만큼 차원이 커짐 |
| Hybrid (결합) | 양쪽의 장점 결합 | 결합 알고리즘 설정 필요, 인덱싱 비용 증가 |

- 결합 방식: 가중치 기반 병합, Reciprocal Rank Fusion(RRF) 등

### 2. Re-ranking이란?

- Retrieval 단계에서 가져온 상위 k개의 문서 후보들을 다시 한번 정밀하게 평가하여 순위를 재조정하는 과정

| 구분 | Bi-encoder | Cross-encoder |
|------|-----------|---------------|
| 입력 방식 | 질문과 문서를 **각각** 인코딩 | 질문과 문서를 **함께** 인코딩 |
| 속도 | 빠름 | 느림 |
| 정확도 | 낮음 | 높음 |
| 사용 위치 | VectorDB에서 청크를 검색할 때 | 선별된 청크에서 정밀하게 청크 선별 |

- Cross-encoder를 검색 단계가 아닌 Re-ranking 단계에서 사용하는 이유: 
- Two-stage retrieval 패턴: Bi-encoder로 후보 선별 → Cross-encoder로 재정렬
- 상용 Re-ranking API: Cohere Rerank의 동작 방식과 장점

### 3. 메타데이터 필터링과 컨텍스트 압축

Advanced RAG에서는 검색 결과의 양보다 질이 중요합니다. 아래 두 가지 Post-Retrieval 기법을 조사하세요.

| 기법 | 조사 내용 |
|------|----------|
| 메타데이터 필터링 | 청크에 포함된 메타데이터(출처, 날짜, 카테고리 등)를 기반으로 검색 범위를 제한하는 기법. 다년도 문서나 다중 소스 RAG에서 특히 중요 |
| Contextual Compression | 검색된 청크에서 질문과 관련 있는 부분만 추출하여 LLM에게 전달하는 기법. 불필요한 정보를 제거하여 LLM의 답변 정확도를 향상 |

**Metadata Filtering**
- 청크에 부여된 속성으로 검색 범위를 사전 제한
- 질문 범위가 특정 연도나 특정 문서로 한정될 때, 무관한 청크가 섞여 발생하는 할루시네이션 방지

```python
vector_db = Chroma(
    persist_directory="chroma_medical_index",
    embedding_function=embeddings)
vector_db.as_retriever(search_kwargs={"k": 10, 'filter': {'year': 2026}})
```

**Contextual Compression**
- 검색된 청크에서 질문과 무관한 문장을 제거하고 핵심 내용만 추출
- Lost in the Middle 현상 방지 및 토큰 비용 절감

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
compressor = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    top_n=3
)

advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)
```


## 실습 과제 예측

실습 과제(TASK.md)를 보고, 실습 전에 아래 가설을 세워주세요.

1. 2025년과 2026년 문서를 동시에 인덱싱했을 때, 년도 혼동이 가장 많이 발생할 질문 유형을 예측하세요
2. Hybrid Search가 가장 효과를 볼 질문 유형을 예측하세요 (예: 특정 의료 용어가 포함된 질문 vs 일반적 표현의 질문)
3. Re-ranking이 결과를 개선할지, 또는 오히려 악화시킬 수 있는 경우가 있을지 예측하세요
4. 메타데이터 필터링이 년도 혼동 문제를 얼마나 해결할 수 있을지 예측하세요


# 4주차 실습 과제: Basic RAG 완성 + Advanced RAG (Hybrid Search & Re-ranking)

## 배경

3주차에서는 RAG Indexing 파이프라인(PDF → 청킹 → 임베딩 → 벡터 저장소)을 구축하고, Golden Dataset으로 검색 품질을 확인했습니다. 하지만 검색된 청크를 실제로 LLM에게 전달하여 답변을 생성하는 단계는 아직 구현하지 않았습니다.

이번 과제에서는 **다년도 문서를 다루는 RAG 시스템**을 구축합니다. 실무에서는 동일한 제도가 매년 개정되며, 사용자는 특정 년도의 정보를 정확히 알고 싶어합니다. 2025년과 2026년 두 해의 의료급여제도 문서를 동시에 인덱싱하고, 올바른 년도의 정보를 검색하여 답변을 생성하는 것이 핵심 과제입니다.

이번 과제에서는 세 가지를 합니다.

1. **Basic RAG 완성**: 두 해의 문서를 벡터 저장소에 인덱싱하고 Generation을 연결하여 end-to-end RAG를 완성합니다
2. **Advanced RAG 구현**: Hybrid Search(벡터 + BM25)와 Re-ranking(Cohere Rerank)을 적용하여 검색 품질을 개선합니다
3. **년도 인식 검색**: 질문이 요구하는 년도의 문서에서 정확히 검색하는 능력을 평가합니다


## 실습 구조

### Step 1: Basic RAG 완성 (Retrieval + Generation)

두 해의 PDF를 모두 인덱싱한 벡터 저장소를 사용하여 end-to-end RAG 파이프라인을 완성합니다.

```
질문 → 임베딩 → 벡터 검색 (Top-K) → 검색된 청크를 컨텍스트로 구성 → LLM 생성 → 답변
```

**1-1. 인덱싱**
1. 2025년, 2026년 PDF를 각각 로드하고 청킹합니다
2. 각 청크의 메타데이터에 `source_year` 필드를 추가합니다 (예: `{"source_year": "2025"}`)
3. 두 년도의 청크를 하나의 벡터 저장소에 함께 인덱싱합니다 (FAISS 또는 Chroma)

**1-2. Generation 파이프라인 연결**
1. 벡터 저장소를 로드합니다
2. Retriever를 생성합니다 (Top-K 설정)
3. 검색된 청크를 LLM 프롬프트에 컨텍스트로 전달하는 체인을 구성합니다
4. RAG 프롬프트 템플릿을 작성합니다 — 년도 정보를 인식할 수 있도록 컨텍스트에 출처 년도를 포함합니다

```python
# 프롬프트 템플릿 예시
rag_prompt = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
```

**1-3. Golden Dataset으로 end-to-end 정답률 측정**
1. Golden Dataset 전체 문항에 대해 RAG 파이프라인을 실행합니다
2. 생성된 답변을 `expected_answer`와 비교합니다
3. 정답/오답과 함께 **올바른 년도에서 검색했는지** 여부를 기록합니다

**판정 기준**
```
정답: LLM이 생성한 답변이 expected_answer의 핵심 값을 포함하는 경우
오답: 핵심 값이 누락되거나 다른 값을 답한 경우
년도 오류: 올바른 주제를 검색했지만 다른 년도의 정보를 사용한 경우 (부분 실패)
```

> 자동 판정이 어려우면 수동 판정도 가능합니다. 판정 기준을 README에 명시하세요.

**기록**

| 질문 ID | 난이도 | source_year | 검색된 청크 포함 여부 | 올바른 년도 검색 여부 | LLM 생성 답변 | 정답 여부 | 오답 원인 |
|---------|--------|-------------|-------------------|-------------------|-------------|----------|----------|
| q01 | easy | 2025 | O/X | O/X | | 정답/오답 | |
| q02 | medium | 2026 | O/X | O/X | | 정답/오답 | |
| q03 | cross-year | 2025+2026 | O/X | O/X | | 정답/오답 | |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **Basic RAG 정답률** | | | | | | /N | |

**년도 혼동 분석**

올바른 주제를 검색했지만 다른 년도의 청크를 가져온 경우를 별도로 분석합니다.

| 항목 | 값 |
|------|-----|
| 올바른 년도 검색 성공률 | /N |
| 년도 혼동으로 인한 오답 수 | |
| 주요 년도 혼동 패턴 | |

**2주차 vs Basic RAG 비교** (해당하는 문항이 있는 경우)

| 방식 | 정답률 | 비고 |
|------|--------|------|
| 2주차 Zero-shot (전체 데이터 in system prompt) | % | |
| 2주차 최고 성능 기법 | % | 기법명 기재 |
| 4주차 Basic RAG | % | |

> Golden Dataset이 2주차 30문제와 다를 수 있으므로, 동일 문항 기준으로 비교하거나 전체 경향을 비교합니다.

### Step 2: Advanced RAG (Hybrid Search + Re-ranking)

Basic RAG의 검색을 개선합니다. 두 가지 기법을 적용할 수 있습니다.

#### 2-1. Hybrid Search (벡터 검색 + BM25 키워드 검색)

벡터 검색만으로는 놓치는 문서가 있습니다. 키워드 기반 검색(BM25)을 결합하여 검색 범위를 넓힙니다.

```
질문 → [벡터 검색 (Top-K)] + [BM25 검색 (Top-K)] → 결과 병합 → 중복 제거
```

**벡터DB와 BM25 구현 방식에 대한 주의사항**

BM25를 벡터DB와 결합하는 방식은 사용하는 벡터DB에 따라 달라집니다.

- **ChromaDB, FAISS**: Dense Vector만 지원하므로 BM25를 DB 내에서 직접 수행할 수 없습니다. LangChain의 `BM25Retriever.from_documents()`처럼 메모리에 별도의 BM25 인덱스를 구성하고, `EnsembleRetriever`로 벡터 검색 결과와 병합하는 방식을 사용해야 합니다. 이 경우 BM25 인덱스는 persist되지 않으므로, 앱 재시작 시 원본 문서를 다시 로드하여 인덱스를 재구축해야 합니다.
- **Qdrant, Weaviate, Pinecone, Milvus**: Sparse Vector를 네이티브로 지원하므로 DB 자체에서 Hybrid Search를 수행할 수 있습니다. Dense Vector와 Sparse Vector를 함께 저장하고, 검색 시 두 벡터를 동시에 활용하는 방식입니다. 별도의 메모리 기반 BM25 인덱스가 필요 없으며, 데이터가 DB에 persist되므로 재구축 부담이 없습니다.

이번 과제에서는 FAISS 또는 ChromaDB를 사용하므로 전자의 방식(메모리 기반 BM25 + EnsembleRetriever)으로 구현합니다. 실무에서 대규모 데이터에 Hybrid Search를 적용할 때는 Sparse Vector를 네이티브 지원하는 벡터DB를 고려하세요.

> LangChain 또는 LlamaIndex 중 3주차에서 사용한 프레임워크를 그대로 쓰는 것을 권장합니다.

#### 2-2. 메타데이터 필터링 (선택)

년도 인식 검색을 개선하기 위해 메타데이터 필터링을 추가로 적용할 수 있습니다. 질문에서 년도를 추출하고, 해당 년도의 청크만 검색 대상으로 제한하는 방식입니다.

```python
# 메타데이터 필터링 예시 (Chroma)
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"source_year": "2025"}}
)
```

> 메타데이터 필터링은 필수가 아닌 선택 사항이지만, 년도 혼동 문제를 해결하는 효과적인 방법입니다. 적용 여부와 결과를 기록하세요.

#### 2-3. Re-ranking (Cohere Rerank)

Hybrid Search로 가져온 후보 문서들을 Re-ranker로 재정렬하여, 질문과 가장 관련성 높은 청크를 상위로 올립니다.

```
Hybrid Search 결과 (N개) → Re-ranker 스코어링 → 상위 K개 선택 → LLM 생성
```

**구현 방법 (권장: Cohere Rerank)**

Cohere Rerank API는 무료 티어(월 1,000회)를 제공하며 한국어를 포함한 다국어를 지원합니다.

1. Cohere API 키를 발급받습니다 (https://dashboard.cohere.com)
2. `langchain-cohere` 패키지를 설치합니다
3. Hybrid Search 결과를 Cohere Rerank로 재정렬합니다

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(
    model="rerank-v3.5",
    top_n=3
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

results = compression_retriever.invoke("질문")
```

**대안 Re-ranker 옵션**

| Re-ranker | 유형 | 한국어 지원 | 특징 |
|-----------|------|-----------|------|
| Cohere Rerank v3.5 | 상용 API | 100+ 언어 | 무료 티어(월 1,000회), 가장 쉬운 연동 |
| Jina Reranker v3 | 상용 API | 다국어 | 저지연, 긴 문서(8K 토큰) 지원 |
| bge-reranker-v2-m3 | 오픈 모델 | 다국어 | BAAI 제작, 무료, 로컬 실행, 한국어 성능 양호 |
| CrossEncoder | 오픈 모델 | 영어 위주 | sentence-transformers, 가장 기본적 |

오픈 모델을 사용하면 API 비용 없이 로컬에서 실행할 수 있습니다.

```python
# 오픈 모델 예시: bge-reranker-v2-m3 (다국어 지원)
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")

pairs = [(question, doc.page_content) for doc in hybrid_results]
scores = cross_encoder.predict(pairs)

reranked = [doc for _, doc in sorted(zip(scores, hybrid_results), reverse=True)][:top_k]
```

**기록**

| 항목 | 설정값 |
|------|--------|
| BM25 Retriever k | 10 |
| Vector Retriever k | 10 |
| Ensemble 가중치 (vector : BM25) | 5 : 5 |
| Re-ranker 종류 및 모델명 | `BAAI/bge-reranker-v2-m3` |
| Re-ranking 후 최종 Top-K | 3 |

#### 2-4. Advanced RAG 정답률 측정

Golden Dataset 전체에 대해 Advanced RAG 파이프라인을 실행하고 정답률을 측정합니다.

**기록**

| 질문 ID | 난이도 | source_year | 검색 방식 | 검색 결과 포함 여부 | 올바른 년도 검색 여부 | Re-rank 후 순위 변화 | LLM 생성 답변 | 정답 여부 |
|---------|--------|-------------|----------|-------------------|-------------------|-------------------|-------------|----------|
| q01 | easy | 2025 | Hybrid | O/X | O/X | 예: 3위→1위 | | 정답/오답 |
| q02 | cross-year | 2025+2026 | Hybrid | O/X | O/X | | | 정답/오답 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **Advanced RAG 정답률** | | | | | | | | /N |

### Step 3: Basic RAG vs Advanced RAG 비교 분석

두 파이프라인의 성능을 비교하고, **왜** 차이가 나는지 분석합니다.

**3-1. 정답률 비교 테이블**

| 방식 | 전체 정답률 | easy 정답률 | medium 정답률 | hard 정답률 | cross-year 정답률 | 년도 검색 정확도 |
|------|-----------|-----------|-------------|-----------|-----------------|----------------|
| Basic RAG (벡터 검색만) | /N | /N | /N | /N | /N | /N |
| Advanced RAG (Hybrid + Re-ranking) | /N | /N | /N | /N | /N | /N |

**3-2. 문항별 변화 분석**

| 질문 ID | Basic RAG | Advanced RAG | 변화 | 변화 원인 분석 |
|---------|----------|-------------|------|-------------|
| q01 | 정답 | 정답 | 유지 | |
| q02 | 오답 | 정답 | 개선 | 예: BM25가 키워드 매칭으로 누락 청크를 보완 |
| q03 | 정답 | 오답 | 악화 | 예: Re-ranking이 관련 청크를 밀어냄 |
| ... | ... | ... | ... | ... |

**3-3. 기법별 기여도 분석** (선택이지만 권장)

Hybrid Search만 적용한 결과와 Hybrid + Re-ranking을 적용한 결과를 분리하여, 각 기법이 얼마나 기여했는지 확인합니다.

| 방식 | 정답률 |
|------|--------|
| Basic RAG (벡터만) | /N |
| + Hybrid Search (벡터 + BM25) | /N |
| + Hybrid Search + Re-ranking | /N |
| + 메타데이터 필터링 (적용 시) | /N |