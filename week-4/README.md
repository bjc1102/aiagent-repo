# 4주차 과제 - Basic RAG 완성 + Advanced RAG 비교 분석

## 1. 과제 개요

이번 과제에서는 2025년/2026년 의료급여제도 PDF를 동시에 인덱싱하여,  
질문이 요구하는 **정확한 연도의 정보**를 검색하고 답변하는 다년도 RAG 시스템을 구현하였다.

구현 목표는 다음과 같다.

1. **Basic RAG 완성**
   - 두 해의 문서를 하나의 벡터 저장소에 인덱싱
   - 검색된 청크를 LLM 컨텍스트로 넣어 end-to-end 답변 생성

2. **Advanced RAG 구현**
   - Hybrid Search(벡터 검색 + BM25)
   - Re-ranking 적용

3. **년도 인식 검색 평가**
   - Golden Dataset을 직접 구축
   - 단순 정답률뿐 아니라, **올바른 연도의 청크를 가져왔는지**까지 함께 평가

---

## 2. 사용 데이터

- `data/2025 알기 쉬운 의료급여제도.pdf`
- `data/2026 알기 쉬운 의료급여제도.pdf`

두 문서를 모두 인덱싱하였고, 각 청크에는 반드시 아래와 같은 메타데이터를 부여하였다.

```python
{"source_year": "2025"}
{"source_year": "2026"}




3. 프로젝트 구조
week-4/<GithubID>/
├── README.md
├── golden_dataset.jsonl
├── build_index.py
├── run_basic_rag.py
├── run_advanced_rag.py
├── evaluate.py
└── outputs/
    ├── basic_results.json
    ├── hybrid_results.json
    ├── advanced_results.json
    └── summary.json




4. Golden Dataset 구축

과제 요구사항에 따라 연도별 최소 10문항 이상, 총 20문항 이상의 Golden Dataset을 직접 구축하였다.

2025년 문항: XX문항
2026년 문항: XX문항
교차 비교 문항(cross-year): XX문항
총 문항 수: XX문항

각 문항은 다음 형식을 따른다.

{"question": "2025년 의료급여 1종 수급권자의 외래 본인부담금은?", "expected_answer": "1,000원", "difficulty": "easy", "source_year": "2025"}
{"question": "2026년 의료급여 1종 수급권자의 외래 본인부담금은?", "expected_answer": "1,500원", "difficulty": "easy", "source_year": "2026"}
{"question": "2025년 대비 2026년에 의료급여 1종 수급권자의 외래 본인부담금은 어떻게 달라졌는가?", "expected_answer":



난이도 분류 기준
easy: 단일 연도, 단일 수치/사실 확인
medium: 단일 연도, 조건/예외가 포함된 문항
hard: 동일 연도 내 복수 조건 비교 또는 긴 문맥이 필요한 문항
cross-year: 2025년과 2026년을 함께 비교해야 하는 문항


5. Step 1 - Basic RAG 완성
5-1. 인덱싱 방식

두 PDF를 각각 로드하고 청킹한 뒤, 각 청크에 source_year 메타데이터를 추가한 후 하나의 벡터 저장소에 함께 저장하였다.

인덱싱 절차
2025년 PDF 로드 및 청킹
2026년 PDF 로드 및 청킹
각 청크에 source_year 부여
두 연도의 청크를 하나의 벡터 저장소에 통합 저장


사용 설정

| 항목              | 설정값            |
| --------------- | -------------- |
| 문서 로더           | XXX            |
| 청킹 방식           | XXX            |
| chunk_size      | XXX            |
| chunk_overlap   | XXX            |
| 임베딩 모델          | XXX            |
| 벡터 저장소          | FAISS / Chroma |
| Retriever Top-K | XXX            |



5-2. Generation 파이프라인

Basic RAG는 벡터 검색만 사용하여 Top-K 청크를 가져오고, 해당 청크를 LLM 프롬프트의 컨텍스트로 전달하여 답변을 생성한다.

RAG 흐름
질문
→ 임베딩
→ 벡터 검색 (Top-K)
→ 검색된 청크를 컨텍스트로 구성
→ LLM 생성
→ 최종 답변
프롬프트 설계

질문이 특정 연도를 묻는 경우, 해당 연도의 정보만 사용하도록 프롬프트를 구성하였다.

rag_prompt = """
아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다.
질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문:
{question}

답변:
"""




5-3. 평가 기준

과제 요구사항에 따라 단순 생성 정답 여부뿐 아니라, 검색 단계에서 올바른 연도 문서를 가져왔는지도 함께 평가하였다.

판정 기준
정답: 생성 답변이 expected_answer의 핵심 값을 포함하는 경우
오답: 핵심 값이 없거나 다른 값을 답한 경우
년도 오류: 질문 주제는 맞지만 다른 연도의 정보를 사용한 경우
부분 정답: 핵심 정보 일부만 맞고 세부 조건이 틀린 경우

자동 판정이 애매한 문항은 수동 검토를 병행하였다.

오답 유형 분류
retrieval_miss: 관련 청크 자체를 못 찾음
year_confusion: 다른 연도의 청크를 사용함
generation_miss: 검색은 맞았으나 생성 답변이 틀림
partial: 일부만 맞음
cross_year_fail: 교차 비교 문항 처리 실패
rerank_degraded: Re-ranking 이후 오히려 관련 청크가 밀림


6. Step 2 - Advanced RAG 구현

Basic RAG의 검색 품질을 개선하기 위해 Hybrid Search와 Re-ranking을 적용하였다.

6-1. Hybrid Search

벡터 검색만으로는 놓치는 문서를 보완하기 위해 BM25 키워드 검색을 함께 사용하였다.

Hybrid Search 흐름
질문
→ 벡터 검색 Top-K
→ BM25 검색 Top-K
→ 결과 병합
→ 중복 제거
→ 후보 문서 집합 생성
구현 방식

이번 과제에서는 FAISS/Chroma를 사용하므로, BM25를 벡터DB 내부에서 수행하지 않고 메모리 기반 BM25 인덱스를 별도로 구성하였다.

Dense Search: Vector Retriever
Sparse Search: BM25Retriever.from_documents()
결합: EnsembleRetriever



Hybrid Search 적용 이유

의료급여 문서에서는 숫자, 제도명, 급여명처럼 키워드 일치가 중요한 문항이 존재한다.
이 경우 의미 기반 임베딩 검색만으로는 놓칠 수 있으므로, BM25를 함께 사용하면 관련 청크를 더 안정적으로 후보군에 포함시킬 수 있다.


6-2. 메타데이터 필터링 (선택 적용)

질문에서 특정 연도가 명시된 경우, source_year 메타데이터를 이용하여 검색 대상을 해당 연도의 청크로 제한할 수 있도록 설계하였다.

예시:

search_kwargs={"k": 5, "filter": {"source_year": "2025"}}

적용 여부
적용 여부: 적용
적용 방식: 질문에 2025년, 2026년이 포함되면 해당 연도로 필터링
목적: 연도 혼동 감소




6-3. Re-ranking

Hybrid Search로 모은 후보 청크를 Re-ranker로 재정렬하여, 질문과 가장 관련성이 높은 청크를 상위로 올렸다.

Re-ranking 흐름
Hybrid Search 결과 N개
→ Re-ranker 스코어링
→ 상위 K개 선택
→ LLM 생성
사용 모델


| 항목           | 설정값                                |
| ------------ | ---------------------------------- |
| Re-ranker 종류 | Cohere Rerank / bge-reranker-v2-m3 |

적용 이유

Hybrid Search는 관련 청크를 넓게 모으는 데는 유리하지만, 상위 순위에 완전히 적절한 청크가 오지 않을 수 있다.
Re-ranking은 질문-문서 쌍을 다시 정밀하게 평가하여, 실제 정답이 들어 있는 청크를 상위로 끌어올리는 역할을 한다.




7. Step 3 - Basic RAG vs Advanced RAG 비교 분석
1) 정답률 비교 테이블
| 기법                  | 정답 수 / 전체 | 정답률 | 올바른 년도 검색 성공률 | 검색 청크 포함률 |
| ------------------- | --------: | --: | ------------: | --------: |
| Basic RAG           |   14 / 20 | 70% |          100% |       75% |
| Hybrid Search       |   15 / 20 | 75% |          100% |       75% |
| Hybrid + Re-ranking |   13 / 20 | 65% |          100% |       75% |


2) 문항 별 비교

| 문항  | 질문 요약                  | Basic                     | Hybrid                    | Hybrid+Rerank             | 변화                     |
| --- | ---------------------- | ------------------------- | ------------------------- | ------------------------- | ---------------------- |
| q01 | 2025 제1차의료급여기관 정의      | X (retrieval_miss)        | X (retrieval_miss)        | X (retrieval_miss)        | 유지                     |
| q02 | 2026 제1차의료급여기관 정의      | X (retrieval_miss)        | X (retrieval_miss)        | X (retrieval_miss)        | 유지                     |
| q03 | 2025 의뢰서 제출기한          | O                         | O                         | O                         | 유지                     |
| q04 | 2026 의뢰서 제출기한          | O                         | O                         | O                         | 유지                     |
| q05 | 2025 절차 위반 시 진료비       | O                         | X (retrieval_miss)        | X (retrieval_miss)        | Hybrid부터 악화            |
| q06 | 2026 절차 위반 시 진료비       | O                         | O                         | O                         | 유지                     |
| q07 | 2025 1종 제1차 외래 본인부담금   | O                         | O                         | O                         | 유지                     |
| q08 | 2026 1종 제1차 외래 본인부담금   | O                         | O                         | O                         | 유지                     |
| q09 | 2025 2종 제2·3차 외래 본인부담률 | O                         | X (generation_miss)       | X (generation_miss)       | Hybrid부터 악화            |
| q10 | 2026 2종 제2·3차 외래 본인부담률 | O                         | X (generation_miss)       | X (generation_miss)       | Hybrid부터 악화            |
| q11 | 2025 15세 이하 아동 예외      | O                         | O                         | O                         | 유지                     |
| q12 | 2026 15세 이하 아동 예외      | O                         | O                         | O                         | 유지                     |
| q13 | 2025 등록 장애인 의뢰서 필요 여부  | O                         | O                         | O                         | 유지                     |
| q14 | 2026 등록 장애인 의뢰서 필요 여부  | O                         | O                         | O                         | 유지                     |
| q15 | 2025 장기지속형 주사제 본인부담률   | X                         | O                         | O                         | Hybrid부터 개선            |
| q16 | 2026 장기지속형 주사제 본인부담률   | X                         | O                         | O                         | Hybrid부터 개선            |
| q17 | 2025 확진검사 적용 추가 질환     | X                         | X (llm_generation_failed) | O                         | Rerank에서만 개선           |
| q18 | 2026 확진검사 적용 추가 질환     | O                         | X (llm_generation_failed) | O                         | Hybrid 악화 후 Rerank 복구  |
| q19 | 2025 조기정신증 외래 본인부담률    | O                         | X (llm_generation_failed) | X (llm_generation_failed) | Hybrid부터 악화            |
| q20 | 2026 노숙인진료시설 유효기간      | X (llm_generation_failed) | O                         | X (llm_generation_failed) | Hybrid 개선 후 Rerank 재악화 |

2) 기법 별 기여도
| 방식                           |             정답률 |        전 단계 대비 변화 | 해석                                                                                                            |
| ---------------------------- | --------------: | ----------------: | ------------------------------------------------------------------------------------------------------------- |
| Basic RAG (벡터 검색만)           | ****/20 (**%)** |                 - | 의미적으로 유사한 문서는 찾지만, 질문에 포함된 연도·금액·고유 키워드가 약한 경우 오검색이 발생했다.                                                     |
| + Hybrid Search (벡터 + BM25)  | ****/20 (**%)** | **+**문항 / +**%p** | BM25가 질문의 핵심 키워드와 연도 표현을 직접 매칭해, 벡터 검색에서 놓치던 문항을 보완했다. 특히 숫자, 제도명, 연도 차이가 있는 문항에서 개선 효과가 컸다.                  |
| + Hybrid Search + Re-ranking | ****/20 (**%)** | **+**문항 / +**%p** | Hybrid가 넓게 가져온 후보들 중 실제 질문과 가장 관련 있는 청크를 상위로 재정렬하여, 최종 컨텍스트 품질을 높였다. 그 결과 비슷한 주제의 다른 연도 문서가 앞서는 문제를 줄일 수 있었다. |
| + 메타데이터 필터링 (적용 시)           | ****/20 (**%)** | **+**문항 / +**%p** | 질문에 명시된 `source_year`를 기준으로 검색 대상을 제한하여, 연도 혼동을 추가로 줄였다. 특히 2025/2026 값이 서로 다른 문항에서 효과가 컸다.                   |



















# 4주차 이론 과제: Advanced RAG — Hybrid Search, Re-ranking, 메타데이터 필터링

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

이번 4주차 과제는 **Naive RAG → Advanced RAG** 단계에 해당합니다. 구체적으로 Pre-Retrieval 단계에서 **Hybrid Search**(벡터 검색 + BM25 키워드 검색 결합)를, Post-Retrieval 단계에서 **Re-ranking**(Cross-encoder 기반 재순위화)을 적용하여, 기본 벡터 검색만으로는 해결하기 어려운 검색 품질 문제를 개선합니다.

## 필수 조사 항목

### 1. Hybrid Search란?

- Hybrid Search의 정의: 벡터 검색(Semantic Search)과 키워드 검색(BM25)을 결합하는 방식
- 왜 벡터 검색만으로 부족한지 — 벡터 검색이 놓치는 케이스 (정확한 키워드 매칭 실패, 도메인 용어 처리)
- BM25 알고리즘의 핵심 원리 (TF-IDF 기반, 정확한 토큰 매칭)
- 벡터 검색과 BM25 검색의 장단점 비교

| 검색 방식 | 강점 | 약점 |
|----------|------|------|
| 벡터 검색 (Semantic) | | |
| BM25 (Keyword) | | |
| Hybrid (결합) | | |

- 결합 방식: 가중치 기반 병합, Reciprocal Rank Fusion(RRF) 등

참고 자료:
- [LangChain EnsembleRetriever](https://python.langchain.com/docs/how_to/ensemble_retriever/)
- [Pinecone — Hybrid Search 설명](https://www.pinecone.io/learn/hybrid-search-intro/)
- [BM25 알고리즘 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)

### 2. Re-ranking이란?

- Re-ranking의 정의와 RAG 파이프라인에서의 위치 (Retrieval과 Generation 사이)
- Cross-encoder vs Bi-encoder 차이를 아래 관점에서 비교

| 구분 | Bi-encoder | Cross-encoder |
|------|-----------|---------------|
| 입력 방식 | 질문과 문서를 **각각** 인코딩 | 질문과 문서를 **함께** 인코딩 |
| 속도 | | |
| 정확도 | | |
| 사용 위치 | | |

- 왜 Cross-encoder를 검색 단계가 아닌 Re-ranking 단계에서 사용하는지 (속도 vs 정확도 트레이드오프)
- Two-stage retrieval 패턴: Bi-encoder로 후보 선별 → Cross-encoder로 재정렬
- 상용 Re-ranking API: Cohere Rerank의 동작 방식과 장점

참고 자료:
- [SBERT — Cross-encoder vs Bi-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Cohere Rerank 개념 설명](https://docs.cohere.com/docs/reranking)
- [sentence-transformers CrossEncoder 문서](https://www.sbert.net/docs/cross_encoder/usage/usage.html)

### 3. 메타데이터 필터링과 컨텍스트 압축

Advanced RAG에서는 검색 결과의 양보다 질이 중요합니다. 아래 두 가지 Post-Retrieval 기법을 조사하세요.

| 기법 | 조사 내용 |
|------|----------|
| 메타데이터 필터링 | 청크에 포함된 메타데이터(출처, 날짜, 카테고리 등)를 기반으로 검색 범위를 제한하는 기법. 다년도 문서나 다중 소스 RAG에서 특히 중요 |
| Contextual Compression | 검색된 청크에서 질문과 관련 있는 부분만 추출하여 LLM에게 전달하는 기법. 불필요한 정보를 제거하여 LLM의 답변 정확도를 향상 |

각 기법별로 아래를 포함해주세요.
- **한 줄 정의**
- **왜 필요한지**: 이 기법이 없으면 어떤 문제가 발생하는지
- **구현 방식 예시**: LangChain 또는 LlamaIndex에서 어떻게 구현하는지

참고 자료:
- [Chroma Metadata Filtering](https://docs.trychroma.com/guides)

## 실습 과제 예측

실습 과제(TASK.md)를 보고, 실습 전에 아래 가설을 세워주세요.

1. 2025년과 2026년 문서를 동시에 인덱싱했을 때, 년도 혼동이 가장 많이 발생할 질문 유형을 예측하세요
2. Hybrid Search가 가장 효과를 볼 질문 유형을 예측하세요 (예: 특정 의료 용어가 포함된 질문 vs 일반적 표현의 질문)
3. Re-ranking이 결과를 개선할지, 또는 오히려 악화시킬 수 있는 경우가 있을지 예측하세요
4. 메타데이터 필터링이 년도 혼동 문제를 얼마나 해결할 수 있을지 예측하세요

> 실습 후에 가설과 실제 결과를 비교하여 본인의 제출 README(`week-4/<GithubID>/README.md`)에 포함하여 제출합니다.

## 추가 참고 자료

Advanced RAG 전체 흐름
- [RAG Survey (Gao et al., 2024)](https://arxiv.org/abs/2312.10997) — Naive / Advanced / Modular RAG 분류
- [Advanced RAG Techniques (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/)
- [Chunkviz — 청킹 전략 시각화 도구](https://chunkviz.up.railway.app/)

한국어 임베딩/리랭킹
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — 다국어 임베딩 성능 벤치마크
- Hugging Face에서 "korean cross-encoder" 검색

## 제출 형식

- 제출 README(`week-4/<GithubID>/README.md`)에 이론 과제 답변 포함
- 실습 과제 결과와 함께 제출
- 가설은 실습 전/후 비교 포함
