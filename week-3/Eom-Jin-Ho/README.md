[README.md](https://github.com/user-attachments/files/26461050/README.md)
# 3주차 과제: RAG Indexing 파이프라인 구축

---

# 1. 개요

본 과제에서는 의료급여 PDF 문서를 기반으로 RAG(Retrieval-Augmented Generation) Indexing 파이프라인을 구축하고 검색 성능을 평가하는 것을 목표로 한다.

2주차에서는 system prompt에 전체 데이터를 삽입하는 방식으로 문제를 해결했지만, 데이터가 커질 경우 토큰 제한과 비용 증가 등의 문제가 발생한다.

이를 해결하기 위해 문서를 청킹하고 벡터화하여 필요한 정보만 검색하는 RAG 구조를 적용하였다.

---

# 2. 실행 환경

- Framework: LangChain  
- Embedding Model: text-embedding-3-small  
- Vector Store: FAISS  
- Language: Python 3.13  
- Loader: PyPDFLoader  
- Splitter: RecursiveCharacterTextSplitter  

---

# 3. RAG 이론 정리

## 3.1 RAG란?

### 정의  
RAG(Retrieval-Augmented Generation)는 외부 데이터를 검색(Retrieval)한 뒤, 그 결과를 기반으로 LLM이 답변을 생성(Generation)하는 구조이다.

### 왜 필요한가  
LLM은 기본적으로 학습된 데이터까지만 알고 있고 최신 정보는 모른다. 또한 틀린 정보도 그럴듯하게 말하는 문제가 있다.  

따라서 다음과 같은 목적을 위해 필요하다.

- 최신 정보 반영  
- 특정 도메인 지식 활용 (의료, 법률 등)  
- 할루시네이션 감소  

---

## 3.2 기존 방식 vs RAG

### 2주차 방식  
- 모든 데이터를 한 번에 넣음  
- 토큰 제한  
- 비용 증가  
- 비효율  

### RAG 방식  
- 필요한 정보만 검색  
- 효율적  
- 정확도 향상  

핵심 차이  
전체를 넣느냐 vs 필요한 것만 찾느냐

---

## 3.3 RAG 파이프라인 흐름

### Indexing (사전 준비)

문서 → 청크 → 임베딩 → 벡터DB 저장

### Retrieval + Generation (실행 단계)

질문 → 임베딩 → 검색 → 컨텍스트 → LLM → 답변

요약  
미리 잘 쪼개고 저장해놓고, 질문 들어오면 필요한 것만 꺼낸다

---

# 4. RAG 구성 요소

## 4.1 Chunking

문서를 LLM이 처리할 수 있도록 작은 단위로 나누는 작업

- chunk_size: 한 덩어리 크기  
- chunk_overlap: 겹치는 영역  

overlap이 필요한 이유  
문장이 끊기면 의미가 깨지기 때문이다

---

## 4.2 Embedding

텍스트를 숫자 벡터로 변환하는 것 (의미 기반 표현)

- 비슷한 의미 → 벡터 거리 가까움  

대표 모델  
- OpenAI Embedding  
- BGE  
- Sentence Transformers  

---

## 4.3 Vector Store

벡터를 저장하고 유사도 검색하는 데이터베이스

- FAISS → 빠름 (로컬)  
- Chroma → 간편  

역할  
가장 비슷한 문서를 찾아주는 엔진

---

## 4.4 Retriever

질문과 유사한 문서를 찾는 컴포넌트

- Top-K: 가장 유사한 K개 반환  

특징  
K가 크면 노이즈 증가  
K가 작으면 정보 부족  

---

## 4.5 Generation

검색된 문서를 기반으로 LLM이 답변을 생성하는 단계

LLM은 지식이 아니라 검색 결과를 기반으로 답한다

---

# 5. 실습 전 가설

가설 1  
PDF 청킹 시 표 구조가 깨질 것이다

가설 2  
조건 + 값이 다른 chunk로 분리될 것이다

가설 3  
hard 문제에서 검색 실패가 많이 발생할 것이다

이유  
정보가 여러 chunk에 분산되기 때문

---

# 6. Golden Dataset 구축

## 설계 기준

| 난이도 | 기준 |
|--------|------|
| easy | 단일 조건 |
| medium | 2~3 조건 |
| hard | 다중 조건 + 예외 |

---

## 설계 특징

- 표 기반 문제 구성  
- FAQ 기반 문제 포함  
- hard 문제는 일부러 실패 유도  

---

# 7. RAG Indexing 파이프라인

## 7.1 전체 흐름

PDF → Chunking → Embedding → FAISS 저장

---

## 7.2 문서 로딩

PyPDFLoader를 사용하여 PDF 문서를 로딩하였다.  
총 21페이지의 문서를 불러왔으며, 이후 청킹 및 임베딩 과정에 사용하였다.

---


## 7.3 청킹

- chunk_size: 1000  
- chunk_overlap: 150  
- 총 청크 수: 35  

청킹은 문서를 일정 크기로 분할하여 embedding 단위로 만드는 과정이다.  
chunk_size가 클수록 문맥 유지에는 유리하지만 검색 정확도가 떨어질 수 있고,  
작을수록 검색 정확도는 올라가지만 정보가 여러 chunk로 분산될 수 있다.

---

## 7.4 임베딩 및 벡터 저장

각 chunk를 embedding 벡터로 변환한 뒤 FAISS에 저장하였다.  
FAISS는 벡터 유사도 기반 검색을 수행하는 로컬 벡터 데이터베이스이다.

---

# 8. build / evaluate 분리 이유

본 실습에서는 코드를 다음과 같이 분리하였다.

- build_index.py → 인덱스 생성 (PDF → chunk → embedding → 저장)  
- evaluate_retrieval.py → 검색 및 평가  

---

## 이유

1. 비용 절감  
embedding 과정은 API 호출이기 때문에 비용이 발생한다.  
따라서 인덱스 생성은 1회만 수행하도록 분리하였다.

2. 실험 효율성  
Top-K 값 변경, 평가 반복 등을 빠르게 수행하기 위해  
검색 및 평가 코드를 분리하였다.

3. 구조적 분리  
- Indexing: 오프라인 단계  
- Retrieval: 온라인 단계  

이는 실제 RAG 시스템에서도 사용하는 구조이다.

---

# 9. 검색 성능 평가

## 평가 방식

- Top-K 유사도 검색 수행  
- 검색된 chunk에 evidence_text 포함 여부로 성공 판단  

---

## Top-K = 3

| 질문 ID | 난이도 | 결과 |
|---------|--------|------|
| q01 | easy | 성공 |
| q02 | easy | 성공 |
| q03 | medium | 실패 |
| q04 | medium | 성공 |
| q05 | hard | 실패 |

검색 성공률: 3/5

---

## Top-K = 5

| 질문 ID | 난이도 | 결과 |
|---------|--------|------|
| q01 | easy | 성공 |
| q02 | easy | 성공 |
| q03 | medium | 성공 |
| q04 | medium | 성공 |
| q05 | hard | 실패 |

검색 성공률: 4/5

---

# 10. 실패 원인 분석

## q03 (medium)

- 연령별 본인부담률 정보가 표 내부에 존재  
- 하나의 chunk 안에 다양한 정보가 포함되어 있어 retrieval 정확도가 낮아짐  
- Top-K를 증가시키면서 해결됨  

---

## q05 (hard)

- 입원 본인부담률 → 표에 존재  
- 식대 본인부담률 → FAQ에 존재  

두 정보가 서로 다른 chunk에 존재하여  
동시에 retrieval되지 못함  

---

# 11. 핵심 인사이트

1. Chunking의 영향  

- chunk_size가 클수록  
  - 문맥 유지 ↑  
  - 검색 정확도 ↓  

- chunk_size가 작을수록  
  - 검색 정확도 ↑  
  - 정보 분산 ↑  

---

2. Top-K의 영향  

Top-K 값을 증가시키면  
더 많은 context를 확보할 수 있어 검색 성공률이 증가한다.

---

3. 데이터 구조의 영향  

- 표 데이터 → chunk 분할에 매우 민감  
- FAQ 데이터 → 자연어 형태라 검색에 유리  

---

4. Hard 문제 특징  

- 단일 chunk로 해결 불가능  
- 여러 chunk 결합 필요  
- retrieval 한계 발생  

---

# 12. 가설 검증

| 가설 | 결과 |
|------|------|
| 표 구조가 깨질 것이다 | 발생 |
| 정보가 분산될 것이다 | 발생 |
| hard 문제에서 실패할 것이다 | 발생 |

실습 전 가설과 실제 결과가 대부분 일치하였다.

---

# 13. 최종 결론

- RAG Indexing 파이프라인을 정상적으로 구축하였다  
- Top-K 조정을 통해 검색 성능이 개선됨을 확인하였다  
- Hard 문제에서 retrieval 기반 RAG의 한계를 확인하였다  

---

# 14. 최종 인사이트

RAG의 성능은 모델보다 데이터 구조와 chunk 설계에 더 크게 영향을 받는다
