# 5주차: RAG 평가 — Golden Dataset, LLM-as-a-Judge, Ragas
---

## 목차

1. [실행 환경 및 모델 정보](#1-실행-환경-및-모델-정보)
2. [Golden Dataset 확장 전략](#2-golden-dataset-확장-전략)
3. [Ragas 평가 파이프라인](#3-ragas-평가-파이프라인)
4. [Step 2: 메트릭 결과](#4-step-2-메트릭-결과)
5. [Step 3: Basic vs Advanced 비교 분석](#5-step-3-basic-vs-advanced-비교-분석)
6. [Step 4: 실패 케이스 Deep Dive](#6-step-4-실패-케이스-deep-dive)
7. [이론 과제 답변](#7-이론-과제-답변)
8. [가설 vs 실제 결과 비교](#8-가설-vs-실제-결과-비교)
9. [심화 A: Custom Metric — YearAccuracy](#9-심화-a-custom-metric--yearaccuracy)

---

## 1. 실행 환경 및 모델 정보

| 항목 | 내용 |
|------|------|
| **생성용 LLM** | `gpt-4o` (temperature=0) |
| **평가용 LLM** | `gpt-4o-mini` (temperature=0) |
| **임베딩 모델** | `text-embedding-3-small` (OpenAI) |
| **Ragas 버전** | 0.4.x (v0.2+ 스키마 사용) |
| **LangChain** | langchain-community, langchain-openai, langchain-cohere |
| **벡터 저장소** | ChromaDB (`./chroma_db_week5`) |
| **BM25** | `langchain_community.retrievers.BM25Retriever` |
| **Re-ranker** | Cohere `rerank-v3.5` (Advanced RAG) |
| **청크 설정** | size=500, overlap=100 |
| **평가 병렬도** | `RunConfig(max_workers=4, timeout=60)` |
| **데이터 소스** | `2025 알기 쉬운 의료급여제도.pdf`, `2026 알기 쉬운 의료급여제도.pdf` |

**생성용/평가용 LLM을 분리한 이유:** 동일 모델이 생성 및 채점을 동시에 담당하면 자기 결과물에 대한 Self-bias가 발생할 수 있다. Claude와 Gemini 모두 잦은 호출로 429 에러 발생 + 비용 문제로, GPT-4o로 생성하고 GPT-4o-mini로 채점하였다. 20문항 × 5메트릭 × 2파이프라인 기준 약 수백 회의 LLM 호출이 연쇄적으로 발생하기 때문에 평가용 모델은 Rate Limit이 넉넉한 gpt-4o-mini를 선택했다.

---

## 2. Golden Dataset 확장 전략

4주차 20문항을 기반으로 Ragas v0.2+ 표준 스키마(`ground_truth`, `ground_truth_contexts`)를 추가했다.

### 2-1. `ground_truth` 정제 원칙

Ragas Answer Correctness는 의미 유사도(임베딩)와 사실 일치도(LLM)의 가중 평균이기 때문에, `ground_truth`가 단어 하나짜리 단답형이면 RAG 답변보다 훨씬 짧아 의미 유사도가 부당하게 낮게 측정된다.

**정제 원칙: `연도 + 대상 + 조건 + 값` 순서로 완전한 한 문장**

| 형태 | 예시 |
|------|------|
| 단답형 | `"1,000원"` |
| 완전한 문장 | `"2025년 기준 의료급여 1종 수급권자의 1차 의료급여기관(의원) 외래 본인부담금은 1,000원입니다."` |
| 장황한 장문 | 2~3줄 이상 → RAG 답변보다 길어져 유사도 하락 |

### 2-2. `ground_truth_contexts` 발췌 원칙

- **PDF 원본 직접 발췌:** 벡터 DB의 검색 결과가 아닌 실제 PDF 원문에서 정답의 근거가 되는 문단을 2~5문장 단위로 추출한다. 시스템이 틀린 문서를 가져왔을 때 Context Recall이 낮게 나오는 기준을 세우는 것이 목적이다.
- **리스트 형태 유지:** 단일 근거라도 리스트(`["..."]`)로 저장한다.
- **Cross-year 문항 처리:** "2025년과 2026년의 차이"를 묻는 문항의 경우 두 연도 근거를 모두 포함한다. 이를 통해 Ragas가 양쪽 연도 검색 성공 여부를 동시에 판단할 수 있다.

### 2-3. 데이터셋 구성

| 난이도 | 문항 수 | 설명 |
|--------|---------|------|
| `easy` | 5 | 단일 수치 단답형 |
| `medium` | 6 | 맥락 이해 필요 |
| `hard` | 6 | 복합 조건 추론 |
| `cross-year` | 3 | 2025/2026 비교 |
| **합계** | **20** | |

### 2-4. 샘플 (golden_dataset_v2.jsonl)

```jsonl
{"question": "2026년에 확대된 조산아 및 저체중 출생아의 의료급여 지원 기간은 총 얼마인가요?", "ground_truth": "2026년부터 조산아 및 저체중 출생아의 의료급여 지원 기간은 기존 출생일로부터 5년에서 5년 4개월로 확대되었습니다.", "ground_truth_contexts": ["6 의료급여 조산아 지원 기간 확대('26. 1. 1. 시행)", "○ 조산아 맞춤형 의료급여 지원을 위해 출생일로부터 5년→5년 4개월 기간 확대"], "difficulty": "hard", "source_year": "2026"}
{"question": "2025년과 2026년의 의료급여 본인부담금 환급 산정기간의 차이는 무엇인가요?", "ground_truth": "의료급여 초과 본인부담금 환급 산정기간이 2025년에는 30일이었으나, 2026년부터는 월단위로 변경되었습니다.", "ground_truth_contexts": ["5 의료급여 초과 본인부담금 환급 산정기간 변경('26. 1. 1. 시행)", "○ 본인부담금 일정 금액 초과 시 초과 금액 환급하는 산정기간을 30일→월단위 변경"], "difficulty": "cross-year", "source_year": "2025+2026"}
```

---

## 3. Ragas 평가 파이프라인

### 3-1. BM25 필터링 누수 문제 해결 (Post-Filtering)

ChromaDB는 `filter` 파라미터로 `source_year` 메타데이터 필터링이 가능하지만, 메모리에 올린 BM25는 메타데이터 필터링을 네이티브로 지원하지 않는다. 이전 주차에도 이 때문에 하이브리드 검색 시 BM25가 잘못된 연도의 문서를 섞는 "필터링 누수"가 발생했다.

해결책으로 Python 후처리(Post-Filtering)를 구현했다.

```python
# 질문에서 연도 파악
year_filter = None
if "2025" in question and "2026" not in question:
    year_filter = "2025"
elif "2026" in question and "2025" not in question:
    year_filter = "2026"

# BM25 결과에서 잘못된 연도 강제 제거
bm25_docs = (
    [d for d in raw_bm25_docs if d.metadata.get("source_year") == year_filter]
    if year_filter else raw_bm25_docs
)
```

이 조치로 Ragas Context Precision이 다른 연도 청크 혼입으로 인해 불필요하게 낮게 산정되는 것을 방지했다.

### 3-2. 데이터셋 구성 (SingleTurnSample 매핑)

Ragas v0.2+는 `SingleTurnSample` 스키마를 요구한다. JSONL 필드와의 매핑은 다음과 같다.

| JSONL 필드 | `SingleTurnSample` 필드 | 출처 |
|-----------|------------------------|------|
| `question` | `user_input` | Golden Dataset |
| (실행 결과) | `response` | RAG 파이프라인 출력 |
| (실행 결과) | `retrieved_contexts` | RAG 파이프라인 출력 |
| `ground_truth` | `reference` | Golden Dataset |
| `ground_truth_contexts` | `reference_contexts` | Golden Dataset |

### 3-3. 평가 실행 구성

```python
metrics = [
    ContextRecall(),
    LLMContextPrecisionWithReference(),
    Faithfulness(),
    ResponseRelevancy(),
    AnswerCorrectness(),
]
run_cfg = RunConfig(max_workers=4, timeout=60, max_retries=10, max_wait=30)

result = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=metrics,
    llm=eval_llm,
    embeddings=eval_embeddings,
    run_config=run_cfg,
)
result.to_pandas().to_csv("basic_ragas_scores.csv", index=False, encoding="utf-8-sig")
```

---

## 4. Step 2: 메트릭 결과

### 4-1. 전체 평균

| 메트릭 | Basic | Advanced | 변화 |
|--------|-------|----------|------|
| Context Recall | 1.0000 | 1.0000 | → |
| Context Precision | 0.7855 | **0.9292** | ↑ +0.1437 |
| Faithfulness | 0.6417 | **0.7000** | ↑ +0.0583 |
| Answer Relevancy | **0.4482** | 0.4422 | ↓ -0.0060 |
| Answer Correctness | 0.7438 | **0.7578** | ↑ +0.0140 |

### 4-2. 문항별 상세 점수

| 질문 ID | 질문 요약 | 난이도 | CR(B/A) | CP(B/A) | Faith(B/A) | AR(B/A) | AC(B/A) |
|---------|-----------|--------|---------|---------|-----------|---------|---------|
| q01 | 2025 1종 의원급 외래 | easy | 1.0/1.0 | 0.806/0.833 | 0.00/1.00 | 0.511/0.516 | 0.976/0.976 |
| q02 | 2026 1종 의원급 외래 | easy | 1.0/1.0 | 1.000/1.000 | 1.00/1.00 | 0.515/0.517 | 0.727/0.727 |
| q03 | 2025 장기지속형 주사제 | medium | 1.0/1.0 | 1.000/0.833 | 0.00/0.00 | 0.431/0.360 | 0.992/0.841 |
| q04 | 2026 장기지속형 주사제 | medium | 1.0/1.0 | 0.833/1.000 | 1.00/1.00 | 0.419/0.419 | 0.993/0.993 |
| q05 | 2026 연간 365회 초과 | medium | 1.0/1.0 | 1.000/0.833 | 1.00/1.00 | 0.468/0.453 | 0.702/0.702 |
| q06 | 2025 조산아 외래 본인부담 | hard | 1.0/1.0 | **0.478/1.000** | 0.00/1.00 | 0.000/0.469 | **0.038/0.714** |
| q07 | 2026 조산아 지원 기간 | hard | 1.0/1.0 | **0.200/1.000** | 1.00/1.00 | 0.498/0.492 | 0.579/0.968 |
| q08 | 2025 우울증·조기정신증 추가 | medium | 1.0/1.0 | 0.833/1.000 | 1.00/1.00 | 0.479/0.475 | 0.995/0.994 |
| q09 | 2026 이상지질혈증 추가 | hard | 1.0/1.0 | **0.250/1.000** | 1.00/**0.00** | 0.476/0.000 | **0.961/0.038** |
| q10 | 2025 노숙인 유효기간 | easy | 1.0/1.0 | 1.000/1.000 | 0.50/1.00 | 0.508/0.515 | 0.744/0.744 |
| q11 | 2026 노숙인 유효기간 | easy | 1.0/1.0 | 1.000/1.000 | 0.50/0.50 | 0.444/0.413 | 0.615/0.614 |
| q12 | 2026 KTAS 4 기준 변경 | hard | 1.0/1.0 | 0.888/1.000 | 1.00/1.00 | 0.318/0.364 | 0.454/0.509 |
| q13 | 장기지속형 인하폭(교차) | cross-year | 1.0/1.0 | 0.756/1.000 | 1.00/**0.00** | 0.371/0.376 | 0.964/0.712 |
| q14 | 환급 산정기간 차이(교차) | cross-year | 1.0/1.0 | 0.833/1.000 | 0.33/0.50 | 0.655/0.586 | 0.503/0.963 |
| q15 | 2025 2종 아동 식대 | easy | 1.0/1.0 | 1.000/1.000 | 1.00/1.00 | 0.467/0.467 | 0.709/0.707 |
| q16 | 2026 1종 임플란트 | easy | 1.0/1.0 | 1.000/1.000 | 0.00/0.00 | 0.431/0.392 | 0.961/0.961 |
| q17 | 2025 2종 제2차 외래 15% | easy | 1.0/1.0 | 1.000/0.583 | 1.00/1.00 | 0.545/0.542 | 0.982/0.735 |
| q18 | 2026 2인실 상급종합 50% | medium | 1.0/1.0 | 0.417/0.500 | 0.50/0.50 | 0.549/0.549 | 0.977/0.977 |
| q19 | 2025 잠복결핵 본인부담 | easy | 1.0/1.0 | 0.417/1.000 | 1.00/1.00 | 0.388/0.459 | 0.496/0.696 |
| q20 | 2025+2026 노숙인 유효기간(교차) | cross-year | 1.0/1.0 | 1.000/1.000 | **0.00**/0.50 | 0.491/0.483 | 0.511/0.585 |

> CR = Context Recall, CP = Context Precision, Faith = Faithfulness, AR = Answer Relevancy, AC = Answer Correctness. B = Basic, A = Advanced.

### 4-3. 4주차 수동 채점 vs Ragas Answer Correctness 비교

| 질문 ID | 질문 요약 | 4주차 판정 | Ragas AC (Basic) | 일치 여부 | 불일치 원인 |
|---------|-----------|-----------|-----------------|-----------|-----------|
| q17 | 2종 외래 본인부담률 15% | 정답 | 0.982 | ✅ 일치 | — |
| q20 | 노숙인 유효기간 교차 비교 | 부분 정답(△) | 0.511 | ✅ 일치 | — |
| q07 | 조산아 지원 기간 | 정답 | 0.579 (Basic) / 0.968 (Adv) | ⚠️ Basic 불일치 | 정답 청크 순위 밀림 → Precision 0.20 |
| q09 | 이상지질혈증 추가 | 정답 | 0.961 (Basic) / 0.038 (Adv) | ⚠️ Adv 불일치 | Sticky Header 텍스트 노이즈로 LLM 답변 거부 |
| q06 | 조산아 외래 본인부담 | 오답 | 0.038 | ✅ 일치 | — |

**관찰:** 사람이 맞다고 보았던 문항도 Ragas는 사실 누락(FN)이나 불필요한 추가 정보(FP)를 잡아내 보수적으로 감점했다. 특히 교차 비교 문항(q20)에서 사람은 핵심 날짜만 대략 맞으면 정답 처리했지만 Ragas는 정확한 날짜 수치 일치 여부까지 따졌다. 반면 일부 hard 문항(q09 Advanced)은 사람이 괜찮다고 판단했을 답변임에도 Ragas가 0.038을 부여하는 극단적 불일치가 발생했는데, 이는 청킹 노이즈로 LLM이 답변을 아예 거부한 데 따른 것이다.

---

## 5. Step 3: Basic vs Advanced 비교 분석

### 5-1. 다차원 비교

| 구분 | 메트릭 | 변화 | 해석 |
|------|--------|------|------|
| **개선** | Context Precision | +0.1437 | 리랭킹이 정답 청크를 상위로 고정 |
| **개선** | Faithfulness | +0.0583 | 정확한 연도 문서를 가져와 환각 감소 |
| **개선** | Answer Correctness | +0.0140 | 전반적 정답 품질 소폭 향상 |
| **유지** | Context Recall | 0.000 | 두 파이프라인 모두 관련 문서를 찾음 |
| **소폭 악화** | Answer Relevancy | -0.0060 | 사실상 유의미하지 않은 수준 |

### 5-2. 년도 혼동 재진단

| 문항 | 현상 | 어느 메트릭에 반영됐나 |
|------|------|----------------------|
| q20 (Basic) | 2025년 유효기간을 "2025.3.22~2026.3.21"로 잘못 추론 | **Faithfulness 0.0** |
| q09 (Advanced) | 2026 문서 청크 내 "2025년" 헤더 텍스트로 LLM 답변 거부 | **Faithfulness 0.0, AC 0.038** |
| q13 (Advanced) | 두 연도 본인부담률 비교 시 2025년 수치 누락 | **Faithfulness 0.0** |

**Ragas 기본 메트릭의 한계:** Ragas의 4대 메트릭은 답변의 의미(Semantic)를 평가하는 데는 탁월하지만, "연도 숫자(2025 vs 2026)를 정확히 구분했는가"를 직접적으로 수치화하지는 못한다. Faithfulness가 0이 되는 것은 연도 혼동의 결과이지 연도 혼동 자체를 감지하는 지표는 아니다. 이것이 YearAccuracy 커스텀 메트릭이 필요한 이유다(→ 심화 A 참고).

### 5-3. 인사이트

**"Advanced가 낫다"는 결론의 유효 범위:** Context Precision(0.786 → 0.929)과 Faithfulness(0.642 → 0.700)에서는 분명한 개선이 있다. 특히 q06, q07, q09처럼 Precision이 0.2~0.5에 불과했던 문항들이 Advanced에서 1.0으로 도달한 것은 리랭킹의 명백한 효과다. 그러나 Answer Relevancy(0.448 → 0.442)는 사실상 불변이고, Answer Correctness 개선폭(+0.014)도 미미하다. 즉, Advanced는 "검색의 질"은 확실히 높였지만 "최종 답변 정확도"의 전면적 도약에는 이르지 못했다.

**프로덕션 가능 여부:** 도메인 임계값을 Faithfulness ≥ 0.9로 설정하면 Basic(0.642)과 Advanced(0.700) 모두 기준 미달이다. 연도 혼동 케이스가 Faithfulness를 0으로 만드는 극단적 실패가 반복되는 한 현재 파이프라인은 프로덕션 투입이 어렵다. Answer Correctness도 평균 0.76 수준으로, 실제 의료 정보 서비스의 요구 수준(≥ 0.9)에 미달한다.

**개선 우선순위:** 가장 시급한 것은 데이터 품질(Parsing & Chunking)이다. q09 케이스에서 입증했듯이, 청크 내 이전 연도 소제목이 텍스트 노이즈로 남아있으면 아무리 리랭킹이 정확한 문서를 가져와도 LLM이 혼동한다. 청킹 단계에서 헤더/소제목 제거 후처리를 추가하는 것이 Faithfulness 개선에 가장 직접적인 효과를 낼 것이다.

---

## 6. Step 4: 실패 케이스 Deep Dive

### Case A: q09 — Advanced에서 오히려 악화된 케이스

```
질문: 2026년 건강검진 후 확진검사 면제 대상에 새롭게 추가된 질환 의심 항목은 무엇인가요?
참고 정답: 2026년 건강검진 후 확진검사 면제 대상에 새롭게 추가된 질환 의심 항목은 이상지질혈증입니다.
```

**검색된 청크 — Basic RAG (Top-3 예시)**
1. `[2025년] 질환 확진검사 본인부담 면제 ... 우울증·조기정신증 추가`
2. `[2026년] 이상지질혈증 질환을 의심하는 경우도 추가하여 ...`
3. `[2025년] 고혈압·당뇨병·결핵 확진 ...`

**검색된 청크 — Advanced RAG (Reranked Top-3)**
1. `[2026년] 04 2025년 변경된 의료급여제도 ... 이상지질혈증 질환 추가 ...`
2. `[2026년] 당뇨병 질환 확진을 위한 경우에 검사 항목 추가 ...`
3. `[2026년] 국가건강검진 후 확진검사 ...`

**생성된 답변**
- Basic: "2026년 건강검진 후 확진검사 면제 대상에 새롭게 추가된 항목은 이상지질혈증입니다." (정답)
- Advanced: "정보를 찾을 수 없습니다." (오답)

**메트릭 점수**

| 메트릭 | Basic | Advanced |
|--------|-------|----------|
| Context Precision | 0.250 | 1.000 |
| Faithfulness | 1.000 | 0.000 |
| Answer Correctness | 0.961 | 0.038 |

**원인 분석:** 검색(Retrieval) 단계는 완벽히 성공했다. Advanced RAG는 2026년 문서만 필터링하여 관련 청크를 1위로 올리는 데 성공했다. 그러나 정답이 담긴 청크의 맨 위에 "**04 2025년 변경된 의료급여제도**"라는 소제목 텍스트가 함께 들어있었다. LLM은 질문이 "2026년" 항목을 묻는데 청크에 "2025년"이라는 글자가 박혀 있는 것을 보고 "컨텍스트가 질문과 맞지 않다"고 판단해 답변을 거부했다. 이는 청킹 단계에서 발생한 **데이터 노이즈** 문제이며, Faithfulness가 검색-생성 경계에서 이 실패를 가장 명확히 드러냈다.

**조치:** PDF 파싱 단계에서 페이지 헤더, 섹션 소제목 등 탐색적 텍스트를 정규표현식으로 제거하는 전처리 추가. 또는 청크 메타데이터에 `section_header` 필드를 분리 저장하고 본문과 혼합되지 않도록 분리 관리.

---

### Case B: q20 — 년도 혼동 (Context 맞는데 답변 년도 틀림)

```
질문: 2025년과 2026년의 노숙인 진료시설 지정 유효기간을 각각 답하세요.
참고 정답: 2025년 기준 노숙인 진료시설 지정 유효기간은 2024년 3월 22일부터 2025년 3월 21일까지이며,
          2026년 기준 유효기간은 2025년 3월 22일부터 2028년 3월 21일까지입니다.
```

**검색된 청크 — Basic RAG**
1. `[2025년] 유효기간 1년 연장 (유효기간: 2024.3.22.~2025.3.21.)`
2. `[2026년] 유효기간 3년 연장 (유효기간: '25. 3. 22.~ '28. 3. 21.)`

**생성된 답변**
- Basic: "2025년의 유효기간은 **2025년 3월 22일부터 2026년 3월 21일**까지입니다."
- (원본 문서: 2024년 3월 22일 ~ 2025년 3월 21일 → 시작 날짜를 잘못 추론)

**메트릭 점수**

| 메트릭 | Basic | Advanced |
|--------|-------|----------|
| Faithfulness | **0.000** | 0.500 |
| Answer Correctness | 0.511 | 0.585 |

**원인 분석:** 검색은 성공했다. 정답 근거 문서가 검색되었음에도 LLM이 "질문이 2025년 유효기간을 물었는데 문서에 2024년 시작이라고 나오니 2025년 기준으로 날짜를 예쁘게 맞춰야겠다"는 잘못된 추론을 수행했다. 이것이 Faithfulness 0.0의 이유다. Ragas의 판정 메커니즘은 LLM 답변에서 "시작일이 2025년 3월 22일"이라는 주장을 추출한 뒤, 원본 문서에서 "2024년 3월 22일"이라는 내용과 불일치함을 검출해 0점을 부여했다. **생성 단계의 과도한 추론(hallucination)**이 원인이며, 프롬프트에 "날짜를 임의로 조정하지 말고 원문 그대로 인용하라"는 제약을 추가하는 것이 효과적이다.

---

### 공통 교훈

- **Ragas가 놓치는 실패 유형:** 답변 자체는 컨텍스트에 충실하지만 연도 숫자를 잘못 구분한 경우, Faithfulness가 1.0이 나올 수 있다. 연도 혼동 탐지에는 YearAccuracy 같은 도메인 특화 커스텀 메트릭이 필수다.
- **Context Precision ≠ Answer Correctness:** q09에서 Advanced의 Context Precision이 1.0(완벽)임에도 Answer Correctness는 0.038이었다. 검색이 완벽해도 청킹 노이즈가 있으면 생성이 실패한다. "얼마나 잘 가져오는가"보다 "DB에 들어있는 데이터가 얼마나 깨끗한가"가 최종 품질을 결정한다.
- **4주차 수동 채점 vs 5주차 자동:** Ragas가 사람보다 더 엄격하다. 사람은 핵심 수치만 맞으면 정답 처리하는 경향이 있지만, Ragas의 Answer Correctness는 사실 누락(FN)과 불필요한 추가 정보(FP)를 모두 감점 요인으로 잡는다. 다만 극단적 사례(q09 Advanced 0.038)처럼 LLM이 답변 자체를 거부한 케이스는 Ragas가 지나치게 가혹하게 채점하는 부작용도 있다.
- **Answer Relevancy의 낮은 변별력:** 평균 0.44~0.45로 두 파이프라인 간 차이가 거의 없었고, 정답 문항에서도 낮은 경우가 많았다. 한국어 도메인에서 이 지표의 신뢰성을 재검토할 필요가 있다.
- **Cross-year 문항의 높은 난이도:** q13, q14, q20 등 두 연도를 동시에 다루는 문항에서 Basic의 Faithfulness 실패율이 높았다. 두 연도 정보를 하나의 답변에 통합하는 작업이 현재 파이프라인에서 가장 취약한 부분이다.

---

## 7. 이론 과제 답변

### 7-1. Golden Dataset

**한 줄 정의:** 질문, 기대 정답(`ground_truth`), 정답의 원문 근거(`ground_truth_contexts`)를 세트로 구성한 고품질 기준 데이터셋으로, RAG 시스템의 변경사항이 성능 회귀를 일으켰는지 객관적으로 검증하는 기준점이다.

**없으면 생기는 문제:** 프롬프트·모델·검색 설정을 바꿨을 때 성능이 나아졌는지 나빠졌는지를 근거 있게 판단할 수 없다. 채점 기준이 매번 달라지고, 무엇을 고쳐야 할지 방향을 잡기 어렵다. (참고: [Ragas — Schema](https://docs.ragas.io/en/stable/concepts/metrics/overview/))

**필수 스키마 (Ragas v0.2+ 기준):**

| 필드 (JSONL) | `SingleTurnSample` 매핑 | 준비 주체 | 어느 메트릭에 필수 |
|-------------|------------------------|---------|-----------------|
| `question` | `user_input` | 사람 | 전 메트릭 |
| `ground_truth` | `reference` | 사람 (완전한 문장) | Context Recall, Answer Correctness |
| `ground_truth_contexts` | `reference_contexts` | **수동 어노테이션** | Context Recall, Context Precision |
| (RAG 출력) | `response` | 파이프라인 | Faithfulness, Answer Relevancy |
| (RAG 출력) | `retrieved_contexts` | 파이프라인 | 전 메트릭 |

> v0.1에서는 `question`/`answer`/`contexts`/`ground_truth` 필드명을 사용했으나, v0.2+에서 `user_input`/`response`/`retrieved_contexts`/`reference`로 변경되었다. 인터넷 블로그 튜토리얼 대부분이 v0.1 기준이므로 주의가 필요하다.

**권장 규모 (Ragas 공식 문서 및 실무 기준):**

| 단계 | 문항 수 | 근거 |
|------|---------|------|
| 초기/파일럿 | 10~20 | 스키마·비용 검증, API 한도 확인 |
| 성숙 | 50~100 | 난이도 분포 확보, 회귀 테스트 신뢰도 |
| 대규모 | 200+ | 서브도메인별 분리 평가 가능 |

(참고: [Ragas — Testset Generation](https://docs.ragas.io/en/stable/concepts/test_data_generation/))

**좋은 Dataset의 조건:**
- **실제 사용자 질문 반영:** 실무에서 자주 묻는 유형 포함
- **함정 케이스 포함:** 연도 혼동, 비슷한 수치 혼동 등 실패하기 쉬운 케이스
- **회귀 이력 추적:** 한 번 틀렸던 문항을 데이터셋에 남겨 개선 이후 재발 방지
- **난이도 균형:** easy/medium/hard/cross-year 고른 분포
- **완전한 문장 형태의 `ground_truth`**

**`ground_truth_contexts` 수동 어노테이션이 필수인 이유:** Context Recall 메트릭은 "정답에 필요한 내용이 검색 결과에 포함됐는가"를 측정하는데, 이 "필요한 내용"을 LLM이 자동으로 생성하면 순환 논리(circular reasoning)가 된다. 즉, LLM이 "이것이 근거 문단이어야 한다"고 판단한 것을 기준으로 삼으면, 동일 LLM 계열의 RAG 시스템이 가져온 청크는 항상 기준과 유사하게 보이므로 Context Recall이 과대평가된다. 사람이 PDF 원문에서 직접 발췌해야 독립적이고 신뢰할 수 있는 기준이 된다.

---

### 7-2. 평가의 필요성과 LLM-as-a-Judge

#### 2-1. 왜 체계적 평가가 필요한가

전통적인 소프트웨어는 입력이 결정되면 출력도 결정되는 **결정론적(deterministic)** 특성을 가지므로 단위 테스트로 충분히 검증할 수 있다. 반면 LLM 기반 시스템은 같은 입력에도 확률적으로 다른 출력을 내는 **확률론적(stochastic)** 특성을 가진다. 따라서 전통적 단위 테스트만으로는 충분하지 않으며, 체계적 평가 없이는 다음 문제가 발생한다.

- **회귀 탐지 불가:** 프롬프트나 청킹 설정을 바꿨을 때 성능 변화를 숫자로 입증할 수 없다.
- **디버깅 지점 모호:** 답변이 틀렸을 때 검색이 원인인지, 생성이 원인인지 알 수 없다.
- **비교 기준 부재:** 여러 모델이나 파라미터 중 무엇이 나은지 직관으로만 판단하게 된다.
- **프로덕션 도입 판단 불가:** 객관적인 "이 정도면 쓸 수 있다"의 기준이 없다.

회귀 검증(regression test)이란 변경 이후에도 기존에 통과했던 케이스가 여전히 통과하는지 확인하는 것으로, Golden Dataset이 이 회귀 검증의 기준점 역할을 한다.

#### 2-2. LLM-as-a-Judge

**등장 배경:** 기존의 자동 평가 지표인 BLEU, ROUGE는 n-gram 겹침만 측정하므로 "1,000원"과 "천 원"처럼 표현만 다른 동의어를 오답 처리하는 한계가 있다. 반면 사람이 직접 채점하는 방식은 비용과 시간이 과도하게 들어 확장성이 떨어진다. LLM-as-a-Judge는 이 두 문제를 동시에 해결하기 위해 등장했다. (참고: Zheng et al. 2023, "Judging LLM-as-a-Judge", MT-Bench, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685))

**작동 원리:** 강력한 LLM(주로 GPT-4 계열)을 심판으로 세우고 루브릭(rubric)과 판정 기준을 프롬프트로 제공한다. 심판 LLM은 질문, 기대 정답, 생성된 답변 세 가지를 비교하여 점수와 근거를 출력한다. (참고: Liu et al. 2023, "G-Eval", [arXiv:2303.16634](https://arxiv.org/abs/2303.16634))

**좋은 루브릭 vs 나쁜 루브릭 예시:**

| 구분 | 예시 |
|------|------|
| ❌ 나쁜 루브릭 | "답변이 정확하면 1점, 부정확하면 0점" (모호한 기준) |
| ✅ 좋은 루브릭 | "정답에 포함된 사실(TP)과 누락된 사실(FN), 답변에만 있는 사실(FP)을 각각 추출하여 F1 방식으로 점수를 계산하라" (측정 가능한 구체적 기준) |
| ❌ 나쁜 루브릭 | "좋으면 5점, 보통이면 3점" (척도 정의 없음) |
| ✅ 좋은 루브릭 | "다음 기준을 Chain-of-Thought로 순서대로 판단하라: 1) 핵심 수치가 일치하는가, 2) 연도가 명시되어 있는가, 3) 불필요한 정보가 없는가" (단계적 판단 기준) |

**사람 평가와의 일치율:** GPT-4 Judge는 사람 평가와 약 80% 이상의 일치율을 보이는 것으로 알려져 있다. MT-Bench 논문(Zheng et al. 2023)에 따르면 GPT-4는 전문가 평가와의 일치율에서 인간 평가자 간 일치율에 필적하는 수준을 달성했다.

**한계와 신뢰성 문제:**
- 동일한 모델로 생성과 평가를 동시에 수행하면 Self-bias 발생
- 프롬프트 표현 방식에 따라 점수가 달라지는 불안정성
- 온도(temperature) 설정에 따른 비결정성
- API 호출 비용

**Ragas와의 관계:** Ragas의 4대 메트릭 중 LLM Judge 기반은 Faithfulness(claim 분해 후 컨텍스트 대조), Context Precision(청크별 관련성 판단), Answer Correctness(TP/FP/FN 분류 + 의미 유사도)이다. Context Recall은 LLM에게 각 ground_truth_contexts 문장이 검색 결과에서 커버되는지를 판단시키므로 이 역시 LLM Judge 원리를 활용한다. Answer Relevancy는 역방향 질문 생성 방식으로 임베딩 기반 유사도와 LLM을 혼합해 사용한다.

---

### 7-3. Ragas 4대 메트릭 (+ Answer Correctness)

#### 3-1. 검색 단계

| 구분 | Context Recall | Context Precision |
|------|---------------|------------------|
| **정의** | 정답에 필요한 내용(`ground_truth`)이 검색된 청크 안에 얼마나 커버되었는가 | 검색된 청크 중 관련 있는 청크가 얼마나 상위에 배치됐는가 |
| **계산 방식** | LLM이 각 `reference_contexts` 문장을 검색 결과에서 찾을 수 있는지 판단 → `(찾은 문장 수)/(전체 ground_truth 문장 수)` | 각 청크를 순서대로 순회하며 LLM이 관련성 판단(1/0) → Mean Average Precision(MAP) 방식 계산. 관련 청크가 하위에 있을수록 더 크게 감점 |
| **낮을 때 의심할 점** | 청크 크기가 너무 작아 정답 근거 문단이 분절됨, 임베딩 모델의 한국어 도메인 표현력 부족, 검색 k 값이 너무 작음 | 노이즈 청크가 상위를 차지함, Re-ranker 부재, 청크 내 혼재된 주제로 관련성 판단 오류 |
| **개선 기법** | 청크 크기·오버랩 조정, 한국어 특화 임베딩 모델 교체, k 값 증가, 하이브리드 검색(BM25 + 벡터) 도입 | Re-ranking(Cohere Rerank 등) 적용, 메타데이터 필터링으로 노이즈 차단, 청킹 단계에서 의미 단위 분리 강화 |

(참고: [Ragas — Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/), [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/))

#### 3-2. 생성 단계

| 구분 | Faithfulness | Answer Relevancy |
|------|-------------|-----------------|
| **정의** | 생성된 답변의 모든 주장이 검색된 컨텍스트에 의해 뒷받침되는가 (환각 탐지) | 생성된 답변이 원래 질문이 요구하는 바를 정확히 다루고 있는가 |
| **계산 방식** | ① LLM이 답변에서 독립적 주장(claim)들을 추출 → ② 각 주장이 컨텍스트 내에 근거가 있는지 다시 LLM이 판단 → `(컨텍스트로 뒷받침된 주장 수)/(전체 주장 수)`. 2단계 LLM 호출로 비용이 가장 큰 메트릭 중 하나 | LLM이 답변을 보고 역으로 질문을 여러 개 생성 → 생성된 가상 질문들과 원래 질문의 임베딩 유사도 평균 계산 |
| **낮을 때 의심할 점** | LLM이 사전 지식으로 컨텍스트에 없는 정보를 덧붙임(환각), 교차 비교 문항에서 LLM이 날짜/수치를 추론으로 "정리"하는 경향 | 답변이 질문과 무관한 내용 위주로 구성됨, 답변 거부("정보를 찾을 수 없습니다")가 다수 | 
| **개선 기법** | 프롬프트에 "컨텍스트에 없는 정보는 절대 추가하지 말라"는 강제 지시 추가, 날짜/수치는 원문 그대로 인용하도록 지시 | 질문에 직접 답하도록 프롬프트 개선, 불필요한 배경 설명 최소화 지시 |

(참고: [Ragas — Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/), [Answer Relevance](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/))

#### 3-3. End-to-End: Answer Correctness

**정의:** 생성된 답변이 `ground_truth`와 얼마나 일치하는가를 나타내는 최종 지표.

**계산 방식:** 두 가지 점수를 가중 평균한다.
1. **사실 기반 점수 (Factual Correctness):** LLM이 답변과 정답을 claim 단위로 분해하여 TP(정답에도 있고 답변에도 있음) / FP(답변에만 있음, 환각 또는 불필요한 정보) / FN(정답에는 있는데 답변에서 누락)을 분류 → F1 방식으로 점수 산출
2. **의미론적 유사도 점수 (Semantic Similarity):** 답변과 정답을 임베딩 벡터로 변환 후 코사인 유사도 측정

최종 점수 = α × Factual + (1-α) × Semantic (기본 α = 0.75)

**`ground_truth` 품질 의존성:** `ground_truth`가 단답형이면 임베딩 유사도가 낮게 산정되어 점수가 과소 평가된다. 또한 `ground_truth`에 없는 내용을 정확히 답변해도 FP로 처리되어 감점된다. 따라서 `ground_truth` 작성 품질이 이 메트릭 신뢰도를 직접 결정한다.

**Answer Correctness만으로 부족한 이유:** 이 메트릭은 "결과"만 본다. Answer Correctness가 낮을 때 검색이 문제인지(관련 청크를 못 가져옴), 생성이 문제인지(청크는 맞는데 LLM이 잘못 해석) 구분할 수 없다. Context Recall/Precision과 Faithfulness를 함께 봐야 원인을 쪼개서 진단할 수 있다.

(참고: [Ragas — Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/), [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217))

#### 3-4. 메트릭 간 관계

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---------|---------------|------|------|
| 정답 청크 자체를 검색이 놓침 | **Context Recall ↓**, Answer Correctness ↓ | k 값 부족, 임베딩 공간에서 거리 멀음 | k 증가, 하이브리드 검색, 임베딩 모델 교체 |
| 정답 청크는 있지만 8~10위로 밀림 | **Context Precision ↓** | 노이즈 청크 상위 점령, Re-ranker 부재 | Re-ranking 적용, 메타데이터 필터 추가 |
| 검색은 맞는데 LLM이 외부 정보 추가 | **Faithfulness ↓** | LLM이 사전 지식으로 컨텍스트를 보완 | 프롬프트에 "컨텍스트 외 정보 금지" 강화 |
| LLM이 질문을 잘못 이해 | **Answer Relevancy ↓** | 답변이 엉뚱한 방향으로 전개됨 | 프롬프트 개선, 질문 재구성 지시 추가 |
| 답은 맞는데 장황 | (해당 없음 — 왜?) | Ragas 기본 메트릭은 간결성/장황성을 측정하지 않음. Answer Correctness는 FP(불필요한 정보)를 일부 감점하지만 소폭이고, Faithfulness는 맞는 말이면 감점 없음. 장황성은 도메인 임계값 또는 커스텀 메트릭으로 보완해야 함 | 커스텀 메트릭 또는 프롬프트에 "간결하게 답하라" 추가 |

---

## 8. 가설 vs 실제 결과 비교

### 가설 1: 4주차 정답률(사람)과 Ragas Ans Correctness(자동)의 일치 정도?

| 항목 | 내용 |
|------|------|
| **실습 전 가설** | 전체 트렌드는 일치하겠지만 Ragas가 사람보다 더 보수적(낮게)으로 채점할 것이다. 특히 교차 비교 문항에서 사람은 핵심 수치만 맞으면 정답 처리했지만 Ragas는 사실 누락까지 감점할 것이다. |
| **실제 결과** | **가설 적중.** q17(2종 외래 15%)처럼 단답형 명확한 정답 문항은 Ragas 0.982로 사람과 일치했다. q20(교차 비교)은 사람이 부분 정답(△)으로 보았고 Ragas도 0.511로 같은 판단을 내렸다. 반면 q09(이상지질혈증) Advanced 케이스는 사람이 정답으로 볼 수 있는 상황임에도 Ragas 0.038로 극단적 불일치가 발생했는데, 이는 LLM 답변 거부라는 특수 케이스에 기인한다. |

### 가설 2: Basic/Advanced의 네 메트릭 중 가장 크게 벌어질 메트릭은?

| 항목 | 내용 |
|------|------|
| **실습 전 가설** | Re-ranking 효과로 Context Precision이 가장 크게 상승할 것이다. |
| **실제 결과** | **가설 적중.** Context Precision이 0.786 → 0.929로 +0.144 상승하여 5개 메트릭 중 가장 큰 변화를 보였다. q07(조산아 지원기간)에서 Precision이 0.20 → 1.00으로, q09에서 0.25 → 1.00으로 수직 상승하여 리랭킹의 효과를 데이터로 증명했다. |

### 가설 3: 년도 혼동 문제는 어느 Ragas 메트릭에 주로 반영될 것인가?

| 항목 | 내용 |
|------|------|
| **실습 전 가설** | Faithfulness(환각 체크)에 가장 강하게 반영될 것이다. 컨텍스트에 없는 날짜를 LLM이 추론으로 생성하면 Faithfulness가 0이 될 것이다. |
| **실제 결과** | **가설 적중.** q20(노숙인 유효기간 Basic: Faithfulness 0.0), q09(이상지질혈증 Advanced: Faithfulness 0.0), q13(장기지속형 인하폭 Advanced: Faithfulness 0.0) 모두 Faithfulness가 0이 되었다. 연도 혼동이 발생한 모든 케이스에서 Faithfulness가 가장 민감하게 반응했다. |

### 가설 4: Advanced에서 Faithfulness가 오히려 낮아질 시나리오가 있을까?

| 항목 | 내용 |
|------|------|
| **실습 전 가설** | 있을 수 있다. 리랭킹이 청크를 재정렬하는 과정에서 관련성은 높지만 텍스트 노이즈(다른 연도 헤더 등)가 포함된 청크를 1위로 올리면, LLM이 오히려 혼동해 환각을 일으킬 수 있다. |
| **실제 결과** | **가설 적중.** q09(이상지질혈증)가 정확히 이 시나리오였다. Advanced는 리랭킹으로 2026년 문서를 1위로 올리는 데 성공했으나, 그 청크 본문에 "2025년 변경된 의료급여제도"라는 소제목이 포함되어 있어 LLM이 답변을 거부했다. Basic에서는 Faithfulness 1.0이었던 문항이 Advanced에서 0.0으로 역전된 케이스다. q13에서도 Advanced Faithfulness가 Basic(1.0)보다 낮은 0.0을 기록했다. |

---

## 9. 심화 A: Custom Metric — YearAccuracy

### 9-1. 설계 배경

Ragas 기본 메트릭은 답변의 의미적 정확도를 측정하지만, "질문이 요구한 연도(2025 또는 2026)가 답변에 정확히 반영됐는가"를 직접 측정하지는 않는다. 다년도 의료급여 문서 도메인에서 연도 혼동은 가장 치명적인 실패 유형이므로 규칙 기반 커스텀 메트릭을 구현했다.

### 9-2. 채점 로직

```python
@dataclass
class YearAccuracy(SingleTurnMetric):
    name: str = "year_accuracy"
    
    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        q = sample.user_input
        ans = sample.response
        
        q_years = set(re.findall(r'(2025|2026)', q))
        ans_years = set(re.findall(r'(2025|2026)', ans))
        
        if not q_years:              # 질문에 연도 없음 → 평가 면제
            return 1.0
        if "정보를 찾을 수 없습니다" in ans:  # 답변 거부 → 0점
            return 0.0
        if q_years.issubset(ans_years):
            if not ans_years.issubset(q_years):  # 질문에 없는 연도 혼입
                return 0.5           # 연도 혼용
            return 1.0               # 정상
        return 0.0                   # 필수 연도 누락
```

### 9-3. 측정 결과

```
==================================================
[심화 A] Year Accuracy (연도 정확도) 측정 결과
==================================================
- Basic RAG Year Accuracy    : 0.875
- Advanced RAG Year Accuracy : 0.900
==================================================
분석: Advanced RAG의 메타데이터 필터링이 연도 혼동 방지에 기여했음이 커스텀 메트릭으로 증명되었습니다.
```

### 9-4. 인사이트: 왜 Advanced도 만점(1.0)이 아닌가

메타데이터 필터링으로 검색 단계는 완벽했지만, **생성(Generation) 단계에서 두 가지 원인**으로 감점이 발생했다.

**원인 1 — Sticky Header 노이즈 (q09):** 2026년 문서의 청크 내 "2025년 변경된 의료급여제도" 헤더 텍스트로 인해 LLM이 답변을 거부("정보를 찾을 수 없습니다") → YearAccuracy 0.0 처리

**원인 2 — Cross-year 추론 실패 (q13, q14):** "2025년과 2026년의 차이"를 묻는 문항에서 LLM이 두 연도를 동시에 처리하지 못하고 한 연도를 누락

**핵심 교훈:** 메타데이터 필터링은 검색(Retrieval)의 연도 정확도를 100%에 가깝게 보장할 수 있지만, 최종 답변(Generation)의 연도 정확도를 보장하지는 않는다. 진정한 연도 혼동 방지는 "얼마나 잘 가져오는가"보다 "DB에 들어있는 데이터가 얼마나 깨끗한가(Parsing & Chunking 품질)"에서 승부가 결정된다.

---

## 참고 자료

| 분류 | 자료 |
|------|------|
| Ragas | [공식 문서](https://docs.ragas.io/), [RAGAS Paper (Es et al. 2023)](https://arxiv.org/abs/2309.15217) |
| LLM-as-a-Judge | [MT-Bench (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685), [G-Eval (Liu et al. 2023)](https://arxiv.org/abs/2303.16634) |
| Ragas 커스텀 메트릭 | [Write your own Metric](https://docs.ragas.io/en/stable/howtos/customizations/metrics/write_your_own_metric/) |
| Ragas 메트릭 상세 | [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) · [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) · [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) · [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) · [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/) |
| 데이터 소스 | 건강보험심사평가원 「2025·2026 알기 쉬운 의료급여제도」 |