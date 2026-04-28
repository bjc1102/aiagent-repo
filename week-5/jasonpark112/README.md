# 5주차 과제: RAG 시스템 정량 평가 — Ragas

---

## 이론 정리

### 1. Golden Dataset

**정의**: RAG 시스템의 성능을 객관적으로 검증하기 위한 고정 기준 데이터셋. 질문·정답·근거 청크를 사전에 정의해두고, 시스템 변경 시마다 동일 기준으로 회귀 검증하는 용도로 사용한다.

**없으면 생기는 문제**: "Advanced가 좋아 보였다"처럼 직관에 의존하게 되고, 프롬프트·모델·청크 크기를 바꿨을 때 성능 변화를 근거 있게 판단할 수 없다. 또한 어느 단계(검색/생성)에서 실패했는지도 알 수 없다.

**Ragas 필수 스키마 (v0.2+ 기준)**

| 필드 | 설명 | 준비 주체 |
|------|------|---------|
| `user_input` | 평가할 질문 | 사람이 작성 |
| `reference` | 완전한 문장 형태의 기대 답변 (ground_truth) | 사람이 작성 |
| `reference_contexts` | 정답 도출에 필요한 근거 청크 리스트 | 사람이 수동 어노테이션 |
| `response` | RAG 시스템이 생성한 답변 | 시스템이 자동 생성 |
| `retrieved_contexts` | RAG가 검색해 온 청크 리스트 | 시스템이 자동 생성 |

> v0.1에서는 `question` / `answer` / `ground_truth` / `contexts` 필드명을 사용했으나, v0.2+에서는 위 명칭으로 변경됐다.

**`reference_contexts`를 수동 어노테이션하는 이유**: Context Recall · Context Precision 계산 시 "정답 근거 청크"를 기준으로 삼기 때문이다. 자동 생성하면 실제 정답과 무관한 청크가 포함될 수 있어 평가 신뢰도가 떨어진다.

**권장 규모**

| 단계 | 개수 | 근거 |
|------|------|------|
| 초기 | 50~100개 | 핵심 질문 유형 커버 + 빠른 피드백 사이클 |
| 성숙 | 300~500개 | 엣지케이스·난이도 다양화, 회귀 이력 누적 |
| 대규모 | 1,000개+ | A/B 테스트·모델 비교 수준 |

양보다 질: 실제 사용자 질문, 함정 케이스(비슷한 수치가 여러 년도에 걸쳐 있는 경우), 회귀 이력(과거에 틀린 문항)을 포함해야 평가 신뢰도가 높다.

---

### 2. 평가의 필요성과 LLM-as-a-Judge

#### 2-1. 왜 체계적 평가가 필요한가

RAG 시스템은 확률적 출력을 생성하므로 전통 소프트웨어처럼 단위 테스트로 검증하기 어렵다.

| 문제 | 설명 |
|------|------|
| 회귀 탐지 불가 | 설정을 바꿨을 때 성능 변화를 근거 있게 판단 불가 |
| 디버깅 지점 모호 | 답변이 틀렸을 때 검색·생성·프롬프트 중 어디가 원인인지 불명 |
| 비교 기준 부재 | 여러 모델·파라미터 중 무엇이 나은지 직관으로 판단 |
| 프로덕션 도입 판단 불가 | "이 정도면 쓸 수 있다"는 합의 기준 없음 |

전통 소프트웨어는 입출력이 결정적(deterministic)이라 assert로 검증 가능하지만, LLM은 동일 입력에도 출력이 달라지므로 Golden Dataset 기반 통계적 평가가 필요하다. Golden Dataset은 소프트웨어의 회귀 테스트(regression test)와 같은 역할을 한다.

**자동 평가 vs 사람 평가 trade-off**

| 구분 | 비용 | 시간 | 신뢰성 | 확장성 |
|------|------|------|--------|--------|
| 사람 평가 | 높음 | 느림 | 높음 | 낮음 |
| 자동(규칙 기반) | 낮음 | 빠름 | 낮음 (의미 놓침) | 높음 |
| LLM-as-a-Judge | 중간 | 빠름 | 중간 (모델 의존) | 높음 |

#### 2-2. LLM-as-a-Judge

**등장 배경**: BLEU·ROUGE 같은 n-gram 기반 자동 평가는 단어 매칭만 보므로 의미론적 유사성을 포착하지 못한다. 예를 들어 "1,000원"과 "천 원"은 같은 의미지만 BLEU 점수는 0이다. 사람 평가는 신뢰도는 높지만 비용·시간이 크다. 이 간격을 메우기 위해 LLM을 판정자(judge)로 활용하는 방식이 등장했다.

**작동 원리**: 루브릭(채점 기준)과 판정 기준을 프롬프트로 제공하면 LLM이 답변을 읽고 점수·근거를 출력한다.

```
[프롬프트]
다음 기준으로 답변을 평가하세요:
- 정답과 의미적으로 일치하면 1점, 아니면 0점

[답변]: "본인부담금은 1,000원입니다"
[정답]: "2025년 1종 수급권자 의원 외래 본인부담금은 1,000원입니다"

→ LLM 출력: 1점 (핵심 수치 일치)
```

**좋은 루브릭 vs 나쁜 루브릭**

| | 예시 |
|--|------|
| 나쁨 | "답변이 좋으면 높은 점수를 주세요" |
| 좋음 | "정답의 수치(금액/비율)가 답변에 포함되어 있으면 1, 아니면 0" |

기준이 모호하면 LLM마다 점수가 달라져 신뢰도가 낮아진다.

**한계**
- 프롬프트·모델에 따라 점수가 달라지는 비결정성
- 생성용 LLM과 같은 모델로 평가하면 자기 편향(self-serving bias) 발생 → 다른 모델 패밀리 사용 권장
- API 호출 비용 발생, 한국어 등 비영어권 텍스트에서 파싱 실패 가능

**Ragas와의 관계**

| 메트릭 | LLM Judge 기반 여부 |
|--------|-------------------|
| Context Recall | LLM 기반 (reference 문장이 retrieved에 있는지 판단) |
| Context Precision | LLM 기반 (각 청크 관련성 판단) |
| Faithfulness | LLM 기반 (claim 추출 + 지지 여부 판단) |
| Answer Relevancy | LLM 기반 (역질문 생성) + 임베딩 유사도 |
| Answer Correctness | LLM 기반 (claim 추출 + TP/FP/FN 분류) |

> 참고: [Judging LLM-as-a-Judge — Zheng et al. 2023](https://arxiv.org/abs/2306.05685), [G-Eval — Liu et al. 2023](https://arxiv.org/abs/2303.16634)

---

### 3. Ragas 4대 메트릭 (+ Answer Correctness)

#### 3-1. 검색 단계

| 구분 | Context Recall | Context Precision |
|------|---------------|------------------|
| **정의** | 정답에 필요한 근거가 검색 결과에 빠짐없이 들어왔는가 (재현율) | 검색해 온 청크가 얼마나 실제 정답과 관련 있는가 (정밀도) |
| **계산 방식** | reference_contexts 각 문장이 retrieved_contexts에 있는지 LLM 판단 → `TP / (TP + FN)` | retrieved_contexts 각 청크가 reference와 관련 있는지 LLM 판단 → 평균 관련도 |
| **낮을 때 의심할 점** | 청크 크기 부족, 임베딩 모델 품질, 검색 k 값 부족 | 무관한 청크를 과도하게 검색, Re-ranker 미적용 또는 역효과 |
| **개선 기법** | 청크 크기 확대, Hybrid Search, k 값 증가 | Re-ranking 적용, 메타데이터 필터링, 청크 품질 개선 |

#### 3-2. 생성 단계

| 구분 | Faithfulness | Answer Relevancy |
|------|-------------|------------------|
| **정의** | 답변이 검색 결과에 근거하는가 (환각 체크) | 답변이 질문에 적절하게 답하고 있는가 |
| **계산 방식** | response의 각 claim이 retrieved_contexts에 의해 지지되는지 판단 → `지지된 claim / 전체 claim` | response로부터 역질문 N개 생성 → 원래 user_input과 임베딩 유사도 평균 |
| **낮을 때 의심할 점** | LLM이 컨텍스트 외 지식을 사용, 청크 분절로 맥락 부족 → 추론으로 보완 | "정보를 찾을 수 없습니다" 같은 거절 응답, 질문을 잘못 이해한 답변 |
| **개선 기법** | 컨텍스트 외 정보 사용 금지 프롬프트 강화, 청크 크기 확대 | 답변 형식 유도 프롬프트, 검색 품질 개선 |

#### 3-3. End-to-End: Answer Correctness

**정의**: 답변이 실제 정답(reference)과 의미적으로 일치하는가를 claim 단위 F1으로 측정.

**계산 방식**:
1. LLM이 response와 reference 각각에서 factual claims 추출
2. 각 claim을 TP(둘 다 있음) / FP(response에만 있음) / FN(reference에만 있음)으로 분류
3. `F1 = 2*TP / (2*TP + FP + FN)`

**ground_truth 품질 의존성**: reference가 짧거나 불완전하면 FN이 과소평가되어 점수가 부풀려진다. "1,000원"보다 "2025년 1종 수급권자 의원 외래 본인부담금은 1,000원입니다"처럼 완전한 문장으로 작성해야 한다.

**Answer Correctness만으로 부족한 이유**: 검색이 실패했는지 생성이 실패했는지 구분하지 못한다. 점수가 낮아도 "청크를 못 찾아서"인지 "찾았는데 LLM이 잘못 답한 것"인지 알 수 없다. Context Recall과 함께 봐야 진단이 가능하다.

#### 3-4. 메트릭 간 관계

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---------|---------------|------|------|
| 정답 청크 자체를 검색이 놓침 | Context Recall | 청크 분절, 임베딩 미매칭 | 청크 크기 확대, Hybrid Search |
| 정답 청크는 있지만 비관련 청크가 함께 검색됨 | Context Precision | 검색 범위 과도, Re-ranker 미적용 | Re-ranking, k 값 조정 |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness | 프롬프트 미흡, 컨텍스트 부족 | "컨텍스트 외 사용 금지" 프롬프트 강화 |
| LLM이 질문을 잘못 이해 | Answer Relevancy | 역질문이 원래 질문과 유사도 낮음 | 프롬프트 개선, 답변 형식 유도 |
| 답은 맞는데 장황 | 해당 없음 | Ragas는 주관적 품질(간결성) 미측정 | 커스텀 메트릭(`MetricWithLLM` 상속) 또는 도메인 임계값 설정 |

> 참고: [Ragas Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217), [Ragas Metrics 공식 문서](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

---

### 4. 실습 전 가설 vs 실습 후 결과 비교

| 가설 | 예측 | 실제 결과 | 비고 |
|------|------|---------|------|
| 4주차 정답률 vs Ragas Ans Correctness 일치도 | 대체로 일치하되, 수치 형식이 비슷한 오답은 Ragas가 부분 점수 부여할 것 | 약 79~80% 일치. q07·q17처럼 틀린 답변에 0.6 이상 부여하는 케이스 발생 | Ragas가 수동보다 항상 엄격하지 않음 |
| Basic/Advanced 중 가장 크게 벌어질 메트릭 | Context Precision — Re-ranking이 불필요한 청크를 제거하므로 | Context Precision이 +0.213으로 가장 크게 개선 (예측 일치) | |
| 년도 혼동은 어느 메트릭에 반영? | Context Recall — 잘못된 년도 청크를 검색하면 Recall이 낮을 것 | Context Recall은 year 구분 없이 ground_truth_contexts 커버 여부만 측정하므로 year confusion 미감지. Answer Correctness에서 드러남 (예측 불일치) | Year Accuracy 커스텀 메트릭 필요 |
| Advanced에서 Faithfulness가 낮아지는 시나리오 | Re-ranking으로 정답 청크가 탈락 → LLM이 부족한 컨텍스트를 추론으로 보완 → 환각 | Faithfulness 0.408 → 0.372로 낮아짐. q03처럼 자연분만 청크가 탈락하고 비관련 청크로 대체된 케이스에서 확인 (예측 일치) | |

---

## 실습

---

## 1. 실행 환경 및 사용 모델

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.14 |
| 프레임워크 | LangChain, Ragas 0.4.3 |
| 생성용 LLM | gpt-4o-mini (OpenAI) |
| 평가용 LLM | claude-sonnet-4-5 (Anthropic) |
| 임베딩 모델 | text-embedding-3-small (OpenAI) |
| Re-ranker | BAAI/bge-reranker-v2-m3 (로컬, sentence-transformers) |
| 벡터 저장소 | FAISS (4주차 인덱스 재사용) |
| Hybrid Search | FAISS (Dense Vector) + BM25Retriever (Sparse) |

> 평가용 LLM으로 생성용(gpt-4o-mini)과 다른 모델 패밀리(Anthropic)를 선택한 이유: 동일 모델이 자신의 출력을 평가하면 편향이 발생할 수 있어 교차 평가 구조를 채택했다.

> Ragas 0.4.x에서 `adapt_prompts`(한국어 프롬프트 전환)가 미지원되어 스킵. Claude Sonnet이 한국어 텍스트를 직접 평가하므로 평가 품질에 큰 영향 없음.

---

## 2. Golden Dataset 확장 전략 (golden_dataset_v2.jsonl)

4주차 20문항에 두 필드를 추가했다.

| 필드 | 의미 | 작성 원칙 |
|------|------|---------|
| `ground_truth` | 완전한 문장 형태의 기대 답변 | "년도 + 대상 + 조건 + 값" 한 문장으로 정제 |
| `ground_truth_contexts` | 정답 근거 청크 리스트 | PDF에서 관련 문단 의미 단위로 발췌 |

### ground_truth 정제 원칙

| 형태 | 예시 |
|------|------|
| 나쁨 | `"1,000원"` |
| 좋음 | `"2025년 의료급여 1종 수급권자가 의원에서 외래 진료를 받을 때 본인부담금은 1,000원입니다."` |

### ground_truth_contexts 발췌 원칙

- 정답 도출에 필요한 근거 문단을 모두 포함 (1~2개)
- cross-year 문항은 2025년/2026년 두 년도 청크 포함
- 계산형 문항(q07)은 계산 근거가 되는 두 규정 모두 포함

### 난이도별 분포

| 난이도 | 문항 수 | 특징 |
|--------|---------|------|
| easy | 6문항 | 단순 수치 조회 |
| medium | 6문항 | 조건부 본인부담률 |
| hard | 4문항 | 계산 또는 복합 조건 |
| cross-year | 4문항 | 2025↔2026 연도 변경 항목 |

---

## 3. Ragas 평가 파이프라인 (Step 1)

### 파이프라인 구성

```
golden_dataset_v2.jsonl (20문항)
        ⬇️
질문별 Basic RAG / Advanced RAG 실행
        ⬇️
SingleTurnSample 생성 (user_input / response / retrieved_contexts / reference / reference_contexts)
        ⬇️
EvaluationDataset (20샘플)
        ⬇️
Ragas evaluate() — 평가용 LLM: claude-sonnet-4-5
        ⬇️
basic_ragas_scores.csv / advanced_ragas_scores.csv
```

### Ragas 필드 매핑

| golden_dataset_v2.jsonl | SingleTurnSample |
|------------------------|-----------------|
| `question` | `user_input` |
| RAG 실행 결과 (답변) | `response` |
| RAG 실행 결과 (검색 청크) | `retrieved_contexts` |
| `ground_truth` | `reference` |
| `ground_truth_contexts` | `reference_contexts` |

### 평가 메트릭 5개

| 메트릭 | 측정 내용 | 
|--------|---------|
| ContextRecall | retrieved_contexts가 reference_contexts를 얼마나 커버하나 | 
| ContextPrecision | retrieved_contexts의 각 청크가 reference와 관련 있는지 판단 | 
| Faithfulness | 답변(response)이 retrieved_contexts에 충실한가 (환각 체크) | 
| AnswerRelevancy | 답변(response)을 보고 역으로 질문을 여러 개 생성한 뒤, 원래 user_input과의 임베딩 유사도 측정| 
| AnswerCorrectness | 답변(response)이 reference(ground_truth)와 일치하는가 | 

### RAG 구성

**Basic RAG**
```
질문 -> FAISS 벡터 검색 (Top-3) -> parent_content 컨텍스트 -> gpt-4o-mini -> 답변
```

**Advanced RAG (Hybrid + Re-ranking, 년도 필터링 없음)**
```
질문 -> FAISS (Top-5) + BM25 (Top-5) -> EnsembleRetriever (RRF 0.5:0.5)
     -> CrossEncoder Re-ranking (BAAI/bge-reranker-v2-m3, Top-3)
     -> parent_content 컨텍스트 -> gpt-4o-mini -> 답변
```

| 항목 | 설정값 |
|------|--------|
| child chunk | 300자, overlap 50자 |
| parent chunk | 1500자, overlap 100자 |
| FAISS k | 5 |
| BM25 k | 5 |
| Rerank Top-N | 3 |

---

## 4. Step 2-1: Ragas 평가 결과 (전체 평균)

| 메트릭 | Basic | Advanced | 변화 |
|--------|-------|---------|------|
| Context Recall | 0.474 | **0.556** |  +0.082 |
| Context Precision | 0.319 | **0.532** |  +0.213 |
| Faithfulness | **0.408** | 0.372 |  -0.036 |
| Answer Relevancy | **0.406** | 0.248 |  -0.158 |
| Answer Correctness | **0.423** | 0.375 |  -0.048 |

---

## 5. Step 2-2: 문항별 결과

| ID | 난이도 | source_year | Ctx Recall B/A | Ctx Precision B/A | Faithfulness B/A | Ans Relevancy B/A | Ans Correctness B/A |
|----|--------|------------|---------------|------------------|-----------------|------------------|---------------------|
| q01 | easy | 2025 | 0.000 / 0.000 | NaN / NaN | 0.500 / 0.250 | 0.447 / NaN | NaN / NaN |
| q02 | easy | 2025 | 1.000 / 1.000 | 1.000 / 1.000 | 0.200 / NaN | 0.981 / 0.981 | 0.649 / 0.649 |
| q03 | easy | 2025 | 1.000 / 1.000 | 0.500 / 1.000 | 0.500 / 0.000 | 0.337 / NaN | 0.692 / 0.041 |
| q04 | medium | 2025 | 1.000 / 1.000 | NaN / NaN | 0.667 / 0.833 | 0.563 / NaN | NaN / 0.583 |
| q05 | medium | 2025 | 0.000 / 0.000 | NaN / NaN | 0.000 / 0.000 | NaN / 0.000 | 0.045 / 0.045 |
| q06 | medium | 2025 | NaN / 1.000 | 0.000 / NaN | 0.333 / 0.000 | NaN / NaN | NaN / NaN |
| q07 | hard | 2025 | 0.000 / 0.000 | NaN / 0.000 | 0.100 / 0.100 | NaN / NaN | 0.658 / 0.741 |
| q08 | hard | 2025 | 0.000 / NaN | 0.000 / 0.000 | 1.000 / 0.600 | NaN / 0.507 | 0.208 / 0.204 |
| q09 | cross-year | 2025+2026 | 0.000 / 0.000 | 0.000 / NaN | 0.750 / 1.000 | NaN / NaN | 0.190 / 0.056 |
| q10 | cross-year | 2025+2026 | 1.000 / 1.000 | NaN / 1.000 | 0.833 / 0.333 | NaN / NaN | 0.601 / 0.836 |
| q11 | easy | 2026 | 1.000 / 0.000 | NaN / 0.333 | 0.400 / 0.500 | NaN / NaN | 0.431 / NaN |
| q12 | easy | 2026 | 1.000 / 1.000 | 1.000 / NaN | 0.600 / 0.667 | NaN / NaN | 0.773 / 0.715 |
| q13 | easy | 2026 | 0.000 / NaN | 0.000 / 0.000 | 0.000 / 0.625 | NaN / NaN | NaN / NaN |
| q14 | medium | 2026 | 1.000 / 1.000 | NaN / 0.583 | 0.556 / 0.714 | 0.565 / NaN | NaN / 0.842 |
| q15 | medium | 2026 | 0.000 / 0.000 | NaN / 0.000 | 0.000 / NaN | NaN / NaN | 0.043 / 0.043 |
| q16 | medium | 2026 | 0.000 / 1.000 | 0.000 / NaN | 0.500 / 0.000 | NaN / NaN | NaN / 0.209 |
| q17 | hard | 2026 | 1.000 / 1.000 | 1.000 / 1.000 | 0.429 / 0.333 | NaN / NaN | 0.527 / NaN |
| q18 | hard | 2026 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.054 / 0.054 |
| q19 | cross-year | 2025+2026 | 0.000 / 0.000 | 0.000 / 1.000 | 0.000 / 0.750 | 0.000 / 0.000 | 0.155 / 0.185 |
| q20 | cross-year | 2025+2026 | 1.000 / 1.000 | 0.333 / 1.000 | 0.800 / 0.000 | 0.357 / 0.000 | 0.901 / 0.425 |

> B=Basic RAG / A=Advanced RAG

### NaN 발생 조건 (메트릭별)

| 메트릭 | 계산법 | NaN 발생 조건 |
|--------|--------|--------------|
| **Context Recall** | `TP / (TP + FN)` — reference_contexts 문장 중 retrieved에서 찾은 비율 | reference_contexts 문장이 없거나 평가 LLM 호출 실패 |
| **Context Precision** | retrieved 청크 각각이 reference와 관련 있는지 평균 | retrieved_contexts가 비었거나 관련 청크가 하나도 없을 때 |
| **Faithfulness** | `지지된 claim / 전체 claim` — response claim이 retrieved에 근거 있는 비율 | response에서 claim 추출 실패 (예: "정보를 찾을 수 없습니다") |
| **Answer Relevancy** | response로부터 역질문 생성 후 원래 질문과 임베딩 유사도 평균 | response가 너무 짧거나 역질문 생성 실패 |
| **Answer Correctness** | `2*TP / (2*TP + FP + FN)` — response/reference 각각 claim 추출 후 F1 | claim 추출 자체 실패 (API 에러, JSON 파싱 실패) |

> TP = 맞게 말한 것, FP = 틀리게 말한 것 (없는 말을 지어냄), FN = 말했어야 하는데 안 한 것
>
> Answer Correctness NaN은 청크 분절이 직접 원인이 아니라 평가 LLM(Claude Sonnet) 호출 실패에 가깝다. 틀린 답변이 생성된 경우 FP ≥ 1이라 분모가 0이 될 수 없기 때문이다.

### 주요 관찰

- **NaN 다수 발생**: 메트릭마다 원인이 다름
  - **Context Recall / Precision**: 청크 분절로 의미 있는 청크가 검색되지 않아 비교 대상 자체가 없음 → 분모 0 → NaN
  - **Faithfulness**: 검색 실패로 LLM이 "정보를 찾을 수 없습니다" 응답 -> claim 추출 불가 -> NaN
  - **Answer Correctness**: 청크 분절이 직접 원인이 아니라 평가 LLM(Claude Sonnet) API 호출 실패 또는 JSON 파싱 실패 -> NaN (틀린 답변이 생성된 경우 FP≥1이므로 분모가 0이 될 수 없어 수학적 0/0이 아님)
- **Answer Relevancy**: Basic(0.406)이 Advanced(0.248)보다 크게 높음 — Advanced 답변이 상대적으로 짧거나 불완전하게 생성되는 경향. 단, 20문항 중 NaN이 다수(Basic ~13개, Advanced ~14개)라 유효값이 소수에 불과해 평균 신뢰도가 낮음
- **q10**: Answer Correctness 기준 Advanced가 Basic 대비 크게 개선 (0.601 -> 0.836)
- **q03, q20**: Answer Correctness 기준 Basic이 Advanced보다 크게 높음 (q03: 0.692 -> 0.041, q20: 0.901 -> 0.425) 
- **근본 원인**: child=300, parent=1500 청크 크기에서 HTML 표 구조가 분절되어 정보 손실 발생. 4주차에서 c1000_p3000으로 개선 시 70% 달성했던 것과 동일한 문제

---

---

## 6. Step 2-3: 4주차 수동 채점 vs Ragas Answer Correctness 비교

### Basic RAG

| ID | 4주차 수동 판정 | Ragas Ans Correctness | 일치 여부 | 불일치 원인 |
|----|--------------|----------------------|---------|-----------|
| q01 | 오답 | NaN | — | 메트릭 계산 실패 |
| q02 | 정답 | 0.649 | 일치 | |
| q03 | 정답 | 0.692 | 일치 | |
| q04 | 정답 | NaN | — | 메트릭 계산 실패 |
| q05 | 오답 | 0.045 |  일치 | |
| q06 | 오답 | NaN | — | 메트릭 계산 실패 |
| q07 | 오답 | 0.658 | 불일치 | 틀린 답변이지만 Ragas가 유사 수치 형식으로 부분 점수 부여 |
| q08 | 오답 | 0.208 |  일치 | |
| q09 | 오답 | 0.190 |  일치 | |
| q10 | 정답 | 0.601 |  일치 | |
| q11 | 정답 | 0.431 |  불일치 | 4주차 답변 "1,500~2,000원" → ground_truth "1,500원"과 표현 차이로 저평가 |
| q12 | 정답 | 0.773 |  일치 | |
| q13 | 오답 | NaN | — | 메트릭 계산 실패 |
| q14 | 정답 | NaN | — | 메트릭 계산 실패 |
| q15 | 오답 | 0.043 |  일치 | |
| q16 | 오답 | NaN | — | 메트릭 계산 실패 |
| q17 | 오답 | 0.527 |  불일치 | 틀린 답변이지만 Ragas가 유사 문장 구조로 부분 점수 부여 |
| q18 | 오답 | 0.054 |  일치 | |
| q19 | 오답 | 0.155 |  일치 | |
| q20 | 정답 | 0.901 |  일치 | |

### Advanced RAG

| ID | 4주차 수동 판정 | Ragas Ans Correctness | 일치 여부 | 불일치 원인 |
|----|--------------|----------------------|---------|-----------|
| q01 | 오답 | NaN | — | 메트릭 계산 실패 |
| q02 | 정답 | 0.649 |  일치 | |
| q03 | 오답 | 0.041 |  일치 | |
| q04 | 정답 | 0.583 |  일치 | |
| q05 | 오답 | 0.045 |  일치 | |
| q06 | 오답 | NaN | — | 메트릭 계산 실패 |
| q07 | 오답 | 0.741 |  불일치 | 틀린 답변이지만 Ragas가 유사 수치 형식으로 높은 점수 부여 |
| q08 | 오답 | 0.204 |  일치 | |
| q09 | 오답 | 0.056 |  일치 | |
| q10 | 정답 | 0.836 |  일치 | |
| q11 | 정답 | NaN | — | 메트릭 계산 실패 |
| q12 | 정답 | 0.715 |  일치 | |
| q13 | 오답 | NaN | — | 메트릭 계산 실패 |
| q14 | 정답 | 0.842 |  일치 | |
| q15 | 오답 | 0.043 |  일치 | |
| q16 | 오답 | 0.209 |  일치 | |
| q17 | 정답 | NaN | — | 메트릭 계산 실패 |
| q18 | 정답 | 0.054 |  불일치 | 4주차 정답이었으나 ground_truth 문장과 달라 Ragas 저평가 |
| q19 | 정답 | 0.185 |  불일치 | 4주차 정답이었으나 ground_truth 문장과 달라 Ragas 저평가 |
| q20 | 오답 | 0.425 |  일치 | |

### 비교 분석 요약

| 항목 | Basic | Advanced |
|------|-------|---------|
| 4주차 수동 정답률 | 40% (8/20) | 45% (9/20) |
| Ragas Ans Correctness 평균 | 0.423 | 0.375 |
| 수동↔Ragas 일치율 | 약 79% | 약 80% |
| NaN 발생 (비교 불가) | 6문항 | 5문항 |
| 불일치 주요 원인 | 수치 형식 유사성으로 부분 점수 부여(q07, q17), 표현 차이(q11) | 수치 형식 유사성(q07), 표현 차이(q18, q19) |

**주요 인사이트:**
- 수동 채점과 Ragas는 대체로 일치하나, **틀린 답변인데도 형식 유사성으로 Ragas가 부분 점수를 부여**하는 케이스 발생 (q07, q17, q18, q19)
- Ragas가 수동 채점보다 **항상 엄격하지는 않음**: 수동 오답이 Ragas 0.6 이상으로 평가되는 케이스(q07)도 존재
- NaN이 많은 문항은 원인이 메트릭마다 다름: **Context Recall/Precision · Faithfulness NaN은 청크 분절로 인한 검색 실패**가 근본 원인이고, **Answer Correctness NaN은 평가 LLM(Claude Sonnet) API 호출 실패 또는 JSON 파싱 실패**가 원인 (틀린 답변이어도 FP≥1이므로 수학적 0/0이 아님)

---

---

## 7. Step 3: Basic vs Advanced RAG 비교 분석

### 3-1. 다차원 비교

| 구분 | 메트릭 | 변화 | 해석 |
|------|--------|------|------|
| 개선 | Context Recall | 0.474 → 0.556 | Hybrid Search로 관련 청크 커버리지 증가 |
| 개선 | Context Precision | 0.319 → 0.532 | Re-ranking으로 불필요한 청크 제거 |
| 악화 | Faithfulness | 0.408 → 0.372 | 청크 분절로 맥락 부족 → LLM이 추론으로 보완하며 환각 증가 |
| 악화 | Answer Correctness | 0.423 → 0.375 | Re-ranking이 일부 정답 청크를 탈락시키는 역효과 발생 |
| 악화 | Answer Relevancy | 0.406 → 0.248 | Advanced 답변이 Basic 대비 짧거나 불완전하여 낮은 점수 유지 |

### 3-2. 년도 혼동 재진단 (cross-year 4문항)

| ID | 질문 | Ctx Recall B/A | Ans Correctness B/A | 진단 |
|----|------|---------------|--------------------|----|
| q09 | 2025년 CT 500원 | 0.000 / 0.000 | 0.190 / 0.056 | 양쪽 모두 관련 청크 미검색, Basic이 소폭 높음 |
| q10 | 2025년 장기지속형주사제 5% | 1.000 / 1.000 | 0.601 / 0.836 | Advanced Re-ranking 후 오히려 개선 |
| q19 | 2026년 CT 5% | 0.000 / 0.000 | 0.155 / 0.185 | 양쪽 모두 낮음, Advanced가 소폭 높음 |
| q20 | 2026년 장기지속형주사제 2% | 1.000 / 1.000 | 0.901 / 0.425 | Basic은 성공, Advanced Re-ranking 후 악화 |

**년도 혼동과 메트릭의 관계:**
- Context Recall은 년도 혼동을 구분하지 못함 — cross-year 4문항에서 Basic/Advanced 모두 동일한 값 (q09·q19: 0.0/0.0, q10·q20: 1.0/1.0). 년도 혼동이 드러나는 건 **Answer Correctness** (q20: Basic 0.901 → Advanced 0.425)
- Ragas 기본 메트릭은 년도 혼동을 **직접** 포착하지 못함. "2025년 정보를 찾았는가"가 아니라 "ground_truth_contexts를 커버했는가"를 측정하기 때문
- Year Accuracy 커스텀 메트릭 없이는 년도 혼동 여부를 정량화하기 어려움

### 3-3. 인사이트

**"Advanced가 낫다" 결론의 재검토**
4주차에서 Advanced RAG(45%)가 Basic(40%)보다 나은 것으로 결론 내렸으나, Ragas로 정량 평가하면 검색(Recall/Precision)만 개선되고 Answer Correctness·Answer Relevancy는 오히려 낮다. 이는 수동 채점이 핵심값 포함 여부만 체크한 반면, Ragas는 의미론적 완결성까지 평가하기 때문이다. Re-ranking이 일부 문항에서 정답 청크를 탈락시키는 역효과도 확인됐다(q20). "Advanced가 낫다"는 결론은 검색 품질에 한해서만 유효하다.

**프로덕션 가능성**
현재 Faithfulness 평균 약 0.4 수준은 도메인 임계값(≥0.9)에 크게 못 미친다. 의료급여 도메인에서 환각이 40% 수준이라면 실제 수급권자에게 잘못된 본인부담금을 안내할 위험이 있어 프로덕션 적용은 불가하다. NaN 비율도 높아 평가 신뢰도 자체도 낮다.

**개선 우선순위**
1. **청크 크기 확대 (c1000, p3000)**: 4주차에서 이미 70% 달성. HTML 표 구조 보존이 핵심으로, 이 변경 하나가 모든 메트릭 개선으로 이어질 가능성이 높다.
2. **Faithfulness 개선**: 컨텍스트 외 정보 사용 금지 프롬프트 강화.
3. **Answer Relevancy 개선**: Advanced RAG에서 Basic 대비 Relevancy가 크게 낮아(0.248 vs 0.406)지는 원인 파악 및 답변 형식 유도 프롬프트 수정.

---

---

## 8. Step 4: 실패 케이스 Deep Dive

### Case A: q03 — Advanced가 Basic보다 악화

**질문:** 2025년 2종 수급권자인 임산부가 자연분만으로 입원할 때 본인부담률은?
**참고 정답:** 2025년 의료급여 2종 수급권자인 임산부가 자연분만으로 입원할 때 본인부담률은 무료(0%)입니다.

| 구분 | Ctx Recall | Ctx Precision | Faithfulness | Ans Correctness |
|------|-----------|--------------|-------------|----------------|
| Basic | 1.000 | 0.500 | 0.500 | **0.692** |
| Advanced | 1.000 | 1.000 | 0.000 | **0.041** |

**생성된 답변 (4주차 기록 기반)**
- Basic: "0% (무료)" → 정답
- Advanced: "정보를 찾을 수 없습니다" → 오답

**원인 분석**
- Advanced Re-ranking 과정에서 **자연분만 면제 청크가 탈락**하고 임플란트·2·3인실 입원료·2026 표지 등 비관련 청크 3개가 최종 retrieved_contexts로 전달됨 (CSV로 확인)
- CrossEncoder가 질문("자연분만 입원 본인부담률")에 대해 "분만 면제" 청크보다 다른 청크를 더 관련 있다고 잘못 판단한 것
- LLM이 비관련 청크만 받아 "정보를 찾을 수 없습니다" 응답 → Answer Correctness 0.041
- **Context Recall 1.0 · Precision 1.0은 평가 LLM(Claude Sonnet)의 오평가**: 실제 retrieved 청크에 자연분만 관련 내용이 없음에도 Ragas 평가 LLM이 관련 있다고 잘못 판단 — 평가 LLM 자체의 hallucination
- **가장 잘 드러낸 메트릭**: Answer Correctness (0.692 → 0.041, 낙폭 최대)
- **조치**: Re-ranker 입력 텍스트에 parent chunk(1500자) 대신 더 넓은 컨텍스트 사용, 또는 청크 크기 확대(c1000, p3000)로 분만 면제 규정 전체를 하나의 청크에 포함

---

### Case B: q20 — 년도 혼동 (Context 검색됐지만 답변 년도 틀림)

**질문:** 2026년 정신질환 외래 진료 시 항정신병 장기지속형 주사제의 본인부담률은?
**참고 정답:** 2026년 기준 2%입니다. (2025년 10월 1일부터 5%에서 2%로 인하 시행)

| 구분 | Ctx Recall | Ctx Precision | Faithfulness | Ans Relevancy | Ans Correctness |
|------|-----------|--------------|-------------|--------------|----------------|
| Basic | 1.000 | 0.333 | 0.800 | 0.357 | **0.901** |
| Advanced | 1.000 | 1.000 | 0.000 | 0.000 | **0.425** |

**생성된 답변 (4주차 기록 기반)**
- Basic: "2%" → 정답
- Advanced: "정보를 찾을 수 없습니다" → 오답

**원인 분석**
- Context Recall 둘 다 1.0 — 관련 청크는 검색됨
- Basic은 Faithfulness 0.800 + Ans Correctness 0.901으로 정답, Advanced는 Faithfulness 0.000 + Ans Correctness 0.425로 크게 악화
- Advanced Re-ranking 후 **2025년 5% 청크가 상위**로 올라오고 2026년 2% 청크가 탈락 → LLM이 2026년 정보를 못 받아 "정보 없음" 응답
- 년도 혼동의 전형적 패턴: 2025/2026 문서에 동일 주제가 반복 수록되어 Re-ranker가 년도 구분 없이 스코어링
- **가장 잘 드러낸 메트릭**: Context Precision (Basic 0.333 → Advanced 1.0이지만 Ans Correctness는 반대) — Precision이 높아도 정답 년도 청크가 아닐 수 있음
- **Ragas의 한계**: 어느 메트릭도 "2026년 정보를 검색했는가"를 직접 측정하지 못함. Year Accuracy 커스텀 메트릭이 필요한 이유
- **조치**: source_year 메타데이터 기반 년도 pre-filtering, 또는 청크에 년도 텍스트를 명시하여 Re-ranker가 년도 구분 가능하도록 설계

---

### Case C: q09 — Faithfulness 높은데 Answer Correctness 낮음 (메트릭 충돌)

**질문:** 2025년 1종 수급권자가 외래에서 CT 검사를 받을 때 본인부담금은?
**참고 정답:** 2025년 기준 500원입니다.

| 구분 | Ctx Recall | Faithfulness | Ans Correctness |
|------|-----------|-------------|----------------|
| Basic | 0.000 | **0.750** | **0.190** |
| Advanced | 0.000 | **1.000** | **0.056** |

**생성된 답변**
- Basic/Advanced 모두: "정보를 찾을 수 없습니다"

**원인 분석**
- Faithfulness 1.0 = 답변이 retrieved_contexts에 충실 (환각 없음)
- Answer Correctness 0.056 = 정답과 전혀 다름
- 이 두 메트릭이 충돌하는 이유: **LLM이 "정보를 찾을 수 없습니다"라고 정직하게 답한 것** — 검색 실패를 인정한 답변이라 환각은 0이지만 정답도 아님
- Ragas가 "정보 없음" 응답을 Faithfulness 높음으로 평가 → 검색 실패가 Faithfulness로 위장되는 맹점
- Context Recall 0.0이 실패의 진짜 원인 — 2025/2026 문서에서 CT 본인부담 수치를 담은 청크를 찾지 못함 (임베딩 포화 또는 청크 분절)
- **조치**: 청크 크기 확대로 CT/MRI/PET 표 전체를 하나의 청크에 포함

---

### 4-3. 공통 교훈

- **Ragas가 놓치는 실패 유형**: 년도 혼동을 직접 측정 못함 (Context Recall은 year 구분 없이 ground_truth_contexts 커버 여부만 측정). "정보 없음" 응답을 Faithfulness 높음으로 오평가. "정보 없음" 응답에 대해 역질문 생성이 성공하면 0점, 실패하면 NaN으로 Answer Relevancy가 불일치하게 동작.
- **NaN 원인은 메트릭마다 다름**: Context Recall/Precision·Faithfulness NaN은 청크 분절로 인한 검색 실패가 원인. Answer Correctness NaN은 평가 LLM API 호출 실패 또는 JSON 파싱 실패가 원인 — 청크 품질과 무관하게 발생.
- **Context Recall이 핵심 지표**: Recall 0.0이면 나머지 메트릭은 무의미. 검색 실패가 대부분 오답의 근본 원인이나, Answer Correctness NaN은 예외.
- **Re-ranking 역효과**: Advanced RAG에서 Re-ranking이 정답 청크를 탈락시키는 케이스(q03, q20) 발생. Re-ranker 입력 품질이 낮으면 오히려 악화.
- **수동 vs 자동 채점 엄격도**: 수동은 핵심값 포함 여부만 체크(관대), Ragas는 의미론적 완결성까지 평가(엄격). 일치 여부 판단 기준은 Answer Correctness 0.5 — 정답이면 0.5 이상, 오답이면 0.5 미만이어야 일치. q11처럼 수동 정답이 Ragas 저평가되는 케이스 존재.
- **근본 해결책**: 청크 크기 확대(c1000, p3000) — 알고리즘 개선보다 데이터 품질이 우선.
