# 5주차 과제: RAG 평가 — Golden Dataset, LLM-as-a-Judge, Ragas

## 실행 환경 및 구성

| 항목                         | 내용                                                                  |
| ---------------------------- | --------------------------------------------------------------------- |
| 생성용 LLM                   | `gemini-2.5-flash` (temperature=0) — week2/3/4 공통                   |
| **판정용 LLM (Ragas Judge)** | **OpenAI `gpt-4.1-mini`** (temperature=0) — 생성용과 다른 모델 family |
| 임베딩                       | Google `gemini-embedding-001`                                         |
| Ragas 버전                   | 0.4.3                                                                 |
| RAG 프레임워크               | LangChain (+ langchain-classic for EnsembleRetriever)                 |
| Re-ranker                    | Cohere `rerank-v3.5` (Advanced RAG만)                                 |
| 벡터 저장소                  | FAISS (dense vector, 26청크 — 2025/2026 PDF 각 13청크)                |
| BM25                         | in-memory, `BM25Retriever.from_documents`                             |
| 평가 범위                    | **전수 20문항**                                                       |
| 실행 환경                    | Python 3.13, macOS, `.venv`                                           |

---

## 이론 과제 답변

### 1. Golden Dataset

**한 줄 정의**: RAG 시스템의 성능을 객관적으로 측정하기 위한 "정답이 명확한" 질문-답변 쌍 데이터셋.

**왜 필요한가**:

- 없으면 프롬프트·모델·검색 설정을 바꿨을 때 개선 여부를 주관적으로 판단하게 됨 → 회귀(regression) 탐지 불가
- 실무에서 변경이 잦을수록 자동 회귀 테스트 필수 — Golden Dataset이 그 기준점
- LLM의 확률적 출력 특성상 단위 테스트만으로는 커버 불가 → 의미적 일치도를 체계적으로 확인하는 수단

**필수 스키마 (Ragas v0.2+ 기준)**:

| JSONL 필드              | Ragas `SingleTurnSample` 필드 | 준비 주체    | 준비 방법                          |
| ----------------------- | ----------------------------- | ------------ | ---------------------------------- |
| `question`              | `user_input`                  | 평가자       | 실제 유저 질문을 자연어로 작성     |
| `ground_truth`          | `reference`                   | 평가자(수동) | PDF 근거를 완전한 문장 형태로 정제 |
| `ground_truth_contexts` | `reference_contexts`          | 평가자(수동) | PDF 원본 문단 2~5문장 발췌 리스트  |
| (RAG 실행 산출물)       | `response`                    | RAG 시스템   | 파이프라인 `invoke()` 결과         |
| (RAG 실행 산출물)       | `retrieved_contexts`          | RAG 시스템   | Retriever가 반환한 청크 리스트     |

> v0.1 필드명(`question`/`answer`/`contexts`/`ground_truths`)은 v0.2+에서 deprecated. 블로그 튜토리얼 대부분이 v0.1 기준이라 최신 Ragas와 맞지 않음.

**권장 규모**:

| 단계         | 문항 수    | 근거                                                   |
| ------------ | ---------- | ------------------------------------------------------ |
| 초기 (pilot) | 5~20문항   | 스키마·비용·파이프라인 검증용. 한 사이클 빠르게 돌리기 |
| 성숙기       | 50~100문항 | 난이도별 분산, CI/CD에 통합해도 시간/비용 감당 가능    |
| 대규모       | 500+문항   | 팀 단위 운영, 이슈·인시던트 이력 누적 후 확장          |

**양보다 질 — 좋은 Golden Dataset 조건**:

- **실제 유저 질문 기반**: 만든 질문이 아닌 서비스 로그/CS 기반 질문이어야 실전 성능 예측 가능
- **함정 케이스 포함**: 년도 혼동·용어 혼동 등 RAG가 실패하기 쉬운 패턴을 의도적 포함
- **회귀 이력 보존**: 과거 버그 재현 질문을 별도 태깅 → 다시 틀리면 즉시 탐지
- **난이도 분산**: easy/medium/hard 고른 분포로 메트릭 민감도 확보

**`ground_truth_contexts`가 수동 어노테이션인 이유**: Ragas `ContextRecall`과 `LLMContextPrecisionWithReference`는 "검색된 청크가 ground truth 청크를 얼마나 포함하는가"를 측정. LLM으로 자동 생성하면 "LLM이 만든 근거를 LLM으로 평가"하는 순환 참조가 되어 검색 품질의 실제 문제를 포착할 수 없음.

---

### 2. LLM-as-a-Judge

#### 2-1. 왜 체계적 평가가 필요한가

- **회귀 탐지 불가**: 프롬프트 한 줄 바꿨을 때 20문항 중 몇 개가 뒤바뀌는지 눈으로 봐서는 알 수 없음. 소프트웨어 테스트는 결정적 입출력(assert x == y)이지만 LLM은 확률적 — 통계적 평가 필요
- **디버깅 지점 모호**: 답변이 틀렸을 때 "검색이 놓쳤나, 생성이 환각했나, 프롬프트가 나쁜가" 중 어디가 문제인지 정답률 하나로는 불가
- **비교 기준 부재**: "이 설정이 더 낫다"는 합의 기준이 없으면 모델·파라미터 선택이 직관에 의존
- **프로덕션 판단 불가**: "몇 점이면 쓸 수 있나"를 도메인별로 정해놓지 않으면 배포 여부 논의 불가

**회귀 검증(regression test)과 Golden Dataset**: Golden Dataset이 곧 regression set. 변경 전/후 같은 입력으로 돌려 차이를 metric delta로 포착. CI에 통합하면 PR마다 자동 검증.

**자동 vs 사람 평가 트레이드오프**:

| 축     | 자동 (LLM Judge)              | 사람                        |
| ------ | ----------------------------- | --------------------------- |
| 비용   | 메트릭당 $수 센트             | 1문항당 수 분 인력          |
| 속도   | 수 분                         | 수 시간~일                  |
| 일관성 | 모델 결정론에 가까움 (temp=0) | 판정자별 편차 큼            |
| 신뢰성 | 프롬프트/모델에 따라 흔들림   | 도메인 전문성이 있으면 높음 |
| 스케일 | 수천 문항 가능                | 수십 문항이 한계            |

#### 2-2. LLM-as-a-Judge 작동 원리

LLM에게 루브릭과 판정 기준을 프롬프트로 제공 → LLM이 답변을 읽고 점수·근거를 출력. Ragas의 `Faithfulness`, `AnswerCorrectness` 등 핵심 메트릭이 이 원리 위에 구축됨.

**등장 배경**: BLEU/ROUGE 등 n-gram 자동 평가는 의미적 동치를 잡지 못함(예: "5%"와 "100분의 5"). 사람 평가는 비용·시간·확장성 한계. 그 사이 빈틈을 LLM-as-a-Judge가 채움.

**사람 평가와의 일치율**: GPT-4 Judge는 사람 평가와 약 80~85% 일치율을 보인다는 연구(Zheng et al. 2023 MT-Bench)가 있으나, 도메인·프롬프트에 따라 크게 달라짐.

**루브릭 설계 원칙**:

- **나쁨**: "답변이 좋은가? 1~10점" — 추상적이라 모델이 흔들림
- **좋음**: "답변이 ground truth의 핵심 수치와 단위를 정확히 포함하는가? 0/1" — 기준이 명확
- Chain-of-Thought를 요구하면 판정 근거까지 나와 디버깅 가능

**한계와 신뢰성**:

- 프롬프트·모델에 따라 결과가 흔들림 → 공정 비교 위해 Judge 모델·프롬프트를 고정하고 추적 필요
- 생성용과 같은 모델을 Judge로 쓰면 자기 답변에 후한 경향 → **다른 패밀리 권장**. 본 과제에서는 OpenAI gpt-4.1-mini를 최종 채택 (생성용=Gemini)
- 호출 비용이 누적됨 (5메트릭 × 20문항 × 2파이프라인 = 200+ 호출)

**Ragas 메트릭과의 관계**:

| 메트릭                               | LLM Judge?                         | 규칙 기반? |
| ------------------------------------ | ---------------------------------- | ---------- |
| LLM Context Recall                   | ✅                                 |            |
| LLM Context Precision with Reference | ✅                                 |            |
| Faithfulness                         | ✅ (claim 추출 → 컨텍스트 대조)    |            |
| Answer Relevancy                     | 일부 (질문 역생성 → 임베딩 유사도) | ✅         |
| Answer Correctness                   | ✅ + 임베딩 유사도                 |            |

---

### 3. Ragas 4대 메트릭 + Answer Correctness

#### 3-1. 검색 단계

| 구분              | Context Recall                                                                               | Context Precision (with Reference)                                                          |
| ----------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 정의              | ground truth의 모든 주장(claim)이 검색된 청크에 존재하는가                                   | 검색된 청크 중 ground truth와 관련된 것의 비율 (상위 랭크일수록 가중)                       |
| 계산 방식         | Judge LLM이 ground_truth에서 claim 추출 → 각 claim이 retrieved_contexts에 있는지 판정 → 비율 | Judge LLM이 각 청크별 relevance(0/1) 판단 → rank 가중 평균 (1위 relevance가 가장 크게 반영) |
| 낮을 때 의심할 점 | 청킹 전략 부적절, k가 작음, 임베딩·BM25 커버 범위 부족                                       | 무관한 청크가 상위, Re-ranker 필요, 메타데이터 필터링 미적용                                |
| 개선 기법         | top-k 증가, Hybrid Search(BM25+벡터), 청크 크기 조정, 쿼리 확장(HyDE)                        | Re-ranking (Cohere/CrossEncoder), 메타데이터 필터링, 더 정교한 임베딩                       |

#### 3-2. 생성 단계

| 구분              | Faithfulness                                                                                                  | Answer Relevancy                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 정의              | 답변의 모든 claim이 검색된 컨텍스트로 뒷받침되는가 (환각 체크)                                                | 답변이 원 질문에 직접적으로 응답하는가                                                         |
| 계산 방식         | Judge LLM이 답변에서 claim 추출 → 각 claim이 contexts에서 도출 가능한지 판정 → 비율                           | Judge LLM이 답변에서 "이 답변을 부를 만한 원 질문"을 N개 역생성 → 원 질문과 임베딩 유사도 평균 |
| 낮을 때 의심할 점 | LLM이 외부 지식으로 빈칸 채움, 프롬프트에 "컨텍스트 밖 금지" 강제 부재, 컨텍스트가 부족해 LLM이 보완하려 시도 | 질문과 동떨어진 답변(주제 벗어남), LLM이 질문 의도 오해                                        |
| 개선 기법         | 프롬프트 강화("컨텍스트 밖 정보 금지"), 컨텍스트 품질 개선(검색 단계 선개선), Faithfulness-기반 re-generation | 프롬프트에 "질문에 직접 답하라" 명시, 답변 구조화, 질문 재작성(self-query)                     |

#### 3-3. End-to-End — Answer Correctness

**정의**: 답변이 ground truth와 의미적으로 일치하는가 — 사실 일치도와 의미 유사도의 가중 평균.

- **사실 일치도 (factual)**: Judge LLM이 답변·ground_truth를 claim 단위로 분해 → TP/FP/FN 산출 → F1
- **의미 유사도 (semantic)**: 임베딩 코사인 유사도
- **기본 가중**: 0.75 × factual + 0.25 × semantic

**ground_truth 품질 의존성**: ground_truth가 "180,000원"처럼 단편이면 의미 유사도가 낮게 나옴. "2025년 2종 수급권자의 틀니 본인부담금은 180,000원입니다."처럼 완전한 문장이어야 제대로 측정됨.

**Answer Correctness만으로 부족한 이유**: 점수가 낮을 때 검색/생성 어디가 원인인지 모름. 4대 메트릭으로 단계 분리 진단 필수.

#### 3-4. 메트릭 간 관계

| 시나리오                           | 낮아지는 메트릭                                    | 원인                                                                       | 대응                                          |
| ---------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------- |
| 정답 청크 자체를 검색이 놓침       | Context Recall, Answer Correctness                 | top-k 부족, 임베딩/BM25 커버 불충분                                        | k↑, Hybrid Search, 청킹 재조정                |
| 정답 청크는 있지만 8~10위로 밀림   | Context Precision (Answer Correctness는 유지 가능) | 관련성 점수 계산이 노이즈에 취약                                           | Re-ranker (Cohere/CrossEncoder)               |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness ↓ (나머지는 유지)                     | 프롬프트 제약 부족, LLM이 "빈칸 채우기"                                    | 프롬프트에 "컨텍스트 밖 금지" 명시, 포맷 고정 |
| LLM이 질문을 잘못 이해             | Answer Relevancy ↓                                 | 모호한 질문, LLM의 의도 오해                                               | 질문 재작성, few-shot 제공                    |
| 답은 맞는데 장황                   | (해당 없음 — 왜?)                                  | 장황함은 기본 메트릭으로 잡히지 않음 — 답변의 factual correctness는 유지됨 | 커스텀 메트릭 정의 필요(예: brevity rubric)   |

> **왜 "장황함"이 잡히지 않는가**: Ragas 기본 메트릭은 "컨텍스트와의 정합성, ground_truth와의 의미 일치"에만 초점. "표현 품질"은 프로덕션 관점의 UX 지표로, 별도 루브릭(`MetricWithLLM` 상속)으로 정의해야 함.

---

## Golden Dataset 확장 전략

### 정제 원칙

- **`ground_truth`**: "년도 + 대상 + 조건 + 값" 순 한 문장. 4주차 `expected_answer`가 이미 이 형태를 대부분 만족 → 그대로 채택.
  - 예: `"180,000원입니다. 65세 이상 2종 수급권자의 등록 틀니 본인부담률은 15%이므로, 1,200,000원 × 15% = 180,000원입니다."`
- **`ground_truth_contexts`**: FAISS 청크(PDF 원본 텍스트 발췌)에서 keyword 기반으로 매칭. LLM 자동 생성 아님(TASK 금지 조항 준수)
  - 매칭 방법: 각 질문의 핵심 키워드 리스트(예: `["장기지속형 주사제", "2%"]`)를 정의 → 모든 청크에 대해 키워드 포함 개수로 스코어링 → 최상위 청크 선택
  - cross-year 질문(q10, q13, q18): 2025/2026 양 연도 청크 각 1개씩 포함
- 구현 스크립트: [`build_dataset_v2.py`](build_dataset_v2.py) — 20문항 전수 자동 매칭

### cross-year 처리

- q10 (조산아 지원기간 2025 vs 2026), q13 (폐쇄병동 가산 12%→16%), q18 (장기지속형 주사제 5%→2%)
- `ground_truth_contexts`에 두 연도 청크 포함
- `source_year` 필드는 `"2025, 2026"`으로 복수 값 저장

---

## Step 2: Ragas 자동 평가 결과 (20문항 전수)

### 전체 평균 메트릭

| 메트릭                           | Basic RAG | Advanced RAG | Δ (Advanced - Basic) |
| -------------------------------- | --------: | -----------: | -------------------: |
| Context Recall                   |     0.850 |        0.850 |            **0.000** |
| Context Precision (w/ Reference) |     0.867 |        0.842 |           **-0.025** |
| Faithfulness                     |     0.509 |        0.576 |               +0.068 |
| Answer Relevancy                 |     0.692 |        0.725 |               +0.033 |
| Answer Correctness               |     0.696 |        0.751 |           **+0.055** |

**충격적인 발견**: 5문항 파일럿(Gemini Flash Judge)에서는 Advanced가 Context Recall +0.50이었으나, **전수(20문항, gpt-4.1-mini Judge)에서는 Recall이 동일(0.85)하고 Precision은 오히려 Advanced가 -0.025 악화**. Advanced RAG의 우위는 **Answer Correctness +5.5%p에 그침** — 4주차 수동 채점의 +10%p(80→90%) 차이와 어느 정도 부합.

### 문항별 상세 (B=Basic, A=Advanced)

| ID  | 난이도 | source_year | Ctx Recall (B/A) | Ctx Precision (B/A) | Faithfulness (B/A) | Ans Relevancy (B/A) | Ans Correctness (B/A) |
| --- | ------ | ----------- | :--------------: | :-----------------: | :----------------: | :-----------------: | :-------------------: |
| q01 | medium | 2026        |   1.00 / 1.00    |     1.00 / 0.83     |    0.40 / 0.20     |     0.74 / 0.75     |      0.74 / 0.98      |
| q02 | hard   | 2025        |   0.67 / 0.67    |     0.58 / 0.83     |    0.67 / 0.29     |     0.00 / 0.75     |      0.22 / 0.83      |
| q03 | medium | 2025        |   1.00 / 1.00    |     1.00 / 1.00     |    0.25 / 0.33     |     0.77 / 0.75     |      0.93 / 0.91      |
| q04 | medium | 2026        |   1.00 / 1.00    |     1.00 / 1.00     |    0.25 / 0.33     |     0.73 / 0.73     |      0.98 / 0.80      |
| q05 | medium | 2025        |   1.00 / 1.00    |     1.00 / 0.83     |    0.40 / 0.33     |     0.75 / 0.77     |      0.66 / 0.89      |
| q06 | medium | 2026        |   1.00 / 1.00    |     1.00 / 0.83     |    0.00 / 0.33     |     0.76 / 0.72     |      0.92 / 0.86      |
| q07 | easy   | 2026        |   0.00 / 0.00    |     0.58 / 1.00     |    0.67 / 0.78     |     0.72 / 0.00     |      0.38 / 0.49      |
| q08 | medium | 2026        |   1.00 / 1.00    |     0.83 / 0.83     |    0.10 / 0.80     |     0.79 / 0.70     |      0.68 / 0.69      |
| q09 | easy   | 2026        |   1.00 / 1.00    |     1.00 / 0.83     |    0.80 / 1.00     |     0.82 / 0.81     |      0.57 / 0.99      |
| q10 | hard   | 2025+2026   |   1.00 / 0.00    |     1.00 / 0.00     |    1.00 / 0.55     |     0.83 / 0.81     |      0.92 / 0.56      |
| q11 | medium | 2026        |   0.00 / 1.00    |     0.00 / 1.00     |    1.00 / 0.75     |     0.00 / 0.79     |      0.22 / 0.54      |
| q12 | hard   | 2025        |   1.00 / 0.33    |     1.00 / 0.00     |    1.00 / 1.00     |     0.76 / 0.76     |      0.45 / 0.41      |
| q13 | easy   | 2025+2026   |   1.00 / 1.00    |     0.33 / 1.00     |    1.00 / 1.00     |     0.81 / 0.80     |      0.95 / 0.70      |
| q14 | easy   | 2026        |   1.00 / 1.00    |     1.00 / 1.00     |    1.00 / 0.60     |     0.76 / 0.77     |      0.45 / 0.84      |
| q15 | medium | 2025        |   0.33 / 1.00    |     1.00 / 1.00     |    0.33 / 0.50     |     0.76 / 0.74     |      0.91 / 0.88      |
| q16 | medium | 2026        |   1.00 / 1.00    |     1.00 / 1.00     |    0.11 / 0.33     |     0.74 / 0.74     |      0.79 / 0.70      |
| q17 | medium | 2025        |   1.00 / 1.00    |     1.00 / 0.83     |    0.00 / 0.43     |     0.72 / 0.74     |      0.92 / 0.85      |
| q18 | hard   | 2025+2026   |   1.00 / 1.00    |     1.00 / 1.00     |    0.07 / 0.90     |     0.78 / 0.84     |      0.86 / 0.65      |
| q19 | medium | 2026        |   1.00 / 1.00    |     1.00 / 1.00     |    0.57 / 0.50     |     0.88 / 0.79     |      0.67 / 0.85      |
| q20 | hard   | 2026        |   1.00 / 1.00    |     1.00 / 1.00     |    0.56 / 0.57     |     0.73 / 0.73     |      0.71 / 0.62      |

### 승/패 분포 (Answer Correctness 기준)

| 구분              | 개수 | 질문 ID                                                                                                                            |
| ----------------- | ---: | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Advanced 우위** |  9건 | q01(+0.25), q02(+0.62), q05(+0.23), q07(+0.10), q08(+0.01), q09(+0.42), q11(+0.32), q14(+0.39), q19(+0.18)                         |
| **Basic 우위**    | 11건 | q03(-0.02), q04(-0.19), q06(-0.07), q10(-0.36), q12(-0.04), q13(-0.25), q15(-0.03), q16(-0.09), q17(-0.07), q18(-0.21), q20(-0.08) |

> Advanced가 크게 이기는 문항과 Basic이 크게 이기는 문항이 혼재 — **"Advanced가 항상 낫다"는 4주차 결론을 재검토해야 함**.

### 4주차 수동 채점 vs 5주차 Ragas 비교 (대표 문항)

| 질문 ID | 4주차 Basic 판정 | Ragas AnsCorr (Basic) | 일치          | 4주차 Advanced 판정 | Ragas AnsCorr (Advanced) | 일치          |
| ------- | ---------------- | --------------------: | ------------- | ------------------- | -----------------------: | ------------- |
| q01     | 정답             |                  0.74 | ✅            | 정답                |                     0.98 | ✅            |
| q02     | **오답**         |                  0.22 | ✅ (낮음)     | 정답                |                     0.83 | ✅            |
| q07     | **오답**         |                  0.38 | ✅ (낮음)     | **오답**            |                     0.49 | ✅ (중간)     |
| q10     | 정답             |                  0.92 | ✅            | **오답**            |                     0.56 | ✅ (낮음)     |
| q11     | **오답**         |                  0.22 | ✅ (낮음)     | 정답                |                     0.54 | 🟡 중간값     |
| q12     | 정답             |                  0.45 | 🟡 Ragas 엄격 | 정답                |                     0.41 | 🟡 Ragas 엄격 |

> q12 (이상지질혈증 2025 미적용)처럼 답은 맞으나 ground_truth와 문장 구조·설명 방식이 다르면 Ragas Answer Correctness가 0.4~0.5에 머무는 케이스가 상당수. **Ragas가 사람보다 더 엄격**.

---

## Step 3: Basic vs Advanced 다차원 비교

### 3-1. 개선/악화 차원

| 구분     | Ragas 5메트릭에서 개선/악화된 차원                                              |
| -------- | ------------------------------------------------------------------------------- |
| **개선** | Faithfulness (+0.068) · Answer Correctness (+0.055) · Answer Relevancy (+0.033) |
| **동일** | Context Recall (0.00)                                                           |
| **악화** | **Context Precision (-0.025)**                                                  |

> Advanced RAG의 Re-ranker가 **"옳은 청크를 상위로 올린다"는 기대와 달리 Context Precision을 오히려 악화**시킴. Rerank 상위 3개 중 상위 1개의 relevance가 낮아지는 사례가 누적된 영향으로 해석.

### 3-2. 년도 혼동 재진단

**년도 혼동이 Ragas 메트릭에 직접 반영되지 않는다는 가설 — 부분 검증**:

- **q10 (조산아 2025 vs 2026 cross-year)**: Basic은 retrieval이 성공(Context Recall 1.00)해 정답 도출, Advanced는 메타데이터 필터가 잘못 걸려 Recall 0.00 → AnsCorr 0.56으로 급락. **Advanced의 오답이 Recall 하락으로 포착됨**.
- **q13 (폐쇄병동 12%→16% cross-year)**: 두 파이프라인 모두 Recall 1.00이지만 Advanced AnsCorr는 0.70으로 Basic(0.95)보다 낮음. **retrieval은 충분했으나 생성이 두 연도 수치를 혼용**. Faithfulness는 둘 다 1.00으로 잡히지 않았으며, **Answer Correctness의 factual component만이 문제를 포착**.
- **q18 (장기지속형 주사제 5%→2% cross-year)**: Advanced AnsCorr 0.65로 Basic(0.86)보다 낮음. Advanced가 cross-year 비교 문항에서 메타 필터를 적용하지 않았는데도 연도 수치 혼용이 발생.

**결론**: 연도 혼동은 **Context Recall 하락(메타 필터 오적용 케이스) + Answer Correctness 하락(생성 단계 연도 혼용 케이스)** 두 경로로 나타남. 단 "둘 다 높은데 답변 연도만 틀린" 순수 혼동 케이스는 기본 메트릭으로 못 잡음 → **YearAccuracy 커스텀 메트릭(심화 A) 필요**.

### 3-3. 인사이트

**3문단 정리**:

**① 4주차 "Advanced가 +10%p 낫다"는 결론은 5주차 Ragas로 부분 지지되지만 패턴이 복잡함.** Answer Correctness 평균 +5.5%p는 4주차 +10%p와 방향은 같으나 폭이 절반. 20문항 중 **Basic 우위 11건, Advanced 우위 9건**으로 승률만 보면 비등. Advanced의 가치는 특정 유형(q02 년도 함정 +0.62, q09 조산아 단일 연도 +0.42, q11 2026 신설 항목 +0.32)에서 집중 발현되며, cross-year 비교 문항(q10 -0.36, q13 -0.25, q18 -0.21)에서는 오히려 손해.

**② Faithfulness는 두 파이프라인 모두 ~0.5대 중반 — 프로덕션 임계값(0.9) 미달.** 도메인 관례상 Faithfulness ≥ 0.9가 "환각 없음"의 기준. 현재는 Basic 0.51, Advanced 0.58로 한참 못 미침. 주 원인은 **LLM이 "본인부담금 180,000원입니다. 1,200,000 × 15% = 180,000원이므로..."처럼 계산식을 친절히 덧붙이면서 컨텍스트에 없는 arithmetic claim을 만들어내는 것**. 프롬프트에 "계산식은 제시하되 계산 결과 외 새 사실 주장 금지" 지시를 추가하거나, 계산 파트를 별도 체인으로 분리하는 구조 변경이 필요.

**③ 개선 우선순위 — Context Precision 회복 > Faithfulness 향상 > cross-year 대응.** Advanced의 Precision 역전(-0.025)은 Re-ranker 설정 또는 k값 재검토가 필요한 신호. Faithfulness는 프롬프트 한 줄로 잡을 수 있는 저비용 개선 대상. cross-year 문항은 메타 필터 로직 고도화(질문 자연어 시제 파싱)와 Year Accuracy 커스텀 메트릭이 세트. 심화 B(평가→개선→재평가)는 Faithfulness 단일 target이 ROI 가장 높음.

### 3-4. 기법별 기여도 관찰

| 방식                                | Context Recall | Context Precision | Answer Correctness |
| ----------------------------------- | -------------: | ----------------: | -----------------: |
| Basic RAG (벡터 Top-3)              |          0.850 |             0.867 |              0.696 |
| Advanced RAG (Hybrid+Rerank+Filter) |          0.850 |             0.842 |              0.751 |

> 두 파이프라인의 **Recall이 동일**(0.85)한 이유: golden_dataset_v2의 `ground_truth_contexts`가 FAISS 청크와 완전 동일한 문단이라, top-3에 들어오기만 하면 Recall이 1.0으로 계산됨 → 차이가 덜 부각. 더 민감한 비교를 위해서는 ground_truth_contexts를 PDF 원문에서 별도 발췌(청크와 부분 중복)하거나 chunk 크기를 작게 재구성해야 함.

---

## Step 4: 실패 케이스 Deep Dive

### Case A: Advanced가 Basic보다 악화된 문항 — q10 (cross-year)

**질문 (hard, 2025+2026)**:

> 의료급여 담당 공무원입니다. 2025년 4월에 태어난 조산아와 2026년 4월에 태어난 조산아의 의료급여 지원 종료 시점을 각각 안내해야 합니다. 두 경우의 지원 만료일이 다른가요?

**참고 정답**:

> 네, 다릅니다. 2025년 4월에 태어난 조산아는 5년 지원 규정에 따라 2030년 4월까지 지원됩니다. 2026년 4월에 태어난 조산아는 2026년 1월 1일부터 시행된 확대 규정(5년 4개월)에 따라 2031년 8월까지 지원됩니다.

**Basic 답변**: "...2030년 4월 말... 2031년 8월 말..." (정답)
**Advanced 답변**: "...두 경우 만료일이 동일하며 2031년 4월까지..." (오답)

| 메트릭                 |    Basic | Advanced |
| ---------------------- | -------: | -------: |
| Context Recall         | **1.00** | **0.00** |
| Context Precision      |     1.00 |     0.00 |
| Faithfulness           |     1.00 |     0.55 |
| Answer Relevancy       |     0.83 |     0.81 |
| **Answer Correctness** | **0.92** | **0.56** |

**원인 분석**:

- 이 질문은 cross-year이므로 `extract_year()`가 "2025"와 "2026" 둘 다 감지 → 필터 미적용(`filter_used="none"`)이어야 함
- 실제 로그 확인 결과 Advanced의 retrieved_years=['2026', '2025'] 즉 양 연도 청크 검색됨 — 그런데도 Ragas Recall 0
- **Ragas가 본 문제**: `ground_truth_contexts`에 2025년 청크(5년 지원)와 2026년 청크(5년 4개월)가 함께 포함돼 있는데, Advanced retrieved_contexts가 특정 청크와 부분 중복되지 않아 Recall 0 판정
- 즉 **chunking 단위가 너무 커서 "정답 영역"이 하나의 거대 청크에 뭉쳐있고, retrieve된 다른 청크가 그 청크와 완전 일치하지 않으면 0점** 구조
- Ragas가 가장 잘 드러낸 메트릭: **Context Recall** (1.0 → 0.0). 청크 경계 문제를 명확히 노출
- **조치**: ① 청크 크기 축소(섹션 대신 테이블/문단 단위) ② `ground_truth_contexts`를 실제 근거 문단만 좁게 발췌

### Case B: 년도 혼동 발생 — q02 (Basic 실패, Advanced 성공)

**질문 (hard, 2025)**:

> 아버지는 조현병으로 2종 의료급여 수급권자이십니다. 작년(현재 2026년 기준) 6월 10일에 병원 외래에서 항정신병 장기지속형 주사를 맞으셨고 주사 비용이 200,000원이었습니다. 그 당시 아버지가 내신 본인부담금은 얼마였나요?

**참고 정답**:

> 10,000원입니다. 2025년 6월 기준 항정신병 장기지속형 주사제 본인부담률은 5%이므로, 200,000원 × 5% = 10,000원입니다.

**Basic 답변**: "정보를 찾을 수 없습니다." (오답)
**Advanced 답변**: "...2025년 6월 10일에는 5% 적용... 200,000 × 5% = **10,000원**" (정답)

| 메트릭                 |    Basic | Advanced |
| ---------------------- | -------: | -------: |
| Context Recall         |     0.67 |     0.67 |
| Context Precision      |     0.58 |     0.83 |
| Faithfulness           |     0.67 |     0.29 |
| Answer Relevancy       | **0.00** |     0.75 |
| **Answer Correctness** | **0.22** | **0.83** |

**원인 분석**:

- 두 파이프라인 모두 Recall 0.67로 근거 청크는 대부분 확보
- Basic: "정보 없음" 응답으로 Answer Relevancy가 0 (질문과 답이 매치 안 됨)
- Advanced: BM25가 "장기지속형 주사제" 키워드 정확 매칭 + 2026년 문서 내 "2025년 10월 이전 요율 5%" 이력 정보 활용 → 정답 도출
- Advanced Faithfulness가 0.29로 낮은 이유: 계산식(200,000 × 5% = 10,000) arithmetic claim 감점
- Ragas가 가장 잘 드러낸 메트릭: **Answer Relevancy + Answer Correctness**. 검색 차원은 비슷한데 생성 단계에서 극명한 차이 발생

### Case C (선택): Faithfulness 높은데 Answer Correctness 낮음 — q12

**질문 (hard, 2025)**:

> 부모님이 2025년 8월에 국가건강검진을 받으셨는데 이상지질혈증 의심 판정을 받으셨습니다. 당시 확진검사 본인부담금 면제 혜택을 적용받으실 수 있었나요?

**참고 정답**:

> 아닙니다. 이상지질혈증 질환 의심 시 확진검사 본인부담 면제는 2026년 1월 1일부터 시행된 제도입니다.

**Basic 답변**: "2025년 8월에는... 적용받으실 수 없었습니다. 2025년 1월 1일부터는 우울증·조기정신증에 대해서만 시행..." (정답)

| 메트릭                 |    Basic |
| ---------------------- | -------: |
| Context Recall         |     1.00 |
| Context Precision      |     1.00 |
| **Faithfulness**       | **1.00** |
| Answer Relevancy       |     0.76 |
| **Answer Correctness** | **0.45** |

**원인 분석**:

- Faithfulness 1.0 = 모든 claim이 컨텍스트로 뒷받침됨 (환각 없음)
- Context Recall 1.0 = 정답 근거 청크가 retrieved_contexts에 포함
- 그런데 Answer Correctness는 0.45로 낮음
- Ragas AnsCorr = 0.75 × factual_F1 + 0.25 × semantic
- 답변은 ground_truth가 없는 "2025년 1월부터 우울증·조기정신증 한정" 같은 **추가 사실**을 포함 → factual claim 분해 시 ground_truth에 없는 claim = FP(false positive) 다수 → F1 하락
- **교훈**: Ragas AnswerCorrectness는 "ground_truth에 없는 추가 정보를 말해도 감점". 실무에서는 유용할 수 있는 **부가 설명이 메트릭 관점에서는 노이즈**. 프로덕션에서는 ground_truth를 "정답 + 허용 범위의 부가 설명"으로 두 개 필드로 관리하는 접근이 필요할 수 있음

### 공통 교훈

- **Ragas의 판정 엄격도는 Judge LLM에 크게 좌우됨**. 파일럿(Gemini Flash, 5문항)에서 Advanced가 Context Recall +0.50으로 압도했으나, 전수(gpt-4.1-mini, 20문항)에서는 Recall 동일·Precision 역전. **생성용과 같은 family를 Judge로 쓰면 자기 편향** 가능성 체감.
- **Faithfulness 저조의 구조적 원인**: 의료급여 도메인처럼 **수치 계산이 핵심**인 분야에서 LLM이 "친절한 설명"으로 arithmetic claim을 생성하는 패턴이 일관. 프롬프트에 "계산식은 허용하되 새 사실 생성 금지" 지시가 low-hanging fruit.
- **cross-year 문항은 오히려 Advanced의 약점**. q10/q13/q18 세 문항 모두 Basic 우위. 메타데이터 필터 로직이 단순 regex(`2025|2026`)라 양 연도 모두 등장 시 필터 해제되는데, 이 경우 Re-ranker가 오히려 한 연도 청크를 상위로 올려버림. → cross-year 전용 retrieval 분기(양 연도 강제 포함) 설계가 필요.
- **Ragas가 4주차 수동 채점보다 엄격**. "숫자만 맞으면 정답"이었던 4주차 대비 Ragas는 문장 구조·claim 단위·ground_truth 부합 여부를 모두 평가 → 같은 "정답"도 0.4~0.98 넓은 분포. 재현 가능한 회귀 테스트로는 더 우수하지만, 프로덕션 deploy-ready 기준은 도메인별 임계값 재설정 필요.

---

## 가설 vs 실제 결과 비교

| 가설                                                       | 실습 전 예측                                      | 실습 후 결과                                                                                                                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4주차 정답률(사람)과 Ragas Ans Correctness(자동) 일치 정도 | Ans Correctness가 사람 채점과 약 80% 이상 일치    | **대체로 부합**. 4주차 정답 → AnsCorr ≥ 0.5, 4주차 오답 → AnsCorr ≤ 0.5 경향. 단 q12처럼 "정답인데 AnsCorr 0.45" 케이스 존재 (Ragas 엄격)                                                  |
| Basic/Advanced 네 메트릭 중 가장 크게 벌어질 메트릭        | Context Recall (메타 필터링·Hybrid이 재현율 상승) | **불발견**. 20문항 전수에서 Recall 동일(0.85). 오히려 Faithfulness +0.068이 최대 Δ. 파일럿에서 본 +0.50 Recall 개선은 Judge LLM 편향에 기인                                                |
| 년도 혼동 문제가 주로 반영될 메트릭                        | Faithfulness 또는 Answer Correctness              | **부분 적중**. cross-year 문항(q10, q13, q18)에서 Advanced AnsCorr 하락이 명확, 단 순수 "연도만 틀린" 케이스는 기본 메트릭으로 직접 포착 안 됨 → 심화 A YearAccuracy 필요                  |
| Advanced에서 Faithfulness가 오히려 낮아질 시나리오         | Re-ranker가 무관한 청크를 상위에 배치             | **관찰 사례 있음**. q01/q06/q14/q19에서 Advanced Faithfulness가 Basic보다 낮음 — Cohere Rerank가 일부 계산·설명 맥락을 잃어버린 청크를 상위로 올려 답변이 컨텍스트 외 정보에 의존하게 만듦 |

---

## 파일 구조

```
week-5/1hjjun/
├── .env                            # API 키 (gitignore)
├── .gitignore
├── README.md                       # (본 파일)
├── golden_dataset.json             # 4주차 원본 20문항
├── golden_dataset_v2.jsonl         # + ground_truth / ground_truth_contexts (5주차 확장)
├── indexing.py                     # PDF → FAISS 인덱싱
├── build_dataset_v2.py             # dataset v2 생성 스크립트
├── medical_advanced_index/         # FAISS 인덱스 (gitignore)
├── week2/
│   └── LLMevaluating.py            # 2주차 Context LLM
├── week3/
│   ├── RAGevaluating.py            # Basic RAG
│   └── RAGresult.json              # 실행 결과
├── week4/
│   ├── 4주차README.md              # 4주차 종합 문서
│   ├── AdvancedRAG.py              # Advanced RAG
│   └── advanced_result.json        # 실행 결과
└── week5/
    ├── ragas_evaluate.py           # Ragas 자동 평가
    ├── basic_ragas_scores.csv      # Basic 문항별 메트릭 (20행)
    ├── advanced_ragas_scores.csv   # Advanced 문항별 메트릭 (20행)
    └── ragas_summary.json          # 평균 요약
```

## 참고 자료

- [Ragas 공식 문서](https://docs.ragas.io/)
- [Ragas Metrics — Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [Ragas Metrics — Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217)
- [Judging LLM-as-a-Judge — Zheng et al. 2023](https://arxiv.org/abs/2306.05685)
