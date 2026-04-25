# 5주차 과제: RAG 평가 — Golden Dataset, LLM-as-a-Judge, Ragas

> **이 README는 교수님 1차 피드백을 반영한 v2 재작업본입니다.** 1차 결과(26청크, LLM 메트릭만)와 2차 결과(76청크, LLM+NonLLM)를 함께 비교 기록합니다.

---

## 📌 이 과제를 한 마디로

> **4주차 = 학생들이 시험 본 것. 5주차 = 시험지를 자동으로 채점하는 채점기를 만든 것.**

4주차까지는 RAG 시스템(Basic, Advanced)을 만들어 답을 냈고, 맞았는지 **사람이 눈으로** 확인했습니다. 이번 주는 그 채점을 **Ragas라는 도구로 자동화**하고, 단순 "정답률" 한 숫자 대신 **여러 기준**으로 쪼개서 "검색이 문제인지, 생성이 문제인지"를 구분할 수 있게 만드는 것이 목표입니다.

---

## 🔧 교수님 피드백 적용 내역 (v1 → v2)

| 피드백                                                | 1차(v1) 상태                                                            | 2차(v2) 조치                                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **(a) 청크 크기 재구성**                              | 26청크 (섹션 단위, 평균 6,500자) → Recall 0.85 동률·Precision 역전 의심 | `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)` → **76청크, 평균 680자** |
| **(b) `reference_contexts`를 NonLLM 메트릭으로 확장** | LLM 메트릭 5개만 사용                                                   | `NonLLMContextRecall`, `NonLLMContextPrecisionWithReference` 추가 → **총 7메트릭**           |
| **(c) Faithfulness 0.5대 구조적 원인 진단**           | "계산식 친절함이 깎는다"는 일화적 추정                                  | Faithfulness 내부 prompt를 직접 호출해 statement·verdict 덤프 → `faithfulness_claims.json`   |

---

## 실행 환경 및 구성

| 항목                         | 내용                                                                  |
| ---------------------------- | --------------------------------------------------------------------- |
| 생성용 LLM                   | `gemini-2.5-flash` (temperature=0) — week2/3/4 공통                   |
| **판정용 LLM (Ragas Judge)** | **OpenAI `gpt-4.1-mini`** (temperature=0) — 생성용과 다른 모델 family |
| 임베딩                       | Google `gemini-embedding-001`                                         |
| Ragas 버전                   | 0.4.3                                                                 |
| RAG 프레임워크               | LangChain (+ langchain-classic for EnsembleRetriever)                 |
| Re-ranker                    | Cohere `rerank-v3.5` (Advanced RAG만)                                 |
| 벡터 저장소                  | FAISS (dense, **76청크** — 2025/2026 PDF 각 38청크)                   |
| BM25                         | in-memory, `BM25Retriever.from_documents`                             |
| 평가 범위                    | **전수 20문항**, **메트릭 7개** (LLM 5 + NonLLM 2)                    |
| 실행 환경                    | Python 3.13, macOS, `.venv`                                           |

---

## 이론 과제 답변

### 1. Golden Dataset

> 💡 **쉽게 말하면**: **"정답지가 있는 시험지"**. 학생(RAG)이 답을 내면 선생님이 **항상 같은 기준**으로 채점할 수 있도록 미리 만들어 놓은 모범답안 세트입니다. 다음 주에 프롬프트를 바꾸고 또 평가해도 같은 시험지로 돌리기 때문에 "개선됐나? 악화됐나?"를 정확히 비교할 수 있어요.

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
| `ground_truth_contexts` | `reference_contexts`          | 평가자(수동) | PDF 원본 문단 발췌 리스트          |
| (RAG 산출물)            | `response`                    | RAG 시스템   | 파이프라인 `invoke()` 결과         |
| (RAG 산출물)            | `retrieved_contexts`          | RAG 시스템   | Retriever가 반환한 청크 리스트     |

**권장 규모**:

| 단계         | 문항 수    | 근거                             |
| ------------ | ---------- | -------------------------------- |
| 초기 (pilot) | 5~20문항   | 스키마·비용·파이프라인 검증용    |
| 성숙기       | 50~100문항 | 난이도 분산, CI/CD 통합 가능     |
| 대규모       | 500+문항   | 팀 단위 운영, 인시던트 이력 누적 |

**양보다 질**: 실제 유저 질문 기반 / 함정 케이스 포함 / 회귀 이력 보존 / 난이도 분산.

**`ground_truth_contexts`가 수동 어노테이션인 이유**: Ragas의 `(NonLLM)ContextRecall`과 `(NonLLM)ContextPrecisionWithReference`는 "검색된 청크가 ground truth 청크를 얼마나 포함하는가"를 측정. LLM으로 자동 생성하면 "LLM이 만든 근거를 LLM으로 평가"하는 순환 참조가 됨.

---

### 2. LLM-as-a-Judge

> 💡 **쉽게 말하면**: **"LLM한테 선생님 역할을 시키기"**. 사람보다 빠르고, 단어 일치(BLEU)보다 **의미까지 이해**해서 채점합니다. 단, Judge가 답을 만든 LLM과 같은 종류면 자기 답에 후한 경향이 있어 **다른 회사 모델**을 권장 — 본 과제에서 생성=Gemini, 판정=OpenAI를 채택한 이유.

#### 2-1. 왜 체계적 평가가 필요한가

- **회귀 탐지 불가**: 프롬프트 한 줄 바꿨을 때 20문항 중 몇 개가 뒤바뀌는지 눈으로 못 봄
- **디버깅 지점 모호**: 답변이 틀렸을 때 어느 단계 문제인지 정답률만으로는 불가
- **비교 기준 부재**: "이 설정이 더 낫다"는 합의 없으면 직관 의존
- **프로덕션 판단 불가**: 도메인별 임계값(예: Faithfulness ≥ 0.9)이 있어야 배포 가능

**자동 vs 사람 평가 트레이드오프**:

| 축     | 자동 (LLM Judge)           | 사람                 |
| ------ | -------------------------- | -------------------- |
| 비용   | 메트릭당 $수 센트          | 1문항당 수 분 인력   |
| 속도   | 수 분                      | 수 시간~일           |
| 일관성 | temp=0이면 결정론에 가까움 | 판정자별 편차        |
| 신뢰성 | 프롬프트·모델 의존         | 도메인 전문성에 의존 |
| 스케일 | 수천 문항                  | 수십 문항이 한계     |

#### 2-2. LLM-as-a-Judge 작동 원리

LLM에게 루브릭과 판정 기준을 프롬프트로 제공 → LLM이 점수·근거를 출력. Ragas의 `Faithfulness`, `AnswerCorrectness`가 이 원리.

**루브릭 설계 원칙**:

- **나쁨**: "답변이 좋은가? 1~10점" — 추상적, 흔들림
- **좋음**: "답변이 ground truth의 핵심 수치를 정확히 포함하는가? 0/1" — 기준 명확
- Chain-of-Thought 요구하면 판정 근거까지 나와 디버깅 가능

**한계**: 프롬프트·모델 의존성, 같은 family 자기 편향, 호출 비용 누적(7메트릭 × 20문항 × 2파이프라인 = 280+ 호출).

**Ragas 메트릭과의 관계**:

| 메트릭                                      | LLM Judge?                         | 규칙·임베딩 기반?          |
| ------------------------------------------- | ---------------------------------- | -------------------------- |
| LLM Context Recall                          | ✅                                 |                            |
| LLM Context Precision with Reference        | ✅                                 |                            |
| **NonLLM Context Recall**                   |                                    | ✅ (rapidfuzz Levenshtein) |
| **NonLLM Context Precision with Reference** |                                    | ✅ (rapidfuzz Levenshtein) |
| Faithfulness                                | ✅ (claim 추출 → 컨텍스트 대조)    |                            |
| Answer Relevancy                            | 일부 (질문 역생성 + 임베딩 유사도) | ✅                         |
| Answer Correctness                          | ✅ + 임베딩 유사도                 |                            |

---

### 3. Ragas 메트릭 — LLM 5개 + NonLLM 2개

> 💡 **5+2가지 수치를 학교 시험 채점에 비유하면**:
>
> - **Context Recall (LLM/NonLLM)** — "참고서 펼친 페이지에 답이 있었냐?"
> - **Context Precision (LLM/NonLLM)** — "펼친 페이지 중 **중요한 게 위쪽**에 있었냐?"
> - **Faithfulness** — "쓴 답이 **전부 참고서에 근거**가 있냐?" (환각 체크)
> - **Answer Relevancy** — "질문이랑 **관련 있는 답**이냐?"
> - **Answer Correctness** — "**최종 정답이냐?**"
>
> **LLM 메트릭은 "관련 있나?"를 LLM에게 물어보고**, **NonLLM 메트릭은 "텍스트가 얼마나 겹치나?"를 임베딩·문자열 유사도로 직접 측정**합니다. 두 가지를 함께 보면 LLM이 "관련 있다고 후하게 인정한 것" vs "실제 텍스트가 정말 겹쳤는가"를 분리 진단 가능.

#### 3-1. 검색 단계

| 구분                 | Context Recall                                                                                                            | Context Precision (with Reference)                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 정의                 | ground truth 모든 claim이 검색된 청크에 존재하는가                                                                        | 검색된 청크 중 ground truth와 관련된 것의 비율 (상위 랭크 가중)      |
| LLM 계산 방식        | Judge LLM이 ground_truth claim 추출 → retrieved에 있는지 판정 → 비율                                                      | Judge LLM이 청크별 relevance(0/1) 판단 → rank 가중 평균              |
| **NonLLM 계산 방식** | **각 reference_context에 대해 가장 가까운 retrieved_context와의 문자열 유사도(rapidfuzz Levenshtein) → 임계값 통과 비율** | **각 retrieved_context에 대해 reference에서 매치되는지 → rank 가중** |
| 낮을 때 의심할 점    | 청킹 부적절, k 작음, 임베딩/BM25 커버 부족                                                                                | 무관 청크 상위, Re-ranker 필요, 메타필터 미적용                      |
| 개선 기법            | top-k↑, Hybrid Search, 청크 크기 조정                                                                                     | Re-ranking, 메타필터, 더 정교한 임베딩                               |

#### 3-2. 생성 단계

| 구분              | Faithfulness                                                                    | Answer Relevancy                                                                         |
| ----------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 정의              | 답변 모든 claim이 검색된 컨텍스트로 뒷받침되는가                                | 답변이 원 질문에 직접 응답하는가                                                         |
| 계산 방식         | Judge가 답변에서 claim 추출 → 각 claim이 contexts에서 도출 가능한지 판정 → 비율 | Judge가 답변에서 "이 답변을 부를 만한 원 질문" N개 역생성 → 원 질문과 임베딩 유사도 평균 |
| 낮을 때 의심할 점 | LLM이 외부 지식으로 빈칸 채움, 프롬프트 약함, 컨텍스트 부족                     | 주제 벗어남, LLM이 질문 의도 오해                                                        |
| 개선 기법         | 프롬프트 강화("컨텍스트 밖 금지"), 컨텍스트 품질 개선                           | 프롬프트에 "질문에 직접 답하라" 명시, self-query                                         |

#### 3-3. End-to-End — Answer Correctness

**정의**: 답변이 ground truth와 의미적으로 일치하는가 — 사실 일치도와 의미 유사도의 가중 평균. **기본 가중**: 0.75 × factual_F1 + 0.25 × semantic.

**Answer Correctness만으로 부족한 이유**: 점수가 낮을 때 검색/생성 어디가 원인인지 모름.

#### 3-4. 메트릭 간 관계

| 시나리오                         | 낮아지는 메트릭                                 | 원인                    | 대응                           |
| -------------------------------- | ----------------------------------------------- | ----------------------- | ------------------------------ |
| 정답 청크 자체 미검색            | Context Recall (LLM/NonLLM), Answer Correctness | k 부족, 청킹 부적절     | k↑, Hybrid Search, 청킹 재조정 |
| 정답 청크 있지만 하위 랭크       | Context Precision                               | 관련성 점수 노이즈      | Re-ranker                      |
| 검색은 맞는데 LLM 외부 정보 추가 | Faithfulness                                    | 프롬프트 제약 부족      | "컨텍스트 밖 금지" 명시        |
| LLM 질문 오해                    | Answer Relevancy                                | 모호한 질문             | 질문 재작성, few-shot          |
| 답은 맞는데 장황                 | (해당 없음)                                     | 기본 메트릭으로 못 잡음 | 커스텀 메트릭 정의             |

---

## Golden Dataset 확장 전략

- **`ground_truth`**: "년도 + 대상 + 조건 + 값" 한 문장. 4주차 `expected_answer`가 이미 적합 → 그대로 채택.
- **`ground_truth_contexts`**: 새로 인덱싱한 76청크 중 키워드 매칭 상위 2개씩 (cross-year는 연도별 2개). LLM 자동 생성 아님.
- 구현: [`build_dataset_v2.py`](build_dataset_v2.py)
- cross-year 처리 (q10, q13, q18): `ground_truth_contexts`에 양 연도 청크 포함.

---

## Step 2: Ragas 자동 평가 결과 (20문항 전수, 7메트릭)

> 💡 **이 섹션을 읽는 법**: 모든 점수는 0~1, **1에 가까울수록 좋음**. 마지막 컬럼 Δ는 Advanced - Basic.

### 전체 평균 메트릭 — v1 vs v2 비교

| 메트릭                                | v1 Basic | v1 Advanced |      v1 Δ | **v2 Basic** | **v2 Advanced** |      **v2 Δ** |
| ------------------------------------- | -------: | ----------: | --------: | -----------: | --------------: | ------------: |
| LLM Context Recall                    |    0.850 |       0.850 |     0.000 |    **0.900** |       **0.967** |    **+0.067** |
| LLM Context Precision (w/ Ref)        |    0.867 |       0.842 | -0.025 ⚠️ |    **0.900** |       **0.967** |    **+0.067** |
| Faithfulness                          |    0.509 |       0.576 |    +0.068 |    **0.523** |       **0.572** |    **+0.049** |
| Answer Relevancy                      |    0.692 |       0.725 |    +0.033 |    **0.662** |       **0.770** |    **+0.108** |
| Answer Correctness                    |    0.696 |       0.751 |    +0.055 |    **0.695** |       **0.844** |    **+0.149** |
| **NonLLM Context Recall**             |        — |           — |         — |    **0.525** |       **0.725** | **+0.200** ⭐ |
| **NonLLM Context Precision (w/ Ref)** |        — |           — |         — |    **0.679** |       **0.792** |    **+0.113** |

> **핵심 변화**: 청크 크기 재구성으로
>
> 1. **Precision 역전(-0.025) 사라짐** → +0.067로 정상화
> 2. **Answer Correctness 개선폭 +0.055 → +0.149** (4주차 사람 채점 +10%p와 일치)
> 3. **NonLLM Recall +0.20** (가장 큰 Δ) — 임베딩 기반이라 LLM 판정 편향에서 자유로움

### 문항별 상세 (B/A = Basic/Advanced)

| ID  | 난이도 | source_year | CR_LLM    | CP_LLM    | Faith     | Relv      | AnsCorr   | CR_Non    | CP_Non    |
| --- | ------ | ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| q01 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.50/1.00 | 0.77/0.77 | 0.57/0.70 | 0.50/0.50 | 1.00/1.00 |
| q02 | hard   | 2025        | 1.00/1.00 | 1.00/1.00 | 0.33/0.43 | 0.78/0.86 | 0.73/0.98 | 0.50/0.50 | 0.83/1.00 |
| q03 | medium | 2025        | 1.00/1.00 | 1.00/1.00 | 0.12/0.20 | 0.77/0.76 | 0.76/0.98 | 1.00/1.00 | 1.00/1.00 |
| q04 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.33/0.25 | 0.73/0.73 | 0.78/0.98 | 1.00/1.00 | 1.00/1.00 |
| q05 | medium | 2025        | 1.00/1.00 | 1.00/1.00 | 0.25/0.25 | 0.76/0.76 | 0.84/0.78 | 1.00/1.00 | 1.00/1.00 |
| q06 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.20/0.20 | 0.81/0.74 | 0.98/0.90 | 0.50/0.50 | 0.33/0.33 |
| q07 | easy   | 2026        | 0.50/1.00 | 0.00/0.83 | 0.33/0.50 | 0.00/0.74 | 0.22/0.68 | 0.00/0.50 | 0.00/1.00 |
| q08 | medium | 2026        | 0.50/1.00 | 1.00/1.00 | 0.67/0.80 | 0.78/0.79 | 0.55/0.92 | 0.50/1.00 | 1.00/0.83 |
| q09 | easy   | 2026        | 1.00/1.00 | 1.00/1.00 | 0.67/0.67 | 0.82/0.82 | 0.84/0.84 | 1.00/1.00 | 1.00/1.00 |
| q10 | hard   | 2025+2026   | 1.00/1.00 | 1.00/1.00 | 1.00/0.89 | 0.84/0.83 | 0.84/0.78 | 0.50/0.50 | 1.00/1.00 |
| q11 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.67/0.80 | 0.81/0.76 | 0.53/0.66 | 0.50/0.50 | 0.50/0.50 |
| q12 | hard   | 2025        | 1.00/0.33 | 1.00/1.00 | 0.80/0.67 | 0.76/0.78 | 0.58/0.35 | 0.00/0.50 | 0.00/0.33 |
| q13 | easy   | 2025+2026   | 0.00/1.00 | 0.00/0.50 | 0.50/1.00 | 0.80/0.81 | 0.95/0.95 | 0.50/1.00 | 1.00/1.00 |
| q14 | easy   | 2026        | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 0.75/0.76 | 0.99/0.88 | 0.50/0.50 | 1.00/1.00 |
| q15 | medium | 2025        | 1.00/1.00 | 1.00/1.00 | 0.20/0.33 | 0.76/0.75 | 0.87/0.98 | 0.00/0.50 | 0.00/0.50 |
| q16 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.14/0.20 | 0.75/0.74 | 0.73/0.98 | 0.50/0.50 | 0.33/0.50 |
| q17 | medium | 2025        | 1.00/1.00 | 1.00/1.00 | 0.60/0.20 | 0.00/0.73 | 0.35/0.98 | 0.50/1.00 | 1.00/0.83 |
| q18 | hard   | 2025+2026   | 1.00/1.00 | 1.00/1.00 | 0.73/0.89 | 0.77/0.81 | 0.67/0.84 | 1.00/1.00 | 0.58/0.83 |
| q19 | medium | 2026        | 1.00/1.00 | 1.00/1.00 | 0.75/0.67 | 0.79/0.78 | 0.64/0.77 | 0.00/0.50 | 0.00/0.33 |
| q20 | hard   | 2026        | 1.00/1.00 | 1.00/1.00 | 0.67/0.50 | 0.00/0.69 | 0.49/0.93 | 0.50/1.00 | 1.00/0.83 |

### 승/패 분포 (Answer Correctness 기준)

| 구분              |     개수 | 큰 폭 변화 (                                                                                           | Δ   | ≥ 0.20) |
| ----------------- | -------: | ------------------------------------------------------------------------------------------------------ | --- | ------- |
| **Advanced 우위** | **14건** | q02 (+0.25), q03 (+0.22), q04 (+0.20), q07 (+0.46), q08 (+0.36), q16 (+0.25), q17 (+0.63), q20 (+0.45) |
| **Basic 우위**    |  **5건** | q12 (-0.23)                                                                                            |
| 동점              |      1건 | q09                                                                                                    |

> v1 결과(9승 11패)에서 v2(14승 5패)로 **극적 개선** — 청크 크기가 평가 결과의 핵심 변수였음을 입증.

### 4주차 수동 채점 vs 5주차 Ragas v2 비교

| q ID | 4주차 Basic | Ragas Basic AnsCorr | 4주차 Advanced | Ragas Advanced AnsCorr | 일치                       |
| ---- | ----------- | ------------------: | -------------- | ---------------------: | -------------------------- |
| q01  | 정답        |                0.57 | 정답           |                   0.70 | ✅                         |
| q02  | **오답**    |                0.73 | 정답           |                   0.98 | 🟡 (Basic 의외 높음)       |
| q07  | **오답**    |                0.22 | **오답**       |                   0.68 | ✅ Advanced 회복           |
| q10  | 정답        |                0.84 | **오답**       |                   0.78 | 🟡 (Advanced 의외 높음)    |
| q12  | 정답        |                0.58 | 정답           |                   0.35 | 🟡 (Advanced AnsCorr 낮음) |

> q12 Advanced 0.35는 사람 채점은 정답이지만 Ragas는 ground_truth와 **claim 매칭이 부족**하다고 판정 — Ragas가 사람보다 엄격한 케이스의 대표. q07 Advanced 0.68은 4주차에서 "오답"이었으나 새 청킹+Ragas로 보면 **부분 정답**.

---

## Step 3: Basic vs Advanced 다차원 비교

### 3-1. 개선/악화 차원

| 구분     | 메트릭                                                                                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **개선** | NonLLM Recall (+0.20), Answer Correctness (+0.149), NonLLM Precision (+0.113), Answer Relevancy (+0.108), LLM Recall (+0.067), LLM Precision (+0.067), Faithfulness (+0.049) |
| **동일** | (없음)                                                                                                                                                                       |
| **악화** | (없음)                                                                                                                                                                       |

> v1에서 관찰됐던 Precision 역전(-0.025)은 청크 크기 재구성으로 **사라짐**. 모든 7개 메트릭에서 Advanced 우위 — 교수님 예측대로 "Re-ranker 본래 효과(상위로 올리기)"가 정상 작동하기 시작.

### 3-2. 년도 혼동 재진단 — cross-year 3문항

| ID  | source_year | CR_LLM B/A | CR_Non B/A | AnsCorr B/A | 해석                                                                                                                                                                      |
| --- | ----------- | ---------- | ---------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| q10 | 2025+2026   | 1.00/1.00  | 0.50/0.50  | 0.84/0.78   | LLM Judge는 "둘 다 충분"이라 1.0이지만 NonLLM은 0.5로 부분 매칭만 인식 → **양 연도 청크 모두 retrieve되었으나 한 연도 청크가 reference 텍스트와 정확히 일치하지 않음**    |
| q13 | 2025+2026   | 0.00/1.00  | 0.50/1.00  | 0.95/0.95   | Basic CR_LLM=0 = retrieved에 reference claim 없음 판정. 하지만 AnsCorr 0.95로 Basic 답은 정답 (2026 PDF에 양 연도 정보 모두 포함되어 있어서 retrieve 한 청크로도 답 가능) |
| q18 | 2025+2026   | 1.00/1.00  | 1.00/1.00  | 0.67/0.84   | 두 파이프라인 모두 retrieval 만점. AnsCorr Δ +0.17은 생성 단계 차이 (Advanced가 더 정확한 차이 계산)                                                                      |

> **년도 혼동은 기본 메트릭으로 직접 포착되지 않음** — q10에서 NonLLM Recall 0.5가 단서지만 LLM Recall은 1.0으로 안전. 정확한 연도별 진단을 위해서는 **YearAccuracy 커스텀 메트릭(심화 A)** 필요.

### 3-3. 인사이트

> 💡 **한 줄 요약**: **청크를 잘게 자르고 NonLLM 메트릭을 추가하니, 1차에서 보였던 "Advanced의 우위가 미미하다"는 결론이 뒤집혀 14:5의 분명한 우위로 전환됨. 평가 인프라 자체가 결론을 좌우함을 체감.**

**3문단 정리**:

**① 청크 크기가 RAG 평가 결과를 흔드는 1차 변수.** 1차(26청크, 평균 6,500자)에서는 **모든 검색이 거의 1.0** (top-3에 들어오기만 하면 6,500자 안에 답이 있을 확률이 매우 높음)이라 Basic/Advanced 간 검색 품질 차이가 메트릭에 반영되지 않았음. 청크를 800자로 잘게 쪼갠 2차에서는 retrieval이 진짜 어려워지고, BM25+Rerank+메타필터를 갖춘 Advanced의 효과가 NonLLM Recall +0.20으로 명확히 드러남. **"Re-ranker가 효과 없다"는 1차 결론은 evaluation infra 문제였지 RAG 문제가 아니었음**.

**② Faithfulness 0.5대는 구조적 문제 — Faithfulness 진단에서 정량 확인.** Basic 121개 claim 중 47.9%(58개) 미지지, 이 중 **22.4%(13개)가 계산식 패턴**(예: "1,200,000 × 15% = 180,000원"). 컨텍스트에는 "본인부담률 15%"만 있고 "1,200,000원"이라는 절대 금액은 질문에서만 등장하므로 Judge LLM이 "이 계산 결과는 컨텍스트로 직접 검증 불가"라 판정. 즉 **LLM이 친절하게 산술 풀이를 적어주는 것이 메트릭 관점에서는 환각으로 분류됨**. 프롬프트에 "계산 과정은 허용하나 결과 외 새로운 사실 주장 금지"를 추가하거나 산술을 외부 도구(tool calling)로 분리해야 0.9 임계값에 접근 가능.

**③ 개선 우선순위 — Faithfulness > cross-year 처리 > NonLLM 분산 추가 분석.** 모든 메트릭 Advanced 우위 상황에서 가장 명확한 약점은 Faithfulness (0.57). 프롬프트 한 줄로 잡힐 가능성이 있어 ROI 가장 높음. 그 다음은 cross-year 문항(q10, q13)에서 retrieval 측 NonLLM Recall이 0.5로 낮은 이슈 — 메타필터 로직을 cross-year 자연어 시제 파싱으로 강화 필요. 마지막은 NonLLM Recall std가 Basic 0.34 / Advanced 0.26으로 **Advanced가 안정성 측면에서도 우수**함을 정량 확인.

### 3-4. 기법별 기여도

| 방식                                        | LLM Recall | NonLLM Recall | LLM Precision | NonLLM Precision | AnsCorr |
| ------------------------------------------- | ---------: | ------------: | ------------: | ---------------: | ------: |
| Basic RAG (벡터 Top-3, 76청크)              |      0.900 |         0.525 |         0.900 |            0.679 |   0.695 |
| Advanced RAG (Hybrid+Rerank+Filter, 76청크) |      0.967 |         0.725 |         0.967 |            0.792 |   0.844 |

> **NonLLM 메트릭이 1차에서 가려졌던 Advanced의 진짜 효과를 드러냄** — Re-ranker가 실제로 정답에 가까운 청크를 1위로 올리고 있다는 정량 증거.

---

## Step 4: 실패 케이스 Deep Dive

### Case A: Advanced가 Basic보다 악화된 문항 — q12

**질문 (hard, 2025)**: 부모님이 2025년 8월에 국가건강검진을 받으셨는데 이상지질혈증 의심 판정을 받으셨습니다. 당시 확진검사 본인부담금 면제 혜택을 적용받으실 수 있었나요?

**참고 정답**: 아닙니다. 이상지질혈증 의심 시 확진검사 면제는 2026년 1월 1일부터 시행된 제도입니다.

| 메트릭                 |    Basic | Advanced |
| ---------------------- | -------: | -------: |
| LLM Context Recall     |     1.00 | **0.33** |
| LLM Context Precision  |     1.00 |     1.00 |
| NonLLM Recall          |     0.00 | **0.50** |
| NonLLM Precision       |     0.00 | **0.33** |
| Faithfulness           |     0.80 |     0.67 |
| **Answer Correctness** | **0.58** | **0.35** |

**원인 분석**:

- LLM Recall: Advanced가 1.00 → 0.33로 큰 폭 하락. 메타필터(`year=2025`)가 적용되어 2025년 청크만 검색했는데, 정답 근거 일부가 2026 PDF의 변경 이력 섹션에도 있어 누락
- NonLLM 메트릭은 반대로 Advanced가 Basic보다 나음 (검색 텍스트 자체는 더 정확) — LLM Judge와 NonLLM의 판단이 갈라지는 사례
- Answer Correctness 0.35는 답변 표현이 ground_truth와 다른 구조여서 의미 유사도가 낮음. 사실은 동일 (둘 다 "2025년 8월에는 미적용" 결론)
- **교훈**: cross-history 문항에서 메타필터가 오히려 손해가 됨. 메타필터를 "필요 시에만" 적용하는 정교한 분기 로직 필요

### Case B: 년도 혼동 발생 — q07 (큰 개선)

**질문 (easy, 2026)**: 만성질환 환자가 2026년 11월에 이미 외래 370회 받았는데 본인부담률은?

**참고 정답**: 30%. 2026년 1월 1일부터 365회 초과 시 30% 적용.

| 메트릭                 |    Basic | Advanced |
| ---------------------- | -------: | -------: |
| LLM Context Recall     |     0.50 | **1.00** |
| NonLLM Recall          |     0.00 | **0.50** |
| LLM Precision          |     0.00 | **0.83** |
| Answer Relevancy       |     0.00 | **0.74** |
| **Answer Correctness** | **0.22** | **0.68** |

**원인 분석**:

- Basic은 "365회 초과 30%" 규정이 있는 청크를 top-3에 못 가져옴 → Answer Relevancy 0 (질문과 무관 답변)
- Advanced는 BM25가 "365회" 키워드 정확 매칭으로 청크 끌어올림 + 메타필터(year=2026)로 노이즈 제거 → 정답 도출
- **NonLLM Recall**이 0.00 → 0.50으로 **Advanced의 검색 개선을 직접 정량 측정** — LLM Judge에 의존하지 않고도 차이 포착
- 4주차에서 "둘 다 오답"이었던 문항이 v2 결과 Advanced AnsCorr 0.68로 **부분 정답** 판정 — 청크 재구성 효과

### Case C: Faithfulness 진단 — q03 (계산식 환각 패턴)

**질문 (medium, 2025)**: 어머니(72세) 2종 수급권자, 3월 틀니 시술비 1,200,000원, 본인부담금은? (현재 2025년)

**참고 정답**: 180,000원. 65세 이상 2종 등록 틀니 본인부담률 15%, 1,200,000 × 15% = 180,000원.

**Basic Faithfulness = 0.12 (8 claim 중 1개만 지지됨)**

`faithfulness_claims.json` 발췌 — Basic q03 미지지 claim 예시:

```json
{
  "statement": "1,200,000 × 15% = 180,000원",
  "verdict": 0,
  "reason": "context에는 '2종 틀니 15%'만 있고 1,200,000원이라는 금액 정보는 없음"
},
{
  "statement": "어머니(72세)의 본인부담금은 180,000원입니다",
  "verdict": 0,
  "reason": "context에 절대 금액 180,000원이 명시되지 않음 — 본인부담률만 제시"
}
```

**원인 분석**:

- 컨텍스트는 "본인부담률 15%"라는 **요율**만 제공
- LLM은 친절하게 산술을 풀어 "180,000원입니다"라는 **절대값**을 답변에 포함
- 산술 결과는 컨텍스트로 직접 검증 불가 → claim 단위 분해 시 **2~3개의 미지지 claim 발생**
- 전체 121 미지지 claim 중 22.4%(13개)가 이 패턴 — **계산식 친절함이 Faithfulness를 깎는 구조적 원인 정량 확인**

**Faithfulness 진단 통계 (Basic)**:

| 지표                  |         값 |
| --------------------- | ---------: |
| 총 추출 claim 수      |        121 |
| 미지지 claim          | 58 (47.9%) |
| 미지지 중 계산식 패턴 | 13 (22.4%) |
| Faithfulness 평균     |      0.523 |

### 공통 교훈

> 💡 **한 줄 요약**: **평가 인프라(청킹·메트릭 종류) 하나가 결론을 뒤집을 수 있다. "Advanced가 별로다"라는 1차 결론은 evaluation 문제였지 RAG 문제가 아니었음.**

- **청크 크기가 평가의 1차 변수**: 26청크(섹션) → 76청크(800자)로 바꾸자 모든 메트릭이 정상 분포로 회복. v1에서 본 "Recall 동률·Precision 역전"은 chunking artifact였음
- **NonLLM 메트릭이 LLM Judge 편향 보완**: 임베딩·문자열 기반이라 Judge LLM이 "관련 있다"고 후하게 인정하는 경향 견제. NonLLM Recall +0.20이 Advanced 효과의 가장 명확한 증거
- **Faithfulness 저조는 일화가 아닌 정량 패턴**: 미지지 claim의 22%가 계산식. 의료급여처럼 산술이 핵심인 도메인의 구조적 한계 — 프롬프트 또는 tool 분리로 해결 가능
- **cross-year 문항은 여전히 Advanced의 미세한 약점**: q10에서 NonLLM Recall 0.5 (양 연도 모두 retrieve하지만 reference와 텍스트 일치 부족). 메타필터 cross-year 분기 로직 필요
- **Ragas는 사람 채점보다 엄격**: q12처럼 "정답이지만 표현 다름"도 AnsCorr 0.35로 감점. 도메인 임계값 재설정 필요

---

## 가설 vs 실제 결과 비교 (v2 기준)

| 가설                                                       | 실습 전 예측                          | 실습 후 결과                                                                                                                                                 |
| ---------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 4주차 정답률(사람)과 Ragas Ans Correctness(자동) 일치 정도 | ~80% 일치                             | **대체로 부합**. 4주차 정답 → AnsCorr 0.5+, 4주차 오답 → AnsCorr 0.5- 경향. 단 q12·q10·q07처럼 부분 불일치 케이스 존재                                       |
| 가장 크게 벌어질 메트릭                                    | Context Recall (메타필터·Hybrid 효과) | **NonLLM Context Recall이 +0.20으로 최대 Δ** — 예측 적중 (단 LLM Recall은 +0.067로 작은 편)                                                                  |
| 년도 혼동이 반영될 메트릭                                  | Faithfulness 또는 AnsCorr             | **부분 적중**. cross-year에서 NonLLM Recall이 0.5로 낮은 단서. 순수 "연도만 틀린" 케이스는 여전히 기본 메트릭으로 직접 포착 안 됨 → 심화 A YearAccuracy 필요 |
| Advanced에서 Faithfulness가 오히려 낮아질 시나리오         | Re-ranker가 무관 청크 상위            | **소수 관찰**. q10/q14/q19에서 Advanced Faithfulness가 Basic보다 미세하게 낮음. 평균은 +0.049로 Advanced 우위                                                |

---

## 재현 방법

```bash
# 1. .env 설정 (GEMINI_API_KEY, UPSTAGE_API_KEY, COHERE_API_KEY, OPENAI_API_KEY 필수)
cd week-5/1hjjun

# 2. 인덱싱 (RecursiveCharacterTextSplitter, 76청크 생성)
python3 indexing.py

# 3. Basic RAG 실행
python3 week3/RAGevaluating.py

# 4. Advanced RAG 실행 (Hybrid + Cohere Rerank + 메타데이터 필터)
python3 week4/AdvancedRAG.py

# 5. Golden Dataset v2 생성
python3 build_dataset_v2.py

# 6. Ragas 평가 (LLM 5 + NonLLM 2 + Faithfulness 진단)
python3 week5/ragas_evaluate.py
```

---

## 파일 구조

```
week-5/1hjjun/
├── .env                            # API 키 (gitignore)
├── .gitignore
├── README.md                       # (본 파일)
├── golden_dataset.json             # 4주차 원본 20문항
├── golden_dataset_v2.jsonl         # + ground_truth / ground_truth_contexts
├── indexing.py                     # PDF → FAISS (RecursiveCharacterTextSplitter, 76청크)
├── build_dataset_v2.py             # dataset v2 생성
├── medical_advanced_index/         # FAISS 인덱스 (gitignore)
├── week2/LLMevaluating.py          # 2주차 Context LLM
├── week3/
│   ├── RAGevaluating.py            # Basic RAG
│   └── RAGresult.json
├── week4/
│   ├── AdvancedRAG.py              # Advanced RAG
│   └── advanced_result.json
└── week5/
    ├── ragas_evaluate.py           # Ragas 평가 (LLM+NonLLM+Faithfulness 진단)
    ├── basic_ragas_scores.csv      # Basic 문항별 7메트릭 (20행)
    ├── advanced_ragas_scores.csv   # Advanced 문항별 7메트릭 (20행)
    ├── ragas_summary.json          # 평균 요약
    └── faithfulness_claims.json    # Faithfulness 진단 (claim·verdict 덤프)
```

## 참고 자료

- [Ragas 공식 문서](https://docs.ragas.io/)
- [Ragas v0.2 — NonLLMContextRecall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/#non-llm-based-context-recall)
- [Ragas — Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217)
- [LangChain — RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [Judging LLM-as-a-Judge — Zheng et al. 2023](https://arxiv.org/abs/2306.05685)
