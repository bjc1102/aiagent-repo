# 5주차 과제: RAG 평가 — Golden Dataset, LLM-as-a-Judge, Ragas

## 실행 환경 및 구성

| 항목 | 내용 |
|------|------|
| 생성용 LLM | `gemini-2.5-flash` (temperature=0) — week2/3/4 공통 |
| 판정용 LLM (Ragas Judge) | `gemini-2.5-flash` (temperature=0) |
| 임베딩 | Google `gemini-embedding-001` |
| Ragas 버전 | 0.4.3 |
| RAG 프레임워크 | LangChain (+ langchain-classic for EnsembleRetriever) |
| Re-ranker | Cohere `rerank-v3.5` (Advanced RAG만) |
| 벡터 저장소 | FAISS (dense vector, 26청크 — 2025/2026 PDF 각 13청크) |
| BM25 | in-memory, `BM25Retriever.from_documents` |
| 실행 환경 | Python 3.13, macOS, `.venv` |

**Judge LLM 선택 과정 (문제 있었던 지점 기록):**
1. Claude Sonnet 4.5 계획 → ANTHROPIC_API_KEY 크레딧 부족으로 실패
2. GPT-4o 대체 시도 → `insufficient_quota` 에러
3. Gemini 2.5 Pro 시도 → 완주했으나 타임아웃 다발 (Advanced 결측치 85%)
4. Gemini 2.5 Pro + GPT-4o fallback (`with_fallbacks()`) → Ragas 내부 `temperature` 속성 접근 비호환으로 전부 실패
5. **최종: Gemini 2.5 Flash** — 무료 티어 사용 가능, 낮은 지연. 생성용과 같은 모델이라는 한계는 있음
6. **샘플 제한**: API 비용 관리를 위해 `RAGAS_LIMIT=5`로 첫 5문항만 평가

---

## 이론 과제 답변

### 1. Golden Dataset

**한 줄 정의**: RAG 시스템의 성능을 객관적으로 측정하기 위한 "정답이 명확한" 질문-답변 쌍 데이터셋.

**왜 필요한가**:
- 없으면 프롬프트·모델·검색 설정을 바꿨을 때 개선 여부를 주관적으로 판단하게 됨 → 회귀(regression) 탐지 불가
- 실무에서 변경이 잦을수록 자동 회귀 테스트 필수 — Golden Dataset이 그 기준점
- LLM의 확률적 출력 특성상 단위 테스트만으로는 커버 불가 → 의미적 일치도를 체계적으로 확인하는 수단

**필수 스키마 (Ragas v0.2+ 기준)**:

| JSONL 필드 | Ragas `SingleTurnSample` 필드 | 준비 주체 | 준비 방법 |
|-----------|----------------------------|---------|----------|
| `question` | `user_input` | 평가자 | 실제 유저 질문을 자연어로 작성 |
| `ground_truth` | `reference` | 평가자(수동) | PDF 근거를 완전한 문장 형태로 정제 |
| `ground_truth_contexts` | `reference_contexts` | 평가자(수동) | PDF 원본 문단 2~5문장 발췌 리스트 |
| (RAG 실행 산출물) | `response` | RAG 시스템 | 파이프라인 `invoke()` 결과 |
| (RAG 실행 산출물) | `retrieved_contexts` | RAG 시스템 | Retriever가 반환한 청크 리스트 |

> v0.1 필드명 (`question`/`answer`/`contexts`/`ground_truths`)은 v0.2+에서 deprecated. 블로그 튜토리얼 대부분이 v0.1 기준이라 최신 Ragas와 맞지 않음.

**권장 규모**:

| 단계 | 문항 수 | 근거 |
|-----|--------|------|
| 초기 (pilot) | 5~20문항 | 스키마·비용·파이프라인 검증용. 한 사이클 빠르게 돌리기 |
| 성숙기 | 50~100문항 | 난이도별 분산, CI/CD에 통합해도 시간/비용 감당 가능 |
| 대규모 | 500+문항 | 팀 단위 운영, 이슈·인시던트 이력 누적 후 확장 |

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

| 축 | 자동 (LLM Judge) | 사람 |
|----|---------------|-----|
| 비용 | 메트릭당 $수 센트 | 1문항당 수 분 인력 |
| 속도 | 수 분 | 수 시간~일 |
| 일관성 | 모델 결정론에 가까움 (temp=0) | 판정자별 편차 큼 |
| 신뢰성 | 프롬프트/모델에 따라 흔들림 | 도메인 전문성이 있으면 높음 |
| 스케일 | 수천 문항 가능 | 수십 문항이 한계 |

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
- 생성용과 같은 모델을 Judge로 쓰면 자기 답변에 후한 경향 → **다른 패밀리 권장** (본 과제에서는 인프라 제약으로 생성/판정 모두 Gemini Flash 사용)
- 호출 비용이 누적됨 (5메트릭 × 20문항 × 2파이프라인 = 200+ 호출)

**Ragas 메트릭과의 관계**:

| 메트릭 | LLM Judge? | 규칙 기반? |
|-------|-----------|---------|
| LLM Context Recall | ✅ | |
| LLM Context Precision with Reference | ✅ | |
| Faithfulness | ✅ (claim 추출 → 컨텍스트 대조) | |
| Answer Relevancy | 일부 (질문 역생성 → 임베딩 유사도) | ✅ |
| Answer Correctness | ✅ + 임베딩 유사도 | |

---

### 3. Ragas 4대 메트릭 + Answer Correctness

#### 3-1. 검색 단계

| 구분 | Context Recall | Context Precision (with Reference) |
|------|---------------|-----------------------------------|
| 정의 | ground truth의 모든 주장(claim)이 검색된 청크에 존재하는가 | 검색된 청크 중 ground truth와 관련된 것의 비율 (상위 랭크일수록 가중) |
| 계산 방식 | Judge LLM이 ground_truth에서 claim 추출 → 각 claim이 retrieved_contexts에 있는지 판정 → 비율 | Judge LLM이 각 청크별 relevance(0/1) 판단 → rank 가중 평균 (1위 relevance가 가장 크게 반영) |
| 낮을 때 의심할 점 | 청킹 전략 부적절, k가 작음, 임베딩·BM25 커버 범위 부족 | 무관한 청크가 상위, Re-ranker 필요, 메타데이터 필터링 미적용 |
| 개선 기법 | top-k 증가, Hybrid Search(BM25+벡터), 청크 크기 조정, 쿼리 확장(HyDE) | Re-ranking (Cohere/CrossEncoder), 메타데이터 필터링, 더 정교한 임베딩 |

#### 3-2. 생성 단계

| 구분 | Faithfulness | Answer Relevancy |
|------|-------------|------------------|
| 정의 | 답변의 모든 claim이 검색된 컨텍스트로 뒷받침되는가 (환각 체크) | 답변이 원 질문에 직접적으로 응답하는가 |
| 계산 방식 | Judge LLM이 답변에서 claim 추출 → 각 claim이 contexts에서 도출 가능한지 판정 → 비율 | Judge LLM이 답변에서 "이 답변을 부를 만한 원 질문"을 N개 역생성 → 원 질문과 임베딩 유사도 평균 |
| 낮을 때 의심할 점 | LLM이 외부 지식으로 빈칸 채움, 프롬프트에 "컨텍스트 밖 금지" 강제 부재, 컨텍스트가 부족해 LLM이 보완하려 시도 | 질문과 동떨어진 답변(주제 벗어남), LLM이 질문 의도 오해 |
| 개선 기법 | 프롬프트 강화("컨텍스트 밖 정보 금지"), 컨텍스트 품질 개선(검색 단계 선개선), Faithfulness-기반 re-generation | 프롬프트에 "질문에 직접 답하라" 명시, 답변 구조화, 질문 재작성(self-query) |

#### 3-3. End-to-End — Answer Correctness

**정의**: 답변이 ground truth와 의미적으로 일치하는가 — 사실 일치도와 의미 유사도의 가중 평균.
- **사실 일치도 (factual)**: Judge LLM이 답변·ground_truth를 claim 단위로 분해 → TP/FP/FN 산출 → F1
- **의미 유사도 (semantic)**: 임베딩 코사인 유사도
- **기본 가중**: 0.75 × factual + 0.25 × semantic

**ground_truth 품질 의존성**: ground_truth가 "180,000원"처럼 단편이면 의미 유사도가 낮게 나옴. "2025년 2종 수급권자의 틀니 본인부담금은 180,000원입니다."처럼 완전한 문장이어야 제대로 측정됨.

**Answer Correctness만으로 부족한 이유**: 점수가 낮을 때 검색/생성 어디가 원인인지 모름. 4대 메트릭으로 단계 분리 진단 필수. 예) Answer Correctness 낮은데 Faithfulness 높다 = 검색이 잘못. Faithfulness 낮다 = 생성이 환각.

#### 3-4. 메트릭 간 관계

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---------|---------------|-----|------|
| 정답 청크 자체를 검색이 놓침 | Context Recall, Answer Correctness | top-k 부족, 임베딩/BM25 커버 불충분 | k↑, Hybrid Search, 청킹 재조정 |
| 정답 청크는 있지만 8~10위로 밀림 | Context Precision (Answer Correctness는 유지 가능) | 관련성 점수 계산이 노이즈에 취약 | Re-ranker (Cohere/CrossEncoder) |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness ↓ (나머지는 유지) | 프롬프트 제약 부족, LLM이 "빈칸 채우기" | 프롬프트에 "컨텍스트 밖 금지" 명시, 포맷 고정 |
| LLM이 질문을 잘못 이해 | Answer Relevancy ↓ | 모호한 질문, LLM의 의도 오해 | 질문 재작성, few-shot 제공 |
| 답은 맞는데 장황 | (해당 없음 — 왜?) | 장황함은 기본 메트릭으로 잡히지 않음 — 답변의 factual correctness는 유지됨 | 커스텀 메트릭 정의 필요(예: brevity rubric) |

> **왜 "장황함"이 잡히지 않는가**: Ragas 기본 메트릭은 "컨텍스트와의 정합성, ground_truth와의 의미 일치"에만 초점. "표현 품질"은 프로덕션 관점의 UX 지표로, 별도 루브릭(`MetricWithLLM` 상속)으로 정의해야 함.

---

## Golden Dataset 확장 전략

### 정제 원칙

- **`ground_truth`**: "년도 + 대상 + 조건 + 값" 순 한 문장. 4주차 `expected_answer`가 이미 이 형태를 대부분 만족 → 그대로 채택.
  - 예: `"180,000원입니다. 65세 이상 2종 수급권자의 등록 틀니 본인부담률은 15%이므로, 1,200,000원 × 15% = 180,000원입니다."`
- **`ground_truth_contexts`**: FAISS 청크(PDF 원본 텍스트 발췌)에서 keyword 기반으로 매칭. LLM 자동 생성 아님(TASK 금지 조항 준수)
  - 매칭 방법: 각 질문의 핵심 키워드 리스트(예: `["장기지속형 주사제", "2%"]`)를 정의 → 모든 청크에 대해 키워드 포함 개수로 스코어링 → 최상위 청크 선택
  - cross-year 질문(q10, q13, q18): 2025/2026 양 연도 청크 각 1개씩 포함
- 구현 스크립트: [`build_dataset_v2.py`](week-5/1hjjun/build_dataset_v2.py) — 20문항 전수 자동 매칭, 1건씩 검증

### cross-year 처리

- q10 (조산아 지원기간 2025 vs 2026), q13 (폐쇄병동 가산 12%→16%), q18 (장기지속형 주사제 5%→2%)
- `ground_truth_contexts`에 두 연도 청크 포함
- `source_year` 필드는 `"2025, 2026"`으로 복수 값 저장

---

## Step 1~2: Ragas 자동 평가 결과

**평가 대상**: golden_dataset_v2 첫 5문항 (q01~q05). Judge = Gemini 2.5 Flash. 20문항 전수 평가는 API 비용·쿼터 제약으로 파일럿 크기로 축소.

### 전체 평균 메트릭

| 메트릭 | Basic RAG | Advanced RAG | Δ (Advanced - Basic) |
|--------|----------:|-------------:|---------------------:|
| Context Recall | 0.300 | 0.800 | **+0.500** |
| Context Precision (w/ Reference) | 0.700 | 0.867 | +0.167 |
| Faithfulness | 0.255 | 0.363 | +0.108 |
| Answer Relevancy | 0.677 | 0.877 | +0.201 |
| Answer Correctness | 0.717 | 0.882 | +0.166 |

**모든 메트릭에서 Advanced RAG가 우위.** 특히 Context Recall은 0.3 → 0.8로 극적 개선, BM25·메타데이터 필터링·Cohere Rerank 조합 효과가 명확히 숫자로 드러남.

### 문항별 메트릭 (B = Basic, A = Advanced)

| 질문 ID | 난이도 | source_year | Ctx Recall (B/A) | Ctx Precision (B/A) | Faithfulness (B/A) | Ans Relevancy (B/A) | Ans Correctness (B/A) |
|---------|--------|-------------|:----------------:|:-------------------:|:------------------:|:-------------------:|:---------------------:|
| q01 | medium | 2026 | 0.50 / 0.00 | 0.50 / 1.00 | 0.50 / 0.40 | 0.88 / 0.86 | 0.81 / 0.98 |
| q02 | hard | 2025 | 0.00 / 1.00 | 0.00 / 0.33 | 0.00 / 0.40 | 0.00 / 0.89 | 0.22 / 0.98 |
| q03 | medium | 2025 | 0.00 / 1.00 | 1.00 / 1.00 | 0.12 / 0.17 | 0.81 / 0.89 | 0.78 / 0.89 |
| q04 | medium | 2026 | 0.00 / 1.00 | 1.00 / 1.00 | 0.25 / 0.25 | 0.86 / 0.88 | 0.98 / 0.98 |
| q05 | medium | 2025 | 1.00 / 1.00 | 1.00 / 1.00 | 0.40 / 0.60 | 0.83 / 0.87 | 0.79 / 0.57 |

### 4주차 수동 채점 vs 5주차 Ragas 비교

| 질문 ID | 4주차 Basic 판정 | Ragas Ans Correctness (Basic) | 일치 여부 | 4주차 Advanced 판정 | Ragas Ans Correctness (Advanced) | 일치 여부 |
|---------|------------------|-------------------------------:|-----------|---------------------|---------------------------------:|-----------|
| q01 | 정답 | 0.81 | 일치 (>0.7) | 정답 | 0.98 | 일치 |
| q02 | **오답** | 0.22 | 일치 (<0.5) | 정답 | 0.98 | 일치 |
| q03 | 정답 | 0.78 | 일치 | 정답 | 0.89 | 일치 |
| q04 | 정답 | 0.98 | 일치 | 정답 | 0.98 | 일치 |
| q05 | 정답 | 0.79 | 일치 | 정답 | **0.57** | **불일치** — Ragas가 더 엄격 |

> q05 불일치: 4주차에서는 정답으로 판정했으나(72,000원 정확히 계산) Ragas는 0.57. Advanced RAG 답변이 ground_truth와 숫자는 일치하나 문장 구조/설명 방식이 다른 경우 Answer Correctness의 의미 유사도 항목이 낮게 산정될 수 있음. 향후 ground_truth 정제 원칙을 더 RAG 답변 형식에 가깝게 맞추면 개선 여지 있음.

---

## Step 3: Basic vs Advanced 다차원 비교

### 3-1. 개선/악화 차원

| 구분 | Ragas 5메트릭에서 개선/악화된 차원 |
|------|---------------------------|
| **개선** | Context Recall (+0.50, 50%p) · Answer Relevancy (+0.20) · Context Precision (+0.17) · Answer Correctness (+0.17) · Faithfulness (+0.11) — **전 영역 개선** |
| **악화** | (없음, 평균 기준) |

> 4주차 과제에서 "Advanced가 오히려 악화될 수 있다"는 가설은 이번 5문항 파일럿에서는 관찰되지 않음. 단 q05 Answer Correctness는 Basic 0.79 → Advanced 0.57로 단일 문항 악화 관찰.

### 3-2. 년도 혼동 재진단

- **q02 (hard, 2025)**: "작년(현재 2026년 기준)" 추론 — Basic은 전 메트릭 0 (거의 "정보 없음"으로 답변), Advanced는 Context Recall 1.0으로 완벽 복구
- 년도 혼동 자체는 Ragas 기본 5메트릭에 **직접적으로 포착되지 않음**. Basic q02가 실패한 이유는 "retrieved_contexts 내에서 LLM이 연도 추론에 실패"한 것인데, Context Recall은 단지 "청크가 ground truth를 포함하는가"만 봄.
- 즉 연도 혼동은 **Faithfulness가 낮지만 Context Recall은 높은 패턴**으로 나타날 수 있으나, claim 단위 분해가 섬세해야 잡힘 → 한계를 보완하려면 **커스텀 YearAccuracy 메트릭**(심화 A) 필요

### 3-3. 인사이트

**3문단 정리**:

**① 4주차 "Advanced가 낫다" 결론은 계량적으로 재확인됨.** 사람 채점 기준 +10%p(80%→90%)였던 것이 Ragas 5메트릭 전 차원에서 일관된 개선으로 다시 확인. 특히 Context Recall +0.50은 Basic RAG의 top-3 재현율 부족을 Advanced의 Hybrid+Rerank+메타데이터 필터링이 극적으로 보완함을 수치로 입증.

**② Faithfulness는 두 파이프라인 모두 낮음 (0.26 vs 0.36) — 프로덕션 임계값 미달.** 도메인 관례상 Faithfulness ≥ 0.9를 "환각 없음"의 기준으로 삼는데, 현재는 둘 다 멀리 못 미침. 원인은 LLM 답변이 "본인부담금은 180,000원입니다. 65세 이상 2종 수급권자의 등록 틀니 본인부담률은 15%이므로..."처럼 **계산 과정을 친절하게 덧붙이면서 컨텍스트에 없는 'arithmetic' claim(1,200,000 × 0.15 = 180,000)을 만들어내는 것**. 개선하려면 프롬프트에 "계산식은 제시하되 본인부담금 수치 외 새로운 사실 주장 금지" 지시를 넣거나, 계산 부분은 별도 체인으로 분리하는 구조 변경이 필요.

**③ 개선 우선순위: Faithfulness > (이미 0.88인 Ans Correctness/Relevancy).** Context Recall/Precision은 Advanced에서 이미 0.8+로 높아 다음 라운드에서는 생성 단계 프롬프트 튜닝이 ROI 높음. 심화 B(평가→개선→재평가)의 대상은 Faithfulness가 가장 적합.

### 3-4. 기법별 기여도 추정

| 방식 | Context Recall | Answer Correctness |
|------|---------------:|-------------------:|
| Basic RAG (벡터 Top-3) | 0.30 | 0.72 |
| Advanced RAG (Hybrid+Rerank+Filter) | 0.80 | 0.88 |

> Hybrid/Rerank/Metadata filter 기여를 분리 측정하려면 각 조합별 중간 실험이 필요. 본 과제에서는 통합 파이프라인 비교에 한정.

---

## Step 4: 실패 케이스 Deep Dive

### Case A: Advanced가 Basic보다 악화된 문항 — q05

**질문 (medium, 2025)**:
> 저는 2종 의료급여 수급권자입니다. 2025년 10월에 요추 디스크 협착증 진단을 받고 한방병원에서 복잡추나 치료를 받았습니다. 치료비가 180,000원이었는데 본인부담금은 얼마인가요?

**참고 정답**:
> 72,000원입니다. 2종 수급권자가 디스크·협착증으로 복잡추나를 받는 경우 본인부담률은 40%이므로, 180,000원 × 40% = 72,000원입니다.

**Basic RAG 답변**: "...180,000원 * 40% = 72,000원... 총 72,000원을 본인부담금으로 내셔야 합니다." (정답)

**Advanced RAG 답변**: "2025년 정보에 따르면, 2종 의료급여 수급권자가 요추 디스크 협착증으로 한방병원에서 복잡추나 치료를 받는 경우 본인부담률은 40%입니다. 따라서 치료비 180,000원에 대한 본인부담금은... 72,000원" (정답)

**메트릭 점수**:
| 메트릭 | Basic | Advanced |
|--------|------:|---------:|
| Context Recall | 1.00 | 1.00 |
| Context Precision | 1.00 | 1.00 |
| Faithfulness | 0.40 | 0.60 |
| Answer Relevancy | 0.83 | 0.87 |
| **Answer Correctness** | **0.79** | **0.57** |

**원인 분석**:
- 검색(Context Recall/Precision)·생성(Faithfulness/Relevancy) 모두 Advanced가 나은데 Answer Correctness만 역전
- Advanced 답변이 Basic보다 구조적 설명("2025년 정보에 따르면" 전제 + 과정 분리)을 더 많이 포함 → ground_truth와 **문장 구조 차이**가 커짐 → 임베딩 유사도 하락
- Ragas AnsCorr = 0.75 × factual_F1 + 0.25 × semantic. 사실은 같지만 의미 유사도가 표현 차이로 감점
- **교훈**: ground_truth를 RAG 답변의 일반적 표현 형식에 더 가깝게 정제하거나, AnsCorr 가중치(factual 비중↑)를 조정해야 이런 역전이 사라짐

### Case B: 년도 혼동 발생 — q02

**질문 (hard, 2025)**:
> 아버지는 조현병으로 2종 의료급여 수급권자이십니다. 작년(현재 2026년 기준) 6월 10일에 병원 외래에서 항정신병 장기지속형 주사를 맞으셨고 주사 비용이 200,000원이었습니다. 그 당시 아버지가 내신 본인부담금은 얼마였나요?

**참고 정답**:
> 10,000원입니다. 2025년 6월 기준 항정신병 장기지속형 주사제 본인부담률은 5%이므로, 200,000원 × 5% = 10,000원입니다. (해당 요율은 2025년 10월 1일부터 2%로 인하되었습니다.)

**Basic 답변**: "2025년도 컨텍스트에서는 조현병 환자가 외래에서 항정신병 장기지속형 주사를 맞았을 때의 본인부담금에 대한 정보를 찾을 수 없습니다." (오답 — 정보 없음)

**Advanced 답변**: "...2025년 10월 1일부터 5%→2%로 완화... 2025년 6월 10일에는 5% 적용... 200,000 × 5% = **10,000원**" (정답)

**메트릭 점수**:
| 메트릭 | Basic | Advanced |
|--------|------:|---------:|
| Context Recall | 0.00 | 1.00 |
| Context Precision | 0.00 | 0.33 |
| Faithfulness | 0.00 | 0.40 |
| Answer Relevancy | 0.00 | 0.89 |
| **Answer Correctness** | **0.22** | **0.98** |

**원인 분석**:
- Basic: 연도 추론 실패 + 검색 부족(Recall 0) → "정보 없음" 응답. 모든 메트릭 0 (Faithfulness도 0 — 답변 claim이 거의 없음)
- Advanced: BM25가 "장기지속형 주사제" 정확 매칭 + 메타데이터 필터(year=2026) 적용했음에도 **2026년 문서에 기재된 "2025년 10월 1일 이전에는 5%였음"이라는 이력 정보**를 활용해 답을 도출
- Ragas가 가장 잘 드러낸 메트릭: **Context Recall** (0 → 1.0). 검색 차원의 극적 개선을 즉시 포착
- Faithfulness가 Advanced에서도 0.40에 그친 이유: 계산식(200,000 × 5% = 10,000)이 컨텍스트에 없는 arithmetic claim으로 분리되어 감점
- **조치**: 이 유형은 메타데이터 필터링이 핵심 — 질문의 자연어 시제("작년" + "현재 2026년")를 더 정교하게 파싱하는 로직을 도입하면 Basic도 회복 가능

### 공통 교훈

- **Ragas는 "정답이지만 표현이 다른 경우" 감점 경향** (q05 Advanced). 프로덕션에서는 ground_truth의 표현 형식이 모델 답변 형식과 얼마나 가까운지가 AnsCorr에 큰 영향. ground_truth 작성 가이드라인 필요.
- **Faithfulness가 전반적으로 낮음 (0.26~0.36)** — 의료급여처럼 "수치 계산"이 핵심인 도메인에서, LLM이 컨텍스트에 없는 arithmetic 주장을 생성하는 패턴이 일관됨. 프롬프트 수준에서 "계산 근거는 명시하되 새 사실 주장 금지" 같은 지침 추가가 과제.
- **년도 혼동은 Ragas 기본 메트릭으로 직접 포착되지 않음** — Context Recall이 높은데 Answer Correctness가 낮은 패턴이 간접 신호지만, 확실한 진단을 위해서는 YearAccuracy 커스텀 메트릭(심화 A) 도입이 바람직.
- **4주차 수동 vs 5주차 자동 엄격도**: 수치만 같으면 "정답" 처리하던 4주차 대비, Ragas는 표현·구조·claim 단위까지 모두 평가 → **더 엄격**. 5문항 중 1건(q05) 불일치.
- **파일럿(5문항) vs 전수(20문항) 결과 예측**: 파일럿에서 관찰된 패턴(Advanced 전 영역 개선, Faithfulness 저조)이 전수에서도 유지될 가능성이 높음. 단 cross-year 문항(q10, q13, q18)에서 추가 insight 가능성 있음.

---

## 가설 vs 실제 결과 비교

| 가설 | 실습 전 예측 | 실습 후 결과 |
|------|-----------|----------|
| 4주차 정답률(사람)과 Ragas Ans Correctness(자동) 일치 정도 | Ans Correctness가 사람 채점과 약 80% 이상 일치할 것 | 5문항 중 4건 일치, 1건 불일치(q05). 80% 일치율로 예측 부합 |
| Basic/Advanced 네 메트릭 중 가장 크게 벌어질 메트릭 | Context Recall — 메타데이터 필터링과 Hybrid Search가 검색 재현율을 크게 올릴 것 | **적중**. Context Recall +0.50으로 5메트릭 중 최대 개선 폭 |
| 년도 혼동 문제가 주로 반영될 메트릭 | Faithfulness (외부 연도 정보 추정)나 Answer Correctness | 부분 적중. 실제로는 Basic에서는 모든 메트릭이 0으로 나타남(답변 자체가 "정보 없음"). 연도 혼동이 "환각"이 아니라 "거부"로 발현되어 Faithfulness 신호가 약함 |
| Advanced에서 Faithfulness가 오히려 낮아질 시나리오 | Re-ranker가 무관한 청크를 가져와 LLM이 기반을 잃는 경우 | **불발견** (5문항 한정). 평균 기준 Advanced Faithfulness가 Basic보다 오히려 +0.11 높음. 20문항 전수에서 재확인 필요 |

---

## 재현 방법

```bash
# 1. .env 설정 (GEMINI_API_KEY, UPSTAGE_API_KEY, COHERE_API_KEY 필수)
cd week-5/1hjjun

# 2. 인덱싱 (FAISS + source_year 메타데이터)
python3 indexing.py

# 3. Basic RAG 실행 (week3/)
python3 week3/RAGevaluating.py

# 4. Advanced RAG 실행 (week4/ — Hybrid + Cohere Rerank + 메타데이터 필터)
python3 week4/AdvancedRAG.py

# 5. Golden Dataset v2 생성 (ground_truth + ground_truth_contexts 추가)
python3 build_dataset_v2.py

# 6. Ragas 평가 (기본: 첫 5문항, 전수는 RAGAS_LIMIT 제거)
RAGAS_LIMIT=5 python3 week5/ragas_evaluate.py
```

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
    ├── basic_ragas_scores.csv      # Basic 문항별 메트릭
    ├── advanced_ragas_scores.csv   # Advanced 문항별 메트릭
    └── ragas_summary.json          # 평균 요약
```

## 참고 자료

- [Ragas 공식 문서](https://docs.ragas.io/)
- [Ragas Metrics — Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [Ragas Metrics — Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217)
- [Judging LLM-as-a-Judge — Zheng et al. 2023](https://arxiv.org/abs/2306.05685)
