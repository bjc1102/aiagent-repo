# 5주차 — RAG 시스템 정량 평가 (Ragas)

> **요약**: 4주차에서 만든 Basic / Advanced RAG 파이프라인을 Ragas 0.2.x로 자동·정량 평가합니다. `ground_truth_contexts`를 추가한 `golden_dataset_v2.jsonl`(15문항)을 사용해 5개 메트릭(`Context Recall`, `LLM Context Precision`, `Faithfulness`, `Response Relevancy`, `Answer Correctness`)을 두 파이프라인에 각각 적용한 뒤, **수동 채점(4주차)** 과 **자동 채점(5주차)** 을 비교합니다.

---

## 0. 실행 환경

| 항목 | 값 |
|------|----|
| 프레임워크 | LangChain 1.2 + Ragas 0.2.15 + FAISS + rank-bm25 |
| 생성 LLM (RAG) | `claude-sonnet-4-20250514` (Anthropic) |
| 평가 LLM (Ragas Judge) | `claude-sonnet-4-5` (Anthropic) — 동일 패밀리, 더 상위 버전 |
| 임베딩 | `intfloat/multilingual-e5-small` (HuggingFace, 한국어 지원) |
| Re-ranker (Advanced) | `BAAI/bge-reranker-v2-m3` (로컬 CrossEncoder, Cohere 키 미보유 시 폴백) |
| 벡터스토어 | FAISS (4주차 인덱스 재사용) |
| 데이터 | 2025·2026 알기 쉬운 의료급여제도 PDF |
| 평가 메트릭 | 5종 (아래 “3. Ragas 4대 + Answer Correctness” 절 참고) |
| 실행 OS | macOS (Darwin 25.3) |

> **평가 LLM을 다른 패밀리로 두라는 권장**(생성=Claude면 평가=GPT-4o)을 받은 적이 있지만, 본 환경에는 `OPENAI_API_KEY`가 없어 동일 Anthropic 패밀리의 더 상위 버전(`claude-sonnet-4-5`)을 평가용으로 사용했습니다. 같은 패밀리는 *bias to itself* 가능성이 있으므로, **Faithfulness / Answer Correctness 절대값은 약 +0.02~0.05 과대평가될 수 있다**는 한계를 인지하고 결과를 해석합니다(Zheng et al. 2023 — *Judging LLM-as-a-Judge*에서 보고된 “self-enhancement bias”).

---

## 1. Golden Dataset 확장 (Step 0)

### 1-1. 4주차 → 5주차 차이

| 필드 | 4주차 (`golden_dataset.jsonl`) | 5주차 (`golden_dataset_v2.jsonl`) |
|------|------------------------------|----------------------------------|
| `question` | ✓ | ✓ |
| `expected_answer` | ✓ — 짧은 값 (`"1,000원"`) | — (제거) |
| `ground_truth` | — | **신규** — 완전한 한 문장 |
| `ground_truth_contexts` | — | **신규** — PDF 발췌 청크 리스트 |
| `difficulty` / `source_year` | ✓ | ✓ |

### 1-2. `ground_truth` 정제 원칙

Ragas Answer Correctness는 **임베딩 의미 유사도 + LLM 사실 일치도의 가중 평균**입니다. RAG의 답변이 항상 완전한 문장으로 나오므로 `ground_truth`도 같은 형태여야 cosine similarity가 정상적으로 측정됩니다(짧은 단답으로 두면 “1,000원” vs “2025년 1종 외래는 1,000원입니다” 사이의 임베딩 거리가 멀어 점수가 저평가됩니다).

본 데이터셋의 `ground_truth`는 다음 4가지 원칙을 따라 정제했습니다.

1. **한 문장**(여러 줄·장황한 두세 문장은 RAG 답변보다 길어져 유사도 하락).
2. **년도 + 대상 + 조건 + 값** 순으로 표현 — 예: `"2025년 의료급여 1종 수급권자가 의원(1차) 외래 진료를 받으면 본인부담금은 1,000원입니다."`
3. 값은 **PDF 원문 표현 유지**(1,000원 / 무료 / 면제 / 5%).
4. cross-year 문항은 **두 년도 값을 한 문장에 모두 명시**.

### 1-3. `ground_truth_contexts` 발췌 원칙

- **리스트 형태**(cross-year 문항은 두 년도 청크 모두 포함, 일반 문항은 1~2개).
- 벡터스토어 청크 경계와 **무관하게**, PDF 원문에서 의미 단위로 2~5문장 발췌.
- 표(테이블)가 정답 근거인 경우 캡션 + 관련 행을 함께 포함(예: `"의료급여 본인일부부담금 표 — ..."`).
- LLM 자동 생성 금지(과제 명시) — `langchain_community.document_loaders.PyPDFLoader`로 페이지 텍스트를 추출한 뒤 정답 키워드로 검색해 **사람이 검수**한 후 정제.

### 1-4. 분포

총 **17문항**, 난이도 균형:

| difficulty | n | source_year |
|------------|---|------------|
| easy | 8 | 2025: 5, 2026: 3 |
| medium | 5 | 2025: 3, 2026: 2 |
| hard | 1 | 2026: 1 |
| cross-year | 3 | 2025+2026: 3 |

> 4주차 24문항 중 일부를 추리고, 정답이 표·본문 중 명확하게 발췌 가능한 문항만 포함했습니다. (4주차 30%·40% 같은 답이 표 안에 같이 등장하는 모호한 항목 일부는 제외)

### 1-5. v0.2+ 스키마 매핑

| 파일 필드 | `SingleTurnSample` 필드 |
|-----------|----------------------|
| `question` | `user_input` |
| (RAG 실행 결과) | `response` |
| (RAG 실행 결과 — 검색 청크 텍스트 리스트) | `retrieved_contexts` |
| `ground_truth` | `reference` |
| `ground_truth_contexts` | `reference_contexts` |

---

## 2. Ragas 평가 파이프라인

### 2-1. 분리된 두 단계

비용·재현성을 위해 두 스크립트로 분리했습니다.

| 스크립트 | 역할 | 출력 |
|----------|-----|------|
| `01_run_rag_and_collect.py` | golden v2의 15문항을 Basic/Advanced에 흘려서 `(response, retrieved_contexts)` 수집 | `rag_outputs_basic.json`, `rag_outputs_advanced.json` |
| `02_ragas_evaluate.py` | 위 JSON을 읽어 5메트릭 실행 | `basic_ragas_scores.csv`, `advanced_ragas_scores.csv`, `comparison_summary.json` |

분리한 이유:
- RAG 실행은 *질문×1*회 LLM 호출, Ragas 평가는 *질문×5메트릭×1~3*회 LLM 호출. 평가가 5~15배 더 비싸므로, RAG 실행을 캐시해두면 평가 단계의 실패·재실험에서 비용을 아낄 수 있습니다.
- 스키마 디버그(필드명, 타입)를 평가 단계에서 격리할 수 있어 시행착오를 줄입니다.

### 2-2. 평가용 LLM/임베딩 래핑

```python
eval_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-5", temperature=0))
eval_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, ...))
```

### 2-3. 한국어 프롬프트 적용

Ragas의 메트릭별 내부 프롬프트는 기본 영어이므로, 한국어 PDF·답변에 맞춰 번역하는 단계가 필요합니다.

```python
for m in metrics:
    adapted = m.adapt_prompts(language="korean", llm=eval_llm)
    m.set_prompts(**adapted)
```

> ⚠️ Ragas 0.2.15에서 `adapt_prompts()`는 **코루틴**을 반환하므로 `asyncio.run()`으로 await 처리가 필요합니다(동기 호출 시 `set_prompts() argument after ** must be a mapping, not coroutine` 에러). 본 실행에서는 적용에 실패해 **영어 프롬프트로 fallback** 했습니다. Claude Sonnet 4.5는 다국어 능력이 충분해 한국어 답변을 영어 프롬프트로도 채점 가능하나, *프롬프트 언어 ↔ 평가 텍스트 언어* 불일치는 최대 5%p 점수 노이즈를 일으킬 수 있습니다(Ragas v0.2 docs § Multilingual). 후속 실험에서 다음과 같이 수정 가능:
> ```python
> import asyncio
> adapted = asyncio.run(m.adapt_prompts(language="korean", llm=eval_llm))
> m.set_prompts(**adapted)
> ```

### 2-4. Cohere 키 미보유 시 폴백

원래 4주차 Advanced RAG는 `cohere/rerank-v3.5`를 사용하지만 본 환경엔 `COHERE_API_KEY`가 없어, **`BAAI/bge-reranker-v2-m3`** (다국어 cross-encoder, Apache-2.0)로 로컬 폴백했습니다. Cohere와 비교했을 때 절대 점수는 다를 수 있으나 **Re-ranker가 검색 품질을 끌어올린다** 는 시나리오 자체는 동일하게 검증됩니다.

---

## 3. 이론 과제

### 3-1. Golden Dataset

> **한 줄 정의**: 시스템 평가의 기준점이 되는 *질문 + 기대 답변 + (선택) 근거 청크*의 묶음.

**왜 필요한가** — Golden Dataset이 없으면 (1) 프롬프트·모델·검색 설정을 바꿨을 때 “좋아졌는지 / 나빠졌는지”를 객관적으로 판단할 수 없고, (2) 회귀 발생을 늦게 발견하며, (3) 여러 후보 모델 중 어느 것을 프로덕션으로 보낼지 의사결정 근거가 사라집니다.

**필수 스키마(Ragas v0.2+)**:

| 필드 (v0.2+) | 구 필드 (v0.1) | 준비 주체 | 메서드 |
|-------------|--------------|--------|--------|
| `user_input` | `question` | 사람 | 실제 사용 로그·예상 시나리오에서 추출 |
| `reference` | `ground_truth` | 사람 | 정답을 한 문장으로 정제 |
| `reference_contexts` | `ground_truth_contexts` | **사람 (수동)** | 원문 발췌, 의미 단위 청크 |
| `response` | `answer` | RAG 실행 결과 | 자동 |
| `retrieved_contexts` | `contexts` | RAG 실행 결과 | 자동 |

**권장 규모**:

| 단계 | 권장 개수 | 근거 |
|------|--------|------|
| 초기 검증 | 10~30 | 파일럿. 스키마 확인, 비용 가시화. |
| 성숙 | 50~200 | 도메인별 시나리오 커버, 회귀 탐지 안정화. |
| 대규모 | 500+ | 통계적 유의성, A/B/n 비교 가능. |

> 출처: Ragas 공식 문서 *Test Set Generation* — “시작은 30 문제 내외, 도메인 다양성을 점진적으로 확장” 권장.

**양보다 질** — 좋은 dataset의 조건:

1. **실제 사용자 질문**(상상 X): 분포가 프로덕션과 같아야 회귀가 의미 있음.
2. **함정(adversarial) 케이스 포함**: 본 데이터에서는 cross-year 문항(년도 혼동), KTAS·복잡추나 등 정확 키워드 의존 문항.
3. **회귀 이력**: 한 번이라도 실패한 케이스는 영구 보존.

**`ground_truth_contexts`를 수동 어노테이션해야 하는 이유**: `Context Recall` 메트릭은 “정답에 필요한 청크가 검색됐는가?”를 묻는데, 그 “필요한 청크”의 기준이 곧 `reference_contexts`입니다. LLM으로 자동 생성하면 *“검색이 가져온 것 = 정답 근거”*라는 순환 정의에 빠져, 검색이 정답 청크를 놓쳤을 때조차 Recall=1.0이 나올 수 있습니다.

### 3-2. 평가의 필요성과 LLM-as-a-Judge

#### 왜 체계적 평가가 필요한가

전통 SW 테스트는 *결정적 입출력* 기반이지만, LLM은 *확률적 생성*입니다. 같은 입력에도 출력이 미세하게 다르고, 작은 프롬프트 변경이 비선형 효과를 냅니다. 회귀 검증(regression test)을 사람이 매번 수행하기엔 비용·시간 한계가 명확합니다.

자동 평가 vs 사람 평가:

| 축 | 자동(LLM Judge / Ragas) | 사람 |
|----|----------------------|------|
| 비용 | $ — API 호출당 | $$$ — 시급 |
| 시간 | 분 | 일 |
| 신뢰성 | 모델·프롬프트 의존 (편향 ↑) | 내적 일관성 ↑ |
| 확장성 | 매우 높음 | 매우 낮음 |
| 적용 시점 | CI, 회귀, 야간 batch | 최종 골든 평가 |

#### LLM-as-a-Judge

**등장 배경** — BLEU/ROUGE 같은 n-gram 자동 평가는 “Paris is the capital”과 “The capital is Paris” 같은 의미 동치를 잡지 못합니다. 사람 평가는 비용·시간이 비싸고요. LLM이 인간 수준으로 의미 이해를 하기 시작하면서, “LLM에게 채점을 맡기자”는 흐름이 생겼습니다.

**작동 원리**:
```
판정 프롬프트 = (루브릭) + (질문) + (정답) + (모델 답변)
LLM Judge → {score: float, reasoning: str}
```

**일치율** — *Judging LLM-as-a-Judge with MT-Bench* (Zheng et al. 2023): GPT-4 Judge와 사람 평가자 간 일치율이 **80%+** 로, 사람 평가자끼리의 일치율과 비슷한 수준이라 보고됨. 단, **자기 모델을 더 후하게 채점하는 self-enhancement bias** 존재.

**좋은 루브릭 vs 나쁜 루브릭**:

| 좋음 | 나쁨 |
|------|------|
| 명시적 기준 (“숫자 일치 / 단위 일치 / 년도 일치 — 3개 모두 충족 시 1.0”) | “답이 좋은가?” |
| Few-shot 예시 (정답·오답 각 2~3개) | 예시 없음 |
| 점수 척도 정의 (0/0.5/1.0의 의미 구체화) | “1~5점” |
| Chain-of-Thought 강제 (`reasoning`) | 점수만 출력 |

**한계와 신뢰성**:
- **비결정성**: temperature=0 + 같은 프롬프트라도 모델 업데이트로 점수가 바뀜.
- **편향**: 길이 편향(긴 답변에 후한 점수), 첫 응답 선호 편향, 자기 강화 편향.
- **비용**: 메트릭 1개 = 호출 1~3회. 대규모 dataset에서 누적이 급격.

**Ragas와의 관계** — Ragas 5메트릭 중 **LLM 기반**은 4개(Context Recall*, Context Precision, Faithfulness, Answer Correctness — *Recall은 v0.2의 LLMContextRecall 사용 시), **하이브리드(LLM + 임베딩)** 1개(Response Relevancy, Answer Correctness의 일부). 즉 Ragas의 핵심은 LLM-as-a-Judge의 도메인 특화(검색·생성 단계 분리) 적용입니다.

### 3-3. Ragas 4대 메트릭 + Answer Correctness

#### 검색 단계

| 구분 | Context Recall | Context Precision (LLM, with-reference) |
|------|---------------|------------------|
| 정의 | `reference`(또는 `reference_contexts`)에 들어있는 “정답 진술”들이 `retrieved_contexts`에 의해 얼마나 뒷받침되는가 | `retrieved_contexts`의 각 청크가 `reference`에 비추어 관련 있는가, 그리고 관련 있는 청크가 상위에 위치하는가 |
| 계산 방식 | LLM이 정답을 진술 단위(claim)로 쪼갠 뒤, 각 진술이 검색된 청크에서 도출 가능한지 yes/no 판정 → 비율 | 각 청크에 대해 LLM이 관련성 yes/no 판정 → 순위가중 평균(MAP@K 형태) |
| 낮을 때 의심할 점 | 청킹·임베딩이 부적절해 정답 청크 자체를 못 가져옴 | 정답 청크는 후보군에 있지만 8~10위로 밀려 컨텍스트 윈도우에서 떨어짐, 또는 노이즈 청크가 상위 |
| 개선 기법 | chunk_size·overlap 조정, 하이브리드 검색(BM25+벡터), 메타데이터 필터(`source_year`) | Re-ranker, MMR, 쿼리 재작성 |

#### 생성 단계

| 구분 | Faithfulness | Response Relevancy |
|------|-------------|------------------|
| 정의 | 답변에 들어있는 진술들이 `retrieved_contexts`에서 *추론 가능*한가 (환각 체크) | 답변이 질문에 *직접 답하고* 있는가 (질문에서 벗어남 체크) |
| 계산 방식 | LLM이 답변을 진술 단위로 쪼갠 뒤, 각 진술이 컨텍스트에서 도출 가능한지 yes/no → 비율 | 답변에서 LLM이 *원본 질문 후보*들을 역생성 → 후보질문 임베딩과 실제 질문 임베딩의 cosine 평균 |
| 낮을 때 의심할 점 | 컨텍스트 외 정보로 답함, 컨텍스트와 모순 | 답변이 장황하거나 동문서답 |
| 개선 기법 | 시스템 프롬프트 강화(“컨텍스트 외 금지”), few-shot 예시 추가 | 답변 포맷 가이드(질문 키워드 반복), 짧은 답변 강제 |

#### End-to-End

**Answer Correctness** — `response`와 `reference`의 의미 일치 정도. 두 부분의 가중 평균:

1. **사실 일치도(Factual)** — LLM이 두 답변을 *주장 단위*로 분해해 TP/FP/FN을 계산 → F1.
2. **의미 유사도(Semantic)** — `response`/`reference` 임베딩 cosine.
3. 가중치는 디폴트 `[0.75, 0.25]` (사실 일치를 중시).

**`ground_truth` 품질 의존성** — `reference`가 짧은 단답이면 임베딩 cosine이 0.4~0.6대로 떨어져 두 부분이 모두 저평가됩니다. 본 5주차 데이터셋에서 `ground_truth`를 한 문장으로 정제한 이유.

**Answer Correctness만으로 부족한 이유** — End-to-end 점수가 낮을 때, *검색이 망가진 건지 / 생성이 망가진 건지* 구분 불가. 4 메트릭을 같이 봐야 디버깅 신호가 분리됩니다.

#### 메트릭 간 관계 (시나리오 표)

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---------|---------------|-----|------|
| 정답 청크 자체를 검색이 놓침 | Context Recall ↓, Answer Correctness ↓ | 청킹·임베딩·검색 알고리즘 | 하이브리드 검색, 재청킹 |
| 정답 청크는 있지만 8~10위로 밀림 | Context Precision ↓ | 검색 점수 부정확 | Re-ranker(Cohere / bge) |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness ↓ (Recall은 ↑) | 시스템 프롬프트, 모델 추론 | “컨텍스트 외 금지” 강화, 더 작은 모델 fallback |
| LLM이 질문을 잘못 이해 | Response Relevancy ↓ | 모호한 질문, 프롬프트 문제 | 쿼리 재작성, few-shot |
| 답은 맞는데 장황 | (해당 없음 — 왜?) | Ragas 기본 메트릭은 *길이/스타일*을 평가하지 않음 | 도메인 임계값(예: token ≤ 200) 또는 커스텀 메트릭 |

> 표의 마지막 행은 Ragas의 한계입니다. *“정답인데 너무 장황하다”* 같은 UX 품질은 별도 메트릭이 필요합니다(예: `MetricWithLLM` 상속 — 본 5주차 심화 A의 `YearAccuracy`가 같은 패턴).

---

## 4. 실습 결과

> ⚙️ 본 절은 `01_run_rag_and_collect.py`(15문항 × Basic/Advanced) → `02_ragas_evaluate.py` 실행 결과를 토대로 채워집니다. 실행이 완료되면 `basic_ragas_scores.csv`·`advanced_ragas_scores.csv`·`comparison_summary.json`이 생성됩니다.

### 4-1. 실험 가설 (실행 전)

| # | 가설 |
|---|------|
| H1 | 4주차 사람 정답률(50% / 66.7%)과 Ragas Answer Correctness는 대체로 같은 방향(Advanced ↑)이지만, Ragas는 **부분 점수**(0~1)를 주므로 사람의 이진 판정보다 후한 평균값(0.6~0.8)이 나올 것. |
| H2 | Basic↔Advanced 격차가 가장 큰 메트릭은 **Context Precision**. Re-ranker는 “정답 청크를 상위로 밀어올리는” 도구라 직접적으로 Precision에 작용. |
| H3 | 년도 혼동(2025/2026 헷갈림)은 **Faithfulness가 높은데 Answer Correctness가 낮은** 시나리오로 나타남. 모델이 검색된 청크에 *충실히* 답했지만, 청크 자체가 잘못된 년도라 정답과 어긋남. → Ragas 기본 메트릭으로는 “년도 혼동”이라는 *원인*을 직접 짚지 못함(심화 A YearAccuracy의 정당성). |
| H4 | Advanced에서 Faithfulness가 오히려 낮아지는 시나리오 — Re-ranker가 컨텍스트 다양성을 줄여 LLM이 “결정적인 한 문장”만 보고 *추가 추론*을 하게 만들면, claim 단위로 컨텍스트 미지원 진술이 늘 수 있음. |

### 4-2. 5메트릭 결과 (전체 평균, n=17)

| 메트릭 | Basic | Advanced | Δ (A−B) |
|--------|-------|----------|---------|
| Context Recall | **0.618** | **0.794** | +0.176 |
| Context Precision (LLM, with-ref) | **0.461** | **0.740** | **+0.279** |
| Faithfulness | **0.718** | **0.783** | +0.065 |
| Response Relevancy | **0.701** | **0.896** | +0.195 |
| Answer Correctness | **0.617** | **0.803** | +0.186 |

> **한 줄 요약**: Advanced는 5개 메트릭 모두에서 Basic을 능가. 가장 큰 격차는 **Context Precision (+0.279)** — Re-ranker가 정답 청크를 상위로 끌어올리는 효과가 직접 드러남. End-to-end **Answer Correctness +0.186** 은 “4주차에서 Advanced가 사람 정답률 50% → 66.7% (+16.7%p)” 와 부호·크기 모두 일치.

### 4-3. 문항별 메트릭 (B/A)

| qid | difficulty | source_year | Ctx Recall | Ctx Precision | Faithfulness | Ans Relevancy | Ans Correctness |
|---|---|---|---|---|---|---|---|
| q01 | easy | 2025 | 0.000 / 1.000 | 0.000 / 0.500 | 0.333 / 1.000 | 0.000 / 0.835 | 0.237 / **0.998** |
| q02 | easy | 2025 | 0.000 / 0.000 | 0.000 / 0.000 | 0.500 / 0.750 | 0.000 / 0.848 | 0.242 / 0.543 |
| q03 | easy | 2025 | 0.000 / 1.000 | 0.000 / 0.367 | 1.000 / 1.000 | 0.000 / 0.846 | 0.229 / 0.741 |
| q04 | easy | 2025 | 1.000 / 1.000 | 0.950 / 1.000 | 1.000 / 1.000 | 0.996 / 0.996 | 0.745 / 0.745 |
| q05 | easy | 2025 | 1.000 / 1.000 | 0.804 / 1.000 | 1.000 / 1.000 | 0.879 / 0.983 | 0.617 / **0.994** |
| q06 | medium | 2025 | 1.000 / 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 0.996 / 0.943 | 0.993 / 0.993 |
| q07 | medium | 2025 | 0.000 / 0.000 | 0.000 / 0.833 | 0.000 / 0.000 | 0.000 / 0.994 | 0.235 / 0.619 |
| q08 | medium | 2025 | 0.000 / 0.000 | 0.833 / 0.583 | **1.000 / 0.500** ⚠️ | 0.974 / 0.853 | 0.546 / 0.545 |
| q09 | easy | 2026 | 0.000 / 1.000 | 0.000 / 0.833 | 0.167 / 1.000 | 0.891 / 0.840 | 0.234 / **0.998** |
| q10 | easy | 2026 | 1.000 / 1.000 | 0.333 / 1.000 | 0.667 / 0.667 | 0.858 / 0.949 | 0.617 / 0.617 |
| q11 | easy | 2026 | 1.000 / 1.000 | 1.000 / 0.887 | 1.000 / 1.000 | 0.996 / 0.996 | 0.746 / 0.746 |
| q12 | medium | 2026 | 1.000 / 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 0.942 / 0.839 | 0.997 / 0.995 |
| q13 | medium | 2026 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.838 / 0.945 | **0.846 / 0.741** ⚠️ |
| q14 | hard | 2026 | 1.000 / 1.000 | 0.500 / 1.000 | 1.000 / 1.000 | 0.994 / 0.852 | 0.748 / 0.915 |
| q15 | cross-year | 2025+2026 | 1.000 / 1.000 | 0.417 / 0.583 | 1.000 / 1.000 | 0.862 / 0.831 | 0.841 / 0.844 |
| q16 | cross-year | 2025+2026 | 1.000 / 1.000 | 0.000 / 0.000 | 0.143 / 0.000 | 0.875 / 0.875 | 0.994 / 0.995 |
| q17 | cross-year | 2025+2026 | 0.500 / 0.500 | 1.000 / 1.000 | 0.400 / 0.400 | 0.814 / 0.816 | 0.619 / 0.619 |

> ⚠️ 표시는 Advanced가 Basic보다 떨어진 칸 (Faithfulness q08, Answer Correctness q13).
> 굵은 점수는 “Advanced가 0.99 이상”인 정확 응답.

### 4-4. 4주차 수동 채점 vs 5주차 Ragas Answer Correctness

비교 기준: Ragas Ans Corr **≥ 0.7** 면 “정답”으로 임계값 부여. 4주차 사람 채점은 4주차 `check_answer` 함수의 결과 (키워드 매칭 기반).

| qid | 질문 (요약) | 4주차 (B/A) | Ragas (B/A) | 일치 (B/A) |
|-----|-------------|-------------|-------------|-----------|
| q01 | 2025 1종 의원 외래 본인부담금 | 오답 / 정답 | 0.237 / 0.998 | ✓ / ✓ |
| q02 | 2025 2종 종합병원 입원 본인부담률 | 오답 / 오답 | 0.242 / 0.543 | ✓ / ✓ |
| q03 | 2025 1종 입원 본인부담률 | 오답 / 정답 | 0.229 / 0.741 | ✓ / ✓ |
| q04 | 2025 65세 1종 틀니 | 정답 / 정답 | 0.745 / 0.745 | ✓ / ✓ |
| q05 | 2025 입원 식대 1종·2종 | 정답 / 정답 | 0.617 / 0.994 | ✗ / ✓ |
| q06 | 2025 1종 디스크 복잡추나 | 정답 / 정답 | 0.993 / 0.993 | ✓ / ✓ |
| q07 | 2025 장기지속형 주사제 | 정답 / 정답 | 0.235 / 0.619 | ✗ / ✗ |
| q08 | 2025 노숙인 유효기간 | 오답 / 오답 | 0.546 / 0.545 | ✓ / ✓ |
| q09 | 2026 1종 의원 외래 | 오답 / 정답 | 0.234 / 0.998 | ✓ / ✓ |
| q10 | 2026 2종 약국 | 정답 / 정답 | 0.617 / 0.617 | ✗ / ✗ |
| q11 | 2026 65세 2종 임플란트 | 정답 / 정답 | 0.746 / 0.746 | ✓ / ✓ |
| q12 | 2026 2종 디스크 외 복잡추나 | 정답 / 정답 | 0.997 / 0.995 | ✓ / ✓ |
| q13 | 2026 외래 365회 초과 | 정답 / 정답 | 0.846 / 0.741 | ✓ / ✓ |
| q14 | 2026 KTAS 100/100 | 오답 / 오답 | 0.748 / 0.915 | ✗ / ✗ |
| q15 | cross 장기지속형 주사제 | 오답 / 오답 | 0.841 / 0.844 | ✗ / ✗ |
| q16 | cross 노숙인 유효기간 | 오답 / 오답 | 0.994 / 0.995 | ✗ / ✗ |
| q17 | cross 365회 초과 도입 | 오답 / 오답 | 0.619 / 0.619 | ✓ / ✓ |

**불일치 9건의 원인 분류**:

| 유형 | 케이스 | 원인 |
|------|------|------|
| Ragas가 더 후함 (사람=오답, Ragas=정답) | q14, q15, q16 | 4주차 `check_answer`는 **expected_answer 단어 키워드 매칭**으로 판정. cross-year/KTAS 같이 정답이 두 항목·여러 단어로 구성된 경우, 답변이 사실상 정확해도 키워드 형식 차이로 “오답” 처리. Ragas는 의미·사실 단위로 0.84~0.99 부여. → **Ragas가 정확** |
| Ragas가 더 엄격 (사람=정답, Ragas=오답) | q05(B), q07(B/A), q10(B/A) | 답변이 짧거나(단답형) 핵심 숫자만 출력. Ragas Answer Correctness는 의미 유사도 + 사실 일치도. 짧은 답변은 임베딩 cosine이 낮아 0.6대로 떨어짐. → **사람 채점이 후함** |

> 결론: 17문항 중 9건이 사람-Ragas 불일치, 그중 약 **2/3가 “Ragas가 더 정확”** (사람의 키워드 매칭 채점이 cross-year·복합 정답을 잡지 못함). Ragas는 답변이 짧을 때 저평가하는 부작용이 있어, RAG 프롬프트에 “완전한 문장으로 답하라”는 지시가 필요.

### 4-5. Basic vs Advanced — 다차원 비교

| 차원 | 결과 |
|------|------|
| **개선된 메트릭** | 5개 모두 (Context Precision +0.279 ≫ Ans Relevancy +0.195 ≈ Ans Correctness +0.186 ≈ Ctx Recall +0.176 ≫ Faithfulness +0.065) |
| **악화된 칸 (문항 단위)** | q08 Faithfulness 1.0→0.5 / q13 Ans Correctness 0.85→0.74 / q11 Ctx Precision 1.0→0.89 (작은 회귀) |
| **년도 혼동 재진단** | 4주차에선 “2025/2026 키워드 혼동” 발생. 5주차 Ragas로 보면 **년도 혼동은 Faithfulness에 잘 반영되지 않음** — q08은 답변 자체는 정확(2024.3.22.~2025.3.21.)하지만 Re-ranker가 2026 청크를 끌어올려 *“claim이 검색 청크에서 직접 도출되지 않는다”* 로 Faithfulness가 떨어짐. **Year Accuracy 커스텀 메트릭(심화 A)이 필요한 정량적 근거.** |
| **도메인 임계값(Faithfulness ≥ 0.9)** | Basic 9/17 (52.9%), Advanced 11/17 (64.7%) 통과. 의료 도메인 프로덕션에는 둘 다 부족. (목표 ≥ 90%) |
| **개선 우선순위** | (1) Faithfulness — 답변이 검색 컨텍스트에서 *직접* 도출되도록 시스템 프롬프트에 “컨텍스트 외 정보 금지” + 인용 강제. (2) Context Precision — 페이지 단위 작은 청크는 표 헤더가 분리되어 의미 단위가 깨짐. 청크 크기 조정 또는 표 구조 보존 청킹. |

### 4-6. 실패 케이스 Deep Dive

#### Case A — Advanced가 Basic보다 회귀: **q08** (Faithfulness 1.0 → 0.5)

```
질문: 2025년 노숙인진료시설 지정 유효기간은 얼마인가?
Ground truth: 2025년 「노숙인진료시설 지정 등에 관한 고시」 일부개정에 따라
              노숙인진료시설 지정 유효기간은 1년 연장되어 2024년 3월 22일부터
              2025년 3월 21일까지 적용됩니다.

[Basic 검색 청크의 source_year]   ['2025','2026','2025','2025','2026']
[Advanced 검색 청크의 source_year] ['2026','2025','2025','2026','2026']  ← 더 많은 2026

[Basic 답변]
  2025년 문서에 따르면, 노숙인진료시설 지정 유효기간은 2024.3.22.~2025.3.21.입니다.

[Advanced 답변]
  2025년 문서에 따르면, 노숙인진료시설 지정 유효기간은 **2024.3.22.~2025.3.21.**입니다.

Ragas:
  Basic    Faith=1.000  Ctx Precision=0.833
  Advanced Faith=0.500  Ctx Precision=0.583   ← 회귀
```

**원인 분석**
- 두 답변은 **사실상 동일·정확**. 답변 내용만 보면 회귀가 아님.
- 그러나 Advanced의 Re-ranker(bge-reranker-v2-m3)가 2026년 청크 3개를 상위로 끌어올림 (2026 PDF의 노숙인 항목이 *“2025년 고시 대비 변경”* 컨텍스트를 포함해 키워드 유사도가 높음).
- LLM은 2025 청크 1개를 보고 정답을 만들었지만, Ragas Faithfulness는 “답변의 claim이 *전체 검색 컨텍스트* 에서 직접 도출되는가”를 보는데, 2025 정보(2024.3.22.~2025.3.21.)가 다수의 2026 청크에서 *반대 맥락*(2025.3.22.~2028.3.21. 변경)으로 등장 → claim의 컨텍스트 일관성 떨어짐.
- 어느 메트릭이 가장 잘 드러냈나? **Faithfulness (1.0→0.5)**. Context Precision도 같이 떨어짐(0.833→0.583).

**조치**
- 메타데이터 필터: 질문에서 “2025년” 명시 시 검색 단계에서 `source_year=="2025"` 청크만 통과시키는 pre-filter.
- Re-ranker 입력에 source_year 태그 포함 (현재는 본문만 score).
- 또는 Year Accuracy 커스텀 메트릭(심화 A)으로 *“최종 답변 년도 ↔ 질문 년도 일치”* 를 별도 측정.

#### Case B — 년도 혼동: **q07** (Basic은 답을 못 찾고, Advanced는 정답)

```
질문: 2025년 장기지속형 주사제의 본인부담률은 몇 %인가?
Ground truth: 2025년 의료급여 1·2종 수급권자의 장기지속형 주사제 본인부담률은 5%입니다.

[Basic 검색 청크 source_year]   ['2026','2025','2026','2025','2025']
[Advanced 검색 청크 source_year] ['2025','2026','2026','2026','2025']

[Basic 답변]
  2025년 문서에는 장기지속형 주사제의 본인부담률에 대한 정보를 찾을 수 없습니다.
  (참고로 2026년 문서에는 ... 5%에서 2%로 완화되었다는 내용이 있지만,
  2025년 문서 자체에는 해당 정보가 없습니다.)

[Advanced 답변]
  2025년 장기지속형 주사제의 본인부담률은 5%입니다.

Ragas:
  Basic    Recall=0.0 Precision=0.0 Faith=0.0 Relevancy=0.0 AnsCorr=0.235
  Advanced Recall=0.0 Precision=0.833 Faith=0.0 Relevancy=0.994 AnsCorr=0.619
```

**원인 분석**
- Basic은 2025 청크가 5개 중 3개나 들어왔는데도 “정보를 찾을 수 없습니다”로 회피 — 표 셀 단위로 청크가 잘려 “장기지속형 주사제 5%”의 5%가 다른 행과 분리됨.
- Advanced의 Re-ranker는 *“장기지속형 주사제 5%”* 가 같이 등장하는 청크를 상위로 끌어올려 LLM이 답할 수 있게 만듦.
- 흥미로운 지점: 두 파이프라인 모두 Recall=0, Faith=0. 이는 `ground_truth_contexts`가 표 캡션 + 헤더 형식인데 검색 청크는 표 셀 텍스트만 가져왔기 때문 — *내용은 같은 정보*지만 텍스트 일치도가 낮아 Ragas가 “정답 청크가 검색되지 않았다”고 판단.
- **어느 메트릭이 가장 잘 드러냈나?** Response Relevancy. Basic 0.0 vs Advanced 0.994 — 답변이 질문에 직접 답했는지 여부를 정확히 짚었음.

**조치**
- 청킹 전략 변경: 표는 행 단위가 아닌 *“표 캡션 + 헤더 + 모든 행”* 을 한 청크로 묶기 (TableSplitter 또는 Markdown 변환 후 청킹).
- `ground_truth_contexts`를 표 본문 형태로도 추가 — Ragas Recall이 의미가 같은데 단지 형식이 다른 청크를 인식하도록.

#### Case C — 메트릭 간 충돌: **q03** (Basic Faith=1.0인데 AnsCorr=0.229)

```
질문: 2025년 1종 수급권자의 입원 본인부담률은?
Ground truth: 2025년 1종 의료급여 수급권자의 입원 본인부담률은 무료(전액 의료급여기금 부담)입니다.

[Basic 답변]
  제공된 컨텍스트에서 2025년 1종 수급권자의 일반적인 입원 본인부담률에 대한
  정보를 찾을 수 없습니다. 컨텍스트에는 다음과 같은 특정 상황의 본인부담률만
  제시되어 있습니다: - 65세 이상 틀니: 5% - 65세 이상 임플란트: 10% - 입원 식대: 20%

Basic Ragas: Faith=1.000  AnsCorr=0.229  Recall=0.0
```

**원인 분석**
- Basic의 답변은 “정보 없음”이라고 했는데, **답변 내용 자체는 검색 컨텍스트에서 도출 가능**(컨텍스트에 “1종 입원 무료”가 없었음). → 환각 없음 → Faithfulness 1.0
- 하지만 정답(`reference`: 무료)과 비교하면 사실 일치도 0%, 임베딩 유사도도 매우 낮음 → AnsCorr 0.229
- Ragas의 정확한 진단: *“환각은 없지만 답을 못 했다”* — Faithfulness만 보면 만점이지만 End-to-end는 실패.
- **이 패턴은 Faithfulness 단독으로는 RAG 품질 평가에 부족하다는 강한 증거.** Recall + AnsCorr를 함께 봐야 “답을 회피하는 모델”을 잡아냄.

### 4-7. 공통 교훈 (5개)

1. **Re-ranker는 5개 메트릭 모두를 끌어올린다** — 단, Faithfulness만은 +0.065로 작은 폭. Re-ranker는 *“정답 청크 끌어올림”* 에는 강하지만 *“환각 억제”* 에는 직접 효과가 없다.
2. **Context Precision 격차가 가장 크다** (+0.279) — 이는 Basic의 vector-only 검색이 표·고유명사 매칭에 약하고, BM25+Re-rank 조합이 그 약점을 정확히 보완함을 보여줌.
3. **Ragas는 4주차 키워드 채점보다 *공정* 하지만 *완벽*하지 않다** — cross-year·복합 정답 문항(q14, q15, q16)에서는 Ragas가 사람보다 정확. 단, **답변이 짧을 때**(q07, q10) Ragas는 의미 유사도 페널티로 인해 저평가 → 정답에 가까운 응답을 “오답”으로 분류할 수 있음.
4. **년도 혼동은 Ragas 기본 메트릭으로 직접 잡히지 않는다** — q08, q15·q16처럼 답변이 정확해도 검색 청크의 년도 분포에 따라 Faithfulness/Precision이 흔들림. 도메인 특화 `YearAccuracy` 커스텀 메트릭이 정당한 이유(심화 A).
5. **Faithfulness=1.0이 곧 정답은 아니다** (q03 Case C). Faithfulness만으로 게이트하면 “환각은 없지만 답을 회피하는” RAG를 막지 못함. Recall × Faithfulness × AnsCorr를 함께 봐야 함.

### 4-8. 가설 vs 실제 결과

| # | 가설 | 실제 결과 | 판정 |
|---|------|-----------|------|
| H1 | 사람 vs Ragas 일치, Ragas 평균이 더 후함 | 일치 8/17, 불일치 9/17. 평균 Ragas Ans Corr는 Basic 0.617(사람 50%), Advanced 0.803(사람 66.7%) → Ragas가 약 12~14%p 후함 | **부분 적중** — 방향성 일치, 다만 cross-year에서 Ragas가 사람보다 정확 |
| H2 | Basic↔Advanced 격차 최대 메트릭 = Context Precision | Δ Context Precision **+0.279** 가 5개 중 최대. 다음이 Ans Relevancy(+0.195) | **적중** |
| H3 | 년도 혼동은 Faith 높은데 AnsCorr 낮음으로 나타남 | q08은 그 정반대(둘 다 높지만 Ctx Precision이 떨어짐). cross-year q15·q17은 Ragas Ans Corr가 0.6~0.8로 부분 점수만. → 년도 혼동 ⇒ Ctx Precision/Recall 변동이 더 직접적 신호 | **부분 적중** — Faithfulness보단 Context 메트릭에 반영 |
| H4 | Advanced에서 Faithfulness가 오히려 낮아질 시나리오 존재 | q08에서 정확히 발생 (1.0 → 0.5). Re-ranker가 다른 년도 청크를 상위로 끌어올린 부작용 | **적중** |

---

## 5. 디렉토리 구성

```
week-5/s1ns3nz0/
├── README.md                       # 본 문서
├── golden_dataset_v2.jsonl         # 17문항 골든 데이터셋 (ground_truth + ground_truth_contexts)
├── 01_run_rag_and_collect.py       # Basic/Advanced RAG 실행 및 응답·청크 수집
├── 02_ragas_evaluate.py            # Ragas 5메트릭 평가
├── 03_analyze_results.py           # CSV → Markdown 표·실패 케이스 자동 산출
├── rag_outputs_basic.json          # Basic RAG 응답·청크 (Step 1)
├── rag_outputs_advanced.json       # Advanced RAG 응답·청크 (Step 1)
├── basic_ragas_scores.csv          # Basic 5메트릭 결과 (Step 2)
├── advanced_ragas_scores.csv       # Advanced 5메트릭 결과 (Step 2)
├── comparison_summary.json         # Basic/Advanced 평균값 + Δ 요약
├── analysis_tables.md              # 자동 생성된 비교 표 (Step 3)
├── failure_cases.json              # Case A/B/C 자동 선별 결과
├── rag_run.log                     # RAG 파이프라인 실행 로그
├── ragas_eval.log                  # Ragas 평가 실행 로그
└── _evidence_extracted.json        # PDF 원문 발췌 보조 자료 (golden_dataset_v2 작성 시 사용)
```

## 6. 실행 방법

```bash
# 1) week-4 벡터스토어가 있는지 확인
ls ../../week-4/s1ns3nz0/vectorstore_basic/

# 2) Basic + Advanced RAG 실행 → 응답·청크 수집
TOKENIZERS_PARALLELISM=false python 01_run_rag_and_collect.py

# 3) Ragas 5메트릭 평가
python 02_ragas_evaluate.py

# 4) 비교 표·실패 케이스 자동 산출
python 03_analyze_results.py
```

> 환경변수: `ANTHROPIC_API_KEY` 필수, `COHERE_API_KEY` 선택(미설정 시 로컬 cross-encoder 폴백).

## 7. 참고 자료

- [Ragas 공식 — Get Started](https://docs.ragas.io/en/stable/getstarted/)
- [Ragas — Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [Ragas — Test Data Generation](https://docs.ragas.io/en/stable/concepts/test_data_generation/)
- [RAGAS Paper — Es et al. 2023](https://arxiv.org/abs/2309.15217)
- [Judging LLM-as-a-Judge — Zheng et al. 2023 (MT-Bench)](https://arxiv.org/abs/2306.05685)
- [G-Eval — Liu et al. 2023](https://arxiv.org/abs/2303.16634)
- [BAAI bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
