# 5주차 과제: RAG 평가 - Golden Dataset, LLM-as-a-Judge, Ragas

## 실행 환경

OpenAI Platform API는 quota 제한으로 사용할 수 없었고, Google Generative Language API의 OpenAI-compatible endpoint를 사용해 실제 Ragas LLM-as-a-Judge 평가를 실행했다. `evaluate_rag.py`로 Basic/Advanced 검색 결과와 응답을 먼저 생성하고, `evaluate_ragas_google.py`가 그 결과를 Ragas `SingleTurnSample`로 매핑해 5개 메트릭을 평가한다.

| 항목 | 값 |
|---|---|
| Python | 3.10.12 |
| PDF 로더 | PyMuPDF 1.27.2.2 |
| 데이터 처리 | pandas 2.3.3 |
| 검색 | scikit-learn 1.7.2 TF-IDF char n-gram |
| Ragas | 0.4.3 |
| 생성용 LLM | 오프라인 추출형 응답 시뮬레이터 |
| 평가용 LLM | Google `gemini-2.5-flash-lite` via OpenAI-compatible endpoint |
| 임베딩 | Google `gemini-embedding-001` |

## 제출 파일

| 파일 | 설명 |
|---|---|
| `golden_dataset_v2.jsonl` | 15문항 Golden Dataset, `ground_truth`, `ground_truth_contexts` 포함 |
| `evaluate_rag.py` | PDF 청킹, Basic/Advanced 검색, 응답 생성, 5개 지표 산출 스크립트 |
| `evaluate_ragas_google.py` | Google API 기반 실제 Ragas 평가 스크립트 |
| `basic_ragas_scores.csv` | Basic RAG 문항별 평가 결과 |
| `advanced_ragas_scores.csv` | Advanced RAG 문항별 평가 결과 |
| `basic_ragas_google_scores.csv` | Basic RAG 실제 Ragas 평가 결과 |
| `advanced_ragas_google_scores.csv` | Advanced RAG 실제 Ragas 평가 결과 |
| `ragas_summary.csv` | Basic/Advanced 평균 비교 |
| `ragas_google_summary.csv` | Google API 기반 실제 Ragas 평균 비교 |
| `manual_vs_ragas.csv` | 수동 판정형 정답 여부와 Answer Correctness 비교 |
| `evaluation_summary.json` | 실행 요약 로그 |
| `ragas_google_evaluation_summary.json` | 실제 Ragas 실행 요약 로그 |
| `env.example` | Google/OpenAI API 환경변수 예시 |

실행 명령:

```bash
python3 week-5/SeungHyeog/evaluate_rag.py
python3 week-5/SeungHyeog/evaluate_ragas_google.py --chat-model gemini-2.5-flash-lite --max-tokens 8192 --batch-size 1 --top-contexts 3 --max-context-chars 700
```

## Golden Dataset 전략

`ground_truth`는 `년도 + 대상 + 조건 + 값` 순서의 한 문장으로 정제했다. 예를 들어 단순히 `2%`라고 쓰지 않고 `2026년 정신질환 외래진료에서 항정신병 장기지속형 주사제의 본인부담률은 2%입니다.`처럼 답변과 같은 완전한 문장으로 작성했다.

`ground_truth_contexts`는 PDF 원문에서 2~5문장 또는 표 1개 단위로 직접 발췌했다. 문서 청크와 완전히 같은 문자열일 필요는 없지만, Ragas의 Context Recall과 Context Precision이 정답 근거를 판단할 수 있도록 수치, 대상, 조건을 빠뜨리지 않았다.

Dataset 구성은 2025 단일년도 6문항, 2026 단일년도 7문항, cross-year 2문항이다. 난이도는 easy 4문항, medium 6문항, hard 3문항, cross-year 2문항으로 분산했다.

Cross-year 문항은 두 년도의 근거를 모두 `ground_truth_contexts`에 넣었다. 예를 들어 장기지속형 주사제 본인부담률 변화 문항은 2025년 5% 근거와 2026년 2% 완화 근거를 모두 포함했다.

## 평가 파이프라인

Basic RAG는 두 PDF를 페이지 단위로 읽고 700단어 내외로 청킹한 뒤, `source_year` 메타데이터는 보존하지만 검색에는 사용하지 않는다. 검색은 문자 n-gram TF-IDF 유사도 Top-5만 사용한다.

Advanced RAG는 질문에서 `2025`, `2026`을 추출해 메타데이터 필터를 적용하고, TF-IDF 점수와 직접 구현한 BM25 점수를 결합한 뒤 질문 토큰, 수치, 년도 일치 여부로 재순위화한다. 실제 4주차 요구사항의 Hybrid Search와 Re-ranking을 로컬에서 재현한 경량 버전이다.

실제 Ragas 평가는 Google API를 LangChain `ChatOpenAI` / `OpenAIEmbeddings`의 `base_url`로 연결해 실행했다. `gemini-2.5-flash`도 동작했지만 장시간 평가 중 503과 출력 길이 제한이 있어, 전체 평가는 `gemini-2.5-flash-lite`, `max_tokens=8192`, Top-3 컨텍스트로 안정화했다.

오프라인 평가지표는 Ragas 실행 전 검색/응답 파이프라인을 빠르게 검증하기 위한 보조 결과다. `context_recall`은 정답 근거가 검색 컨텍스트에 들어왔는지, `context_precision`은 관련 근거가 상위에 있는지, `faithfulness`는 응답의 수치·핵심 사실이 검색 컨텍스트에 의해 지지되는지, `answer_relevancy`는 질문과 응답의 관련성, `answer_correctness`는 응답과 `ground_truth`의 사실·토큰 일치를 측정한다.

실제 Ragas로 전환할 때는 다음 클래스형 메트릭을 사용한다: `ContextRecall`, `LLMContextPrecisionWithReference`, `Faithfulness`, `ResponseRelevancy`, `AnswerCorrectness`.

## Step 2 결과

| Metric | Basic | Advanced | 변화 |
|---|---:|---:|---:|
| Context Recall | 0.6000 | 0.8667 | +0.2667 |
| Context Precision | 0.5778 | 0.7000 | +0.1222 |
| Faithfulness | 0.6311 | 0.8511 | +0.2200 |
| Answer Relevancy | 0.7698 | 0.9049 | +0.1351 |
| Answer Correctness | 0.8887 | 1.0000 | +0.1113 |

| 질문 ID | 난이도 | source_year | Ctx Recall B/A | Ctx Precision B/A | Faithfulness B/A | Ans Relevancy B/A | Ans Correctness B/A |
|---|---|---|---|---|---|---|---|
| q01 | easy | 2025 | 0.00 / 1.00 | 0.00 / 1.00 | 0.50 / 1.00 | 0.00 / 0.95 | 0.16 / 1.00 |
| q02 | easy | 2026 | 0.00 / 1.00 | 0.00 / 0.33 | 0.50 / 1.00 | 0.00 / 0.99 | 0.17 / 1.00 |
| q06 | medium | 2026 | 1.00 / 1.00 | 0.33 / 0.50 | 0.00 / 1.00 | 0.92 / 0.96 | 1.00 / 1.00 |
| q07 | cross-year | 2025+2026 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.50 | 0.77 / 0.77 | 1.00 / 1.00 |
| q08 | hard | 2026 | 1.00 / 1.00 | 0.33 / 1.00 | 1.00 / 1.00 | 0.79 / 0.79 | 1.00 / 1.00 |
| q12 | hard | 2025 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.60 | 1.00 / 0.99 | 1.00 / 1.00 |
| q15 | cross-year | 2025+2026 | 1.00 / 1.00 | 0.50 / 0.50 | 1.00 / 1.00 | 0.84 / 0.89 | 1.00 / 1.00 |

위 표는 `ragas_google_summary.csv`와 `*_ragas_google_scores.csv` 기준 실제 Ragas 결과다. 오프라인 보조 평가는 `ragas_summary.csv`에 별도 저장했다.

## 4주차 수동 판정 vs Answer Correctness

4주차 개인 산출물이 이 저장소에 없어서 기존 수동 채점표는 재사용하지 못했다. 따라서 이번 스크립트의 `manual_correct`를 4주차식 핵심값 포함 판정으로 두고 Answer Correctness와 비교했다.

| 방식 | 정답률 | 년도 검색 정확도 |
|---|---:|---:|
| Basic RAG | 13 / 15 | 8 / 15 |
| Advanced RAG | 15 / 15 | 15 / 15 |

Basic에서 q01, q02는 Answer Correctness가 각각 0.00, 0.1062로 낮고 수동 판정도 오답이다. Advanced는 두 문항 모두 올바른 년도 필터와 근거 페이지 검색으로 Answer Correctness 1.00이 되었다.

q04, q06, q08, q12, q13은 Basic Answer Correctness가 1.00이지만 `year_correct=False`이다. 이는 같은 값이 두 해 문서에 반복되는 의료급여 문서 특성 때문에 답 자체는 맞지만 검색 상위 근거의 년도는 틀린 경우다. Ragas 기본 메트릭만으로는 이런 년도 혼동을 충분히 벌점화하지 못할 수 있어 YearAccuracy 같은 커스텀 메트릭이 필요하다.

## Step 3 분석

| 구분 | 결과 |
|---|---|
| 개선 | Advanced는 Context Recall, Context Precision, Faithfulness, Answer Relevancy, Answer Correctness가 모두 상승했다. |
| 악화 | 평균 기준 악화 없음. q12는 Context Precision이 1.00에서 0.89로 소폭 낮아졌지만 정답과 년도 정확도는 유지됐다. |
| 가장 큰 개선 | Context Recall +0.2667. 년도 필터와 Hybrid 검색이 정답 근거 자체를 검색 결과에 포함시키는 데 가장 크게 기여했다. |
| 년도 혼동 | Basic은 15문항 중 7문항에서 Top-1 년도가 틀렸다. Advanced는 15문항 모두 올바른 년도 근거를 Top-1 또는 cross-year Top-5에 포함했다. |

4주차의 “Advanced가 낫다”는 결론은 검색 정확도뿐 아니라 생성 지표에서도 유효했다. 다만 q04, q06처럼 답이 맞아도 검색 년도가 틀린 경우가 있어 `Answer Correctness`만으로는 년도 인식 품질을 평가하기 어렵다.

프로덕션 기준을 `Faithfulness >= 0.9`, `Answer Correctness >= 0.9`, `YearAccuracy = 1.0`으로 두면 Advanced도 q07, q12의 Faithfulness가 기준에 못 미친다. Cross-year 문항은 두 년도 근거가 모두 상위에 있어야 하고, 생성 응답의 모든 수치가 각 년도 근거로 지지되어야 하므로 별도 프롬프트와 근거 인용 포맷이 필요하다.

개선 우선순위는 년도 인식 메트릭 추가, cross-year 검색에서 년도별 최소 1개 청크 보장, 답변에 근거 년도와 페이지를 함께 출력하도록 프롬프트 강화 순서다.

## Step 4 실패 케이스

### Case A 대체: q01 Basic 실패, Advanced 개선

질문: 2025년 의료급여 1종 수급권자의 외래 본인부담금은 의료급여기관별로 얼마인가요?

참고 정답: 2025년 의료급여 1종 수급권자의 외래 본인부담금은 1차 1,000원, 2차 1,500원, 3차 2,000원, 약국 500원이며 CT, MRI, PET 등은 5%입니다.

Basic 검색은 2026년 목차와 2025년 표지, 2025년 틀니/임플란트 FAQ가 상위에 섞였고 정작 2025년 본인부담금 표 페이지가 누락됐다. Context Recall과 Precision이 모두 0.00이어서 검색 실패가 원인이다.

Advanced 검색은 `source_year=2025` 필터와 Hybrid 점수 결합으로 2025년 9쪽 본인일부부담금 표를 Top-1로 올렸다. Context Recall 1.00, Faithfulness 1.00, Answer Correctness 1.00으로 회복됐다.

조치: 년도 필터를 기본 적용하고, 본인부담금처럼 표 기반 질문은 `1종`, `외래`, `1차`, `2차`, `3차`, `약국` 같은 구조화 키워드가 BM25에 반영되도록 해야 한다.

### Case B: q08 년도 혼동

질문: 2026년 의료급여 외래진료를 연간 365회 초과 이용하면 본인부담률은 어떻게 적용되나요?

참고 정답: 2026년부터 의료급여 외래진료를 연간 365회 초과 이용하면 본인부담률 30%가 적용됩니다.

Basic은 Answer Correctness가 1.00이지만 `year_correct=False`였다. Top-1이 2025년의 다른 본인부담 FAQ였고, 2026년 변경제도 페이지는 뒤쪽에 있었다. 답은 맞았지만 상위 근거의 년도는 틀린 전형적인 년도 혼동이다.

Advanced는 2026년 14쪽 변경제도 페이지를 Top-1로 가져와 Context Precision 1.00, Faithfulness 1.00이 되었다. 이 케이스는 Answer Correctness보다 Context Precision과 별도 YearAccuracy가 더 민감하게 실패를 보여준다.

조치: 질문에 년도가 있으면 필터를 강제하고, 년도 없는 질문은 답변에 “자료 기준 년도”를 명시하게 한다.

### Case C: q07 메트릭 충돌

질문: 2025년 대비 2026년에 항정신병 장기지속형 주사제 본인부담률은 어떻게 달라졌나요?

참고 정답: 항정신병 장기지속형 주사제 본인부담률은 2025년 5%에서 2026년 2%로 낮아졌습니다.

Answer Correctness는 Basic과 Advanced 모두 1.00이지만 실제 Ragas Context Recall과 Context Precision은 둘 다 0.00이었다. Advanced Faithfulness는 0.50까지 올랐지만, Ragas 입력을 Top-3 컨텍스트로 제한하면서 2026년 장기지속형 주사제 근거가 빠지고 틀니/임플란트 FAQ가 상위에 남았다. 응답은 정답과 일치하지만 검색 컨텍스트가 두 년도의 핵심 근거를 모두 충분히 제공하지 못한 메트릭 충돌이다.

조치: cross-year 문항에서는 년도별 근거 청크를 각각 최소 1개씩 고정하고, 답변에 `2025 근거`, `2026 근거`를 분리해 출력하도록 프롬프트를 바꾼다.

## 공통 교훈

- 의료급여 문서는 매년 구조가 거의 같아서 벡터 유사도만 쓰면 목차, FAQ, 이전년도 동일 표현이 상위에 섞인다.
- Ragas 기본 메트릭은 “답이 맞는지”와 “근거가 있는지”를 잘 보지만, “근거 년도가 맞는지”는 별도 메트릭 없이는 놓칠 수 있다.
- Context Recall이 높아도 Context Precision이 낮으면 LLM이 맞는 청크와 방해 청크를 함께 보게 되어 Faithfulness가 낮아질 수 있다.
- Cross-year 문항은 단일 Top-K보다 년도별 Top-K를 보장하는 검색 정책이 더 안정적이다.
- 수동 채점은 핵심값 포함 여부에 관대하고, Ragas 계열 평가는 근거 지지와 문맥 순서에 더 민감하다.

## 실습 전 가설 vs 실제

| 가설 | 실제 결과 |
|---|---|
| 4주차 수동 정답률과 Answer Correctness는 대체로 일치하지만 근거 년도 오류는 불일치할 것이다. | 일치했다. q04, q06, q08, q12, q13은 답은 맞아도 `year_correct=False`라서 Answer Correctness만으로는 부족했다. |
| Basic/Advanced 차이는 Context Precision에서 가장 클 것이다. | 가장 큰 평균 차이는 Context Recall +0.2667이었다. Context Precision도 +0.1222 상승했다. |
| 년도 혼동은 Context Precision과 별도 YearAccuracy에 가장 잘 드러날 것이다. | 맞았다. Basic 년도 정확도는 8/15였고 Advanced는 15/15였다. |
| Advanced에서 Faithfulness가 오히려 낮아질 수 있다. | 평균 악화는 없었다. 다만 q12는 Advanced Faithfulness가 0.60으로 Basic보다 낮아졌고, q07 cross-year 문항도 0.50이라 추가 개선 여지가 있었다. |

## 이론 과제

### Golden Dataset

Golden Dataset은 RAG 시스템이 반드시 맞혀야 하는 대표 질문, 기준 답변, 정답 근거를 사람이 검증해 모아둔 평가 기준선이다. 없으면 프롬프트, 모델, 청킹, 검색기를 바꿨을 때 좋아졌는지 나빠졌는지 객관적으로 비교할 수 없고, 회귀가 발생해도 발견하기 어렵다.

Ragas v0.1 계열 예시는 `question`, `answer`, `contexts`, `ground_truth` 필드를 많이 사용했다. v0.2+의 `SingleTurnSample` 기준으로는 `user_input`, `response`, `retrieved_contexts`, `reference`, `reference_contexts`를 쓰는 방식이 권장된다. 이번 과제 파일의 `question`, `ground_truth`, `ground_truth_contexts`는 평가 직전에 각각 `user_input`, `reference`, `reference_contexts`로 매핑한다.

초기 단계는 10~30문항으로 스키마와 평가 비용을 먼저 검증하고, 팀 내 회귀 테스트는 50~100문항, 운영 전 품질 관리는 200문항 이상으로 확장하는 것이 현실적이다. 규모보다 중요한 것은 실제 사용자 질문, 장애가 났던 회귀 케이스, 쉬운 문항과 함정 문항의 균형이다.

`ground_truth_contexts`는 사람이 직접 어노테이션해야 한다. Context Recall은 “정답에 필요한 근거가 검색되었는가”를 재는 메트릭이므로 기준 근거가 잘못되면 검색기를 잘못 평가한다. 특히 RAG의 검색 실패와 생성 실패를 분리하려면 답만이 아니라 답을 뒷받침하는 원문 근거가 필요하다.

출처: Ragas Testset Generation, Ragas Metrics Overview, RAGAS 논문(Es et al., 2023).

### 체계적 평가와 LLM-as-a-Judge

전통 소프트웨어 테스트는 결정적 입력과 출력이 있는 함수에 강하다. 하지만 LLM/RAG는 같은 질문에도 모델, temperature, 검색 결과, 프롬프트 변경에 따라 표현과 사실 선택이 달라진다. 따라서 단순 unit test보다 Golden Dataset 기반 회귀 평가와 품질 임계값이 필요하다.

LLM-as-a-Judge는 루브릭, 채점 기준, 예시를 프롬프트로 제공하고 LLM이 답변을 읽어 점수와 근거를 내는 방식이다. BLEU/ROUGE 같은 n-gram 지표가 의미적 정답을 놓치고, 사람 평가가 느리고 비싸다는 한계 때문에 등장했다. MT-Bench는 GPT-4 Judge가 기존 자동 지표보다 사람 선호 평가와 더 높은 일치도를 보일 수 있음을 보고했지만, 평가 프롬프트와 모델에 따라 결과가 달라지므로 사람 검수 샘플은 계속 필요하다.

좋은 루브릭 예시는 “답변의 모든 수치가 제공된 컨텍스트에 명시되어 있으면 1점, 일부 수치만 지지되면 0.5점, 컨텍스트와 모순되거나 근거가 없으면 0점”처럼 판정 가능한 기준을 둔다. 나쁜 루브릭은 “좋은 답변이면 높은 점수”처럼 모호해서 모델마다 판단이 흔들린다.

한계도 있다. Judge 모델은 프롬프트와 모델 버전에 민감하고, 비결정적이며, 긴 문맥 평가 비용이 크다. 그래서 평가용 LLM은 생성용 LLM과 다른 모델 패밀리를 쓰고, temperature를 0으로 낮추며, 사람 평가 샘플과 정기적으로 교차검증하는 것이 좋다.

출처: MT-Bench 논문(Zheng et al., 2023), G-Eval 논문(Liu et al., 2023), Ragas Metrics 문서.

### Ragas 메트릭

| 구분 | Context Recall | Context Precision |
|---|---|---|
| 정의 | 정답에 필요한 근거가 검색 컨텍스트에 포함됐는지 | 관련 컨텍스트가 상위 순위에 배치됐는지 |
| 계산 방식 | `reference` 또는 `reference_contexts`의 사실을 retrieved contexts가 커버하는지 평가 | 각 검색 청크가 질문/정답 기준으로 관련 있는지 판정하고 순위 기반 평균 정밀도 계산 |
| 낮을 때 의심 | 청킹 누락, Top-K 부족, 임베딩 검색 실패, 메타데이터 필터 누락 | 방해 청크 과다, BM25/벡터 가중치 문제, re-ranker 오류 |
| 개선 | 청킹 재설계, Top-K 확대, 하이브리드 검색, 메타데이터 필터 | re-ranker, RRF, 중복 제거, 질의 재작성 |

| 구분 | Faithfulness | Answer Relevancy |
|---|---|---|
| 정의 | 답변의 주장들이 검색 컨텍스트로 뒷받침되는지 | 답변이 질문에 직접 답하는지 |
| 계산 방식 | 답변을 claim 단위로 나누고 각 claim이 컨텍스트에서 지지되는지 평가 | 답변으로부터 질문을 역생성하거나 질문-답변 의미 유사도를 평가 |
| 낮을 때 의심 | 환각, 컨텍스트 외 지식 사용, 근거 혼합 | 질문 오해, 포맷 불일치, 불필요한 장황함 |
| 개선 | 컨텍스트 외 답변 금지 프롬프트, 근거 인용, 낮은 temperature | 질문 재작성, 답변 포맷 제한, 프롬프트 명확화 |

Answer Correctness는 end-to-end 정확도다. Ragas는 의미 유사도와 사실 일치도를 가중 평균해 응답이 `reference`와 의미적으로 맞는지 본다. 기본 아이디어는 embedding 기반 semantic similarity와 LLM Judge 기반 factuality를 함께 보는 것이며, 가중치는 평가 목적에 맞게 조정할 수 있다. 다만 이 지표만으로는 검색이 실패했는지, 검색은 맞지만 생성이 틀렸는지 알 수 없다.

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|---|---|---|---|
| 정답 청크 자체를 검색이 놓침 | Context Recall, Answer Correctness | Top-K 부족, 청킹 실패 | Top-K 확대, 청킹 재설계, Hybrid Search |
| 정답 청크는 있지만 8~10위로 밀림 | Context Precision, Faithfulness | 방해 청크가 상위 점유 | Re-ranking, RRF, 중복 제거 |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness | 환각, 프롬프트 약함 | 컨텍스트 외 답변 금지, 근거 인용 |
| LLM이 질문을 잘못 이해 | Answer Relevancy, Answer Correctness | 질문 의도 파악 실패 | 질문 재작성, few-shot, 출력 포맷 제한 |
| 답은 맞는데 장황 | 기본 Ragas 메트릭에는 직접 반영 어려움 | 간결성은 주관적 품질 | 커스텀 MetricWithLLM 또는 별도 rubric 추가 |

Ragas에서 Faithfulness, Context Recall, LLMContextPrecisionWithReference, ResponseRelevancy, AnswerCorrectness는 대체로 LLM Judge 또는 LLM+embedding 판단을 사용한다. ID 기반 Precision 같은 일부 변형은 규칙 기반으로도 가능하지만, 이번 과제의 표준 5개 메트릭은 평가용 LLM과 임베딩 품질에 의존한다.

## 참고 자료

- Ragas 공식 문서: https://docs.ragas.io/
- Ragas Metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Ragas Test Data Generation: https://docs.ragas.io/en/stable/concepts/test_data_generation/
- RAGAS Paper, Es et al. 2023: https://arxiv.org/abs/2309.15217
- Judging LLM-as-a-Judge, Zheng et al. 2023: https://arxiv.org/abs/2306.05685
- G-Eval, Liu et al. 2023: https://arxiv.org/abs/2303.16634
