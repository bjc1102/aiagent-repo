## 0. 금주 해당 과제 목적

AI 모델에 대한 평가자체를 어떻게 개선시켜야하는가.
- 수동으로 반복 채점 문제에 대해 자동으로 해결하는 것
- RAG 파이프라인에서 정확히 어떤 원인때문에 문제가 발생했는지 아는 것

0-1. Step 1 Ragas 자동평가 실행 메모
- 설치: `python3 -m pip install -r week5/monkana/requirements-ragas.txt`
- Python 3.9 환경에서는 `eval-type-backport`까지 같이 설치되도록 requirements에 포함해둠
- 파일럿 5문항 실행: `python3 week5/monkana/ragas_eval.py --limit 5 --korean-prompts`
- 전체 실행: `python3 week5/monkana/ragas_eval.py --korean-prompts`
- 결과 저장 위치: `week5/monkana/result/step2_pilot_5/basic_ragas_scores.csv`, `advanced_ragas_scores.csv`, `ragas_summary.csv`
- evaluator 기본값은 `openai`로 고정해둠. 이후 Claude evaluator를 쓰려면 `ANTHROPIC_API_KEY`를 준비한 뒤 `--evaluator-provider anthropic` 또는 `auto`로 실행

## 1. 필수 조사 항목

**GoldenDataset**

- 한 줄 정의: 모델 적확성 평가를 위해 만들어진 문제와 답 모음. 모델이 얼마나 적절하고 정확한 답을 내놓는지 평가하기위해 만들어진 가상의 문제와 답변
- 왜 필요한가: 해당 평가 지표를 통해 모델 답변의 신뢰성이 어떻게 되는지 혹은 어떻게 앞으로 개선시킬지에 대한 방향성을 제공하기 떄문에 꼭 필요하다
- 스키마: `user-input`, `response`, `response_context`, `groud_truth`, `grount_truth context`
- 권장 규모: 초기(easy, medium, hard, cross-year 문제 각각 한 문제씩 - 한 문제씩 반복 프로세싱을 거쳐 한 번씩 검증을 해봐야하고, 이는 초반에 많은 양보단 최대한 질 좋은 문제를 만들어내는 것이 적절하다고 판단), 성숙(각 난이도별 2문제씩 해보기 - 초기 테스트 성공시, 다른 시나리오도 구성해서 확인해보기위해), 대규모(각 난이도별 3문제씩 해보기 - 다양한 유형의 문제와 예외 사례(난이도별 하나씩) 만들어주는 것이 좋겠다는 생각)
- reference context: https://docs.ragas.io/en/v0.4.1/concepts/datasets/#loading-an-existing-dataset 의 데이터 셋 생상을 위한 모법 사례 참조

**좋은 dataset의 조건**

- 대표 샘플 : 데이터 세트가 AI 시스템이 접하게 될 실제 시나리오(현실에 있을 법한 질의)인지 확인
- 균형 잡힌 분포 : 다양한 난이도, 주제 및 예외 사례를 아우르는 샘플을 포함하기.
- 양보다 질 : 질이 낮은 샘플을 많이 갖는 것보다 질이 높고 잘 선별된 샘플을 적게 갖는 것이 더 났다.
- 풍부한 메타데이터 : 다양한 측면에서 성능을 분석할 수 있도록 관련 메타데이터를 포함시키는 것이 좋다.
- 버전 관리 : 데이터 세트의 변경 사항을 추후 다시 테스트 해볼 수 있게금 기록해둬야한다.

1-1. 추가 조사

**Ragas framework와 관련된 추가 개념**

- 실험: 가설이나 아디어 검증을 위한 ai 어플리케이션에 의도적으로 변경사항을 적용하는 것을 말한다. 시스템에서 검색기 모델 즉, 임베딩 모델을 교체하는것이 LLM 응답에 어떤 영향을 미치는지를 확인하는 것이 위 변경사항 중 하나하라고 보면 된다.

**좋은 실험의 원칙**

- 측정 가능 지표 정의: 정확도 정밀고 재현율과 같은 지표를 통해 변경 사항 영향을 정령화해야한다.
- 체계적인 결과 정의: 결과를 쉽게 비교하고 추적할 수 있도록 체계적인 방식으로 저장해야한다.
- 변경 사항 분리: 한 번에 하나의 변경만 적용하여 구체적인 영향을 파악해야한다.
- 반복적인 프로세스 구조화된 접근 방식(변경사항 적용 -> 평가 실행 -> 결과 관찰: 반복)을 따라야한다.


## 2. 평가의 필요성과 LLM-as-a-Judge

체계적 평가의 필요성: 단위 테스트 만으로는 LLM RAG 시스템의 확률적 출력을 내는 것 때문에 검증하기 힘들다. 체계적 평가가 없으면 변경의 영향을 판단하기 힘들고, 개선 방향도 알 수 없다. 이 체계를 구현하는 방법이 LLM as a Judge 이며, Ragas Framework의 핵심 매트릭도 해당 원리에 기반되어 있다.

2-1. 왜 체계적 평가가 필요한가

평가 체계가 없을때 발생하는 구체적인 문제들은 아래외 같다.
- 회귀 탐지 불가: 프로픔트 모델 검색 설정을 바꿨을 때 성능 변화를 객관적 근거 기반으로 판단하지 못한다.
- 디버깅 지점 모호: 답변이 틀렸을 때 어느 단계(검색/생성/프롬프트)가 원인이지 모호하다.
- 비교 기준 부재: 여러 모델 파라미터 중 무엇이 나은지 감으로 판단하게된다.
- 프로던션 도입 판단 불가: 이 정도면 쓸만하다 라는 합리적 기준이 없다.

이러한 문제가 발생하는 이유는 근본적으로 소프트웨어 테스트와 LLM,RAG 시스템 테스트의 근본적인 차이 때문이다. 이는 아래와 같다.

| 구분 | deterministic test | probabilistic test |
|------|--------------------|--------------------|
| 기본 가정 | 같은 입력이면 같은 결과 | 같은 입력이어도 결과가 달라질 수 있음 |
| 테스트 목표 | 명세대로 정확히 동작하는지 확인 | 품질, 안전성, 성공률이 충분한지 확인 |

출처:
- https://edms.etas.com/deterministic_behavior.html : deterministic test
- https://arxiv.org/pdf/2509.02012 : probabilistic test
- 바로 위 논문 같은 경우, probabilistic test에서 대체 몇 번 반복해야 하는지까지 알려줌. ProbTest 라는 것

그래서 매번 결과가 달라질 수 있는 특성때문에, 성능상 변화가 있는지 확인해봐야하고, 여러 모델 파라미터 중 어떤 것이 나은지 실제로 보고 확인해보기 위해, 디버깅 지점을 잡기 위해, 프럼프트 검색 모델을 바꿀 때 등 전체 한 번 테스트 과정에서 하나의 변인에 변화를 주어 어떻게 달라지는지 확인해보기위해 LLM RAG 에서도 회귀 검증이 쓰인다.


회귀 검증 개념과 goldenDataSet과의 연결: 시간이 지남에 따라 어플리케이션의 버전간 일관성을 확인한다는 개념이고, 새 버전이 현재 버전의 성능을 떨어뜨리지는 않는지 확인한다는 개념이다. 회귀 검증 수행시, goldenDataset은 고정 기준 데이터 셋이다. 새 버전이 나오면 해당 기준 데이터 셋을 돌려봄으로서 예전 답변과 일치하는지 확인하는 용도로 종종 쓰인다.

출처:
- https://docs.langchain.com/langsmith/evaluation-types : 회귀 테스트 정의
- https://www.evidentlyai.com/blog/llm-testing-tutorial : goldenDataset과의 연관성

**자동 평가 vs 사람 평가 스펙트럼과 trade-off(비용 시간 신뢰성)**

| 구분 | human evaluation | LLM-as-a-Judge(auto judge) |
|------|------------------|----------------------------|
| 장점 | 가장 정확 | 확장성과 설명 가능성 |
| 단점 | 비용이 많이 들고, 오래 걸림 | 사람보단 비교적 정확도가 낮음(80%정도 사람 판단과 일치) |

- 확장성: 질문, 모델, 실험 횟수가 많아져도 감당가능한가
- 설명 가능성: 왜 그런 판정이 나왔는지 자연어 이유를 줌, 점수만 보는 것보다 디버깅과 개선에 유리

출처:
- https://arxiv.org/abs/2306.05685


2-2. Ragas 메트릭을 이해하기위한 선행 개념: LLM as a Judge
- 등장 배경: 기존 방식인 BLEU/ROUGE 등 n-gram 자동 평가인 경우, 형태소들을 n 개 단위로 나누고, 나눠진 것들이 비교 대상과 얼마나 겹치는지 확인하는 것인데, 이는 의미는 같으나 단어 형태가 다를 경우, 겹치지 않는다 라고 판단하는 한계가 있으며, 사람만으로 평가하기에는 비용과 시간이 많이 든다.
- 작동 원리: 루브릭(판정 기준), 출력 형식 필요하면 예시까지 프롬프트에 넣고, 모ㄹ이 답변을 읽은 뒤, 각 답변에 관한 점수 및 답변간 비교 그리고 이유까지 출력하게 하는 구조로 이뤄진다. G-EVAL 은 이것을 더 체계화 시켜, Task Introduction → Evaluation Criteria → Evaluation Steps → Evaluation Form 구조로 설계한다. Task Introduction는 처리할 업무 대상이 무엇인지 알려주는 것이고, Evaluation Criteria는 무슨 기준으로 볼지 적는 부분이다. Evaluation Steps는 어떤 순서로 판단할지 적는 부분이고, Evaluation Form은 최종 결과를 어떤 제출 형식으로 고정할지 적는 부분이다.
- 사람 평가와의 일치율: MT-Bench / Chatbot Arena 논문에 따르면, GPT-4 judge가 사람 전문가 평가와 85% agreement를 보였고, 사람끼리의 agreement는 **81%**였다고 보고한다. 즉, 완전한 대체라고 보기는 어렵디만, 논문 실험 조건에서는 사람 다수 판단과 꽤 가깝게 움직였다고 볼 수 있다. 다만 이 수치는 권장 참고치로만 보는 것이 맞고, 모든 과제 및 모든 프로픔트에서 그대로 재현된다고 보면 안된다. 말그대로 LLM 이기에 말이다.
- 루브릭 설계 원칙: 무엇을 평가하는지, 각 점수가 무슨 뜻인지, 어떤 형식으로 답해야 하는지가 분명히다. 클로드 문서와 G-EVAL을 같이 보면 원칙은 총 4가지이다. 첫째, 평가 항목을 쪼꺠야하며, 둘째, 점수 구간의 의미를 적어야하고, 셋째 출력 형식을 고정하고, 넷째, 가능하면 예시를 넣어야한다. 다음은 좋은 루브릭의 예시이다.

```text
당신은 RAG 답변 평가자다.
아래 4개 항목만 평가하라.

/평가 항목 쪼개야하며,
1. 정확성(1~5): 사실이 맞는가 
2. 근거성(1~5): 주어진 context로 뒷받침되는가
3. 관련성(1~5): 질문 의도에 직접 답하는가
4. 간결성(1~5): 불필요한 반복 없이 핵심만 말하는가
/각 항목의 정의를 적고

출력 형식:
- accuracy: [1~5]
- groundedness: [1~5]
- relevance: [1~5]
- conciseness: [1~5]
- overall: [1~5]
- rationale: 3문장 이내
/출력 형식을 고정한다.
```

- 한계 및 신뢰성: 고정된 답안 채점기처럼 항상 똑같이 동작하지는 않는다. MT-Bench 논문은 position bias, verbosity bias, self-enhancement bias, math/reasoning grading 한계를 직접 보여준다. 답변 순서를 바꾸면 판정이 달라진다거나, 길게 부풀린 답변에 속을 수도 있고, 자신과 비슷한 모델에게 더 높은 점수를 부여하는 편향성을 보이거나, 수학 및 추론 영역 문제에 있어서도 생각보다 잘 틀린다는 실험 결과가 나온것이 그 예시다. 그리고 비용도 무시못한다 결국, 채점을 LLM 모델에게 한다는 것이기에 토큰 사용량이 늘어난다는 것이고, 그리고 판단을 더 똑똑하게 할수록 비용 추가 증가와 생각 하는 시간으로 인해 지연율이 올라가게된다. 그래서 고정 평가셋 + LLM judge + 샘플 인간 검토를 섞는것이 현실적인 대안으로 나오는 추세다.

- Claude의 thinking / interleaved thinking과 평가 루브릭의 관계: thinking은 바로 점수부터 찍지 말고 먼저 한 번 생각하고 판단하라는 뜻이다. 예를 들어 루브릭이 정확성, 근거성, 관련성을 이라면, thinking을 넣을시, 정확한가 -> 근거가 있는가 -> 질문에 맞는가 -> 최종 점수는? 이렇게 루브릭을 단계적으로 적용시킨다. 
interleaved thinking은 중간에 새로운 정보가 들어올때 한 번 생각을 하는 것이다. 
질문을 읽음 -> 답변을 읽음 -> 한 번 생각함 -> 검색/툴 결과를 받음 -> 그 결과를 보고 다시 생각함 -> 최종 점수 줌 이렇게 새로운 정보가 들어올 때마다 생각을 넣는 것이다. 그래서 루브릭이 채점표 이면, thinking은 채점표를 순대로 체크하는 채점 절차이고, interleaved thinking은 자료를 더 받으면 채점표를 다시 보고 다시 점수를 수정하는 절차라고 이해하면 된다.

- Ragas와의 관계: Ragas 자체가 LLM + 규칙 기반 judge 방식이다. 즉, LLM as judge 개념이 Ragas 개념에 포함되는 관계이다. Ragas가 RAG가 잘됐는지 확인해보러면 크게 4가지 관점에서 봐야한다고 한다.

| 항목 | 쉽게 말해 뭘 보나 | 주된 방식 |
|------|------------------|-----------|
| Faithfulness | 답변이 근거 문서에 충실한가 | LLM Judge형 |
| Answer Relevancy | 답변이 질문에 맞는가 | LLM + 유사도 계산 |
| Context Precision | 가져온 문서가 앞쪽에 잘 정렬됐나 | LLM형도 있고 규칙형도 있음 |
| Context Recall | 필요한 정보가 retrieval에 빠지지 않았나 | LLM형도 있고 규칙형도 있음 |

이런식으로 잡혀있다면, LLM 형으로 판단하는 방법이 결국 Ragas가 4가지 관점에서의 하나의 방식일 뿐이다.

## 3. Ragas 4대 메트릭(+Answer Correctness)

3-1. 검색 단계

| 구분 | Context Recall | Context Precision |
|------|----------------|------------------|
| 정의 | 검색된 context가 ground truth와 얼마나 잘 일치하는지 평가 | 가져온 context가 우선 순위 앞쪽에 잘 정렬됐나 |
| 계산 방식 | retrieved context가 뒷받침하는 ground truth claim 수 / ground truth 전체 claim 수 | `Precision@k = top k 내 관련 문서 수 / k` |
| 낮을 때 의심할 점 | retriever가 아예 필요한 청크를 못찾았거나, top-k가 너무 작은지, 청크 분할의 문제, reference(정답)가 retrival로 얻을 수 없는 정보까지 포함 | 검색 랭킹의 문제, 청크 분할의 문제, 쿼리가 너무 넓거나 애매한 문제 |
| 개선 기법 | top-k 늘리기, 청킹 방식 바꾸기, 검색 쿼리 강화, 하이브리드 검색 사용, 문서(표, 리스트) 전처리 개선 | reranker 붙이기, top-k 줄이기, 청크를 더 작게 집중되게(의미단위 분할), 메타데이터 필터링 추가, 질문 문서 매칭 기준을 더 엄격하게(exact keyword, entity match, title boost 같은 걸 추가) |

- claim : 어떤 문장 안에 들어있는 개별 주장 및 사실 단위
- context: 문서 조각들 즉, 청크들의 묶음
- context presision의 계산 방식 이해: Precision@k(1등부터 k등까지의 정답 관련 청크 비율), K(가져온 청크 개수), v_k = k번째 청크가 관련 있으면 1, 아니면 0 / topk 안의 관련 청크 총 개수 라서 예를 들어 1등 청크가 관련 있고, 2등 청크는 관련 없고, 3등 청크는 관련 있고, 4등 청크는 관련이 없다면, Precision@1 = 1, Precision@2 = 0.5, Precision@3 = 0.66, Precision@4 = 0.5 이고, 각각 v_k를 곱하면 순서대로 1,0, 0.66, 0 이며, 이들을 다 더하면 1.66 이다. 그리고 이를 관련 청크 개수가 총 2개 이므로 1.66 / 2 = 0.833 이므로 즉 점수는 0.833 이다.
- exact keyword: 질문에 들어있는 단어가 문서에도 정확히 있으면 점수 추가
- entity match: 질문 속 핵심 객체가 문서에도 있으면 점수 추가
- title boost: 문서 제목에 중요 단어 포함하면 점수 플러스



3-2. 생성 단계

| 구분 | Faithfulness | Answer Relevancy |
|------|--------------|------------------|
| 정의 | response가 retrived context와 얼마나 사실적으로 일치 | 답변이 질문의 의도와 맞는지 |
| 계산 방식 | context가 지지하는 답변 claim 수 / response의 전체 claim 수 | 답변으로부터 생성한 질문들과 원래 질문의 의미 유사도 평균 |
| 낮을 때 의심할 점 | retrived context이 불안정(필요 정보 공백 = 청크 안에 필요 정보 자체가 없거나 짤렸거나 등 -> LLM 추정으로 공백 메움 -> 환각 발생), 답변이 과하게 길어져 claim 수 증가 | 1) 질문 해석 실패: 질문 조건, 형식, 멀티턴 맥락을 놓침. 2) 검색 실패: 질문과 안 맞는 context를 가져옴. 3) 생성 단계 실패: 핵심 답 대신 배경설명만 길게 함. 4) 복합 질문 처리 실패: 질문 일부만 답함 |
| 개선 기법 | 청크 완전성 확보, context 밖 정보 사용 금지, 답변 길이를 줄이고 claim 수도 줄이기 | 질문 의도를 더 강하게 프롬프트에 삽입, 답변에서 쓸데없는 주변 설명 줄이도록 프롬프트에 전달, 복합 질문이면 하위 질문으로 분해 |

3-3. End to End

**Answer Correctness 정의**

최종 답변이 ground truth와 얼마나 정확하게 맞는지 semantic similarity와 factual similarity를 함께 보는 지표

**Answer Correctness 계산 방식**

1. answer와 ground truth를 사실 단위로 비교해 TP / FP / FN을 만든다.
2. factual correctness를 F1 score로 계산한다.
3. answer와 ground truth의 semantic similarity를 계산한다.
4. 두 값을 가중 평균하여 최종 점수를 만든다.

- TP: 맞게 말한 사실 수
- FP: 모델이 말했는데 정답 기준으로 틀리거나 불필요한 사실 수
- FN: 정답에는 있는데 모델이 놓친 사실 수
- F1 = 2TP / (2TP + FP +FN)

**ground_truth 품질 의존성**

ground truth 품질에 강하게 의존한다. 비교 기준이 groud truth 이여서 groud truth 자체거 부정확하고 너무 짧고 필요 사실 누락 미 애매하게 쓰여있다면 Answer Correctness도 왜곡되기때문이다.

**Answer Correctness만으로 부족한 이유**

이 지표는 최종 결과가 맞았는지는 말해주지만, 왜 그렇게 됐는지는 말해주지 못한다. 그래서 retrieval 쪽의 Context Precision / Context Recall, generation 쪽의 Faithfulness / Answer Relevancy 같은 지표를 같이 봐야한다. 

3-4. 메트릭 간 관계

| 시나리오 | 낮아지는 메트릭 | 원인 | 대응 |
|----------|------------------|------|------|
| 정답 청크 자체를 검색이 놓침 | Context Recall | 청킹 문제, top-k 부족, 질문 자체의 문제 |  |
| 정답 청크는 있지만 8~10위로 밀림 | Context Precision | 청킹 문제, retriver 모델 문제, 랭킹 기준 문제 | metadata filtering, exact keyword/entity boost, top-k 재조정 |
| 검색은 맞는데 LLM이 외부 정보 추가 | Faithfulness | context에 정보 공백 및 프롬프트 통제 부족 | context만 사용 제약 강화, 답변 길이 축소, 근거 인용 강제 프롬프팅 |
| LLM이 질문을 잘못 이해 | Answer Relevancy | 질문 해석 실패, 복합 질문 분해 실패, 대화 맥락 놓침 | 복합 질문을 하위 질문으로 나눠서 질문, 이 질문을 왜 하는지 강조, 질문의 이해를 돕기 위해 예시 추가 |
| 답은 맞는데 장황 | (해당 없음) | 불필한 배경 및 주변 설명이 많음 | 간결하고 핵심만 말하도록 프롬프팅 강제 |

**답은 맞는데 장황한 시나리오에서 낮아지는 메트릭 항목이 존재하지 않는 이유**

질문에 맞는 내용이 들어 있고, 사용자가 궁금한 정보도 빠지지 않았다면, 일반적으로 Answer Relevancy 자체가 반드시 낮아진다고 보기는 어렵다. 또한 Context Recall은 검색 단계에서 필요한 정보가 빠졌는지를 보는 지표이므로, 답변이 장황하다는 사실만으로 직접 낮아지지 않는다. Context Precision 역시 상위 검색 결과에 관련 청크가 잘 배치되었는지를 보는 지표이므로, 최종 답변의 길이와는 직접적인 관계가 없다.
다만 답변이 길어질수록 context에 없는 추가 claim이 섞일 가능성은 커지므로, 이 경우에는 Faithfulness가 낮아질 수 있다. 따라서 “답은 맞지만 장황하다”는 상황은 기본적으로는 현재 4개 메트릭만으로 직접 포착되는 문제가 아니라, 간결성/스타일 품질 문제로 보는 것이 더 정확하다. 단, 장황함 때문에 문맥 밖 정보가 섞이면 그때는 Faithfulness 저하로 이어질 수 있다.

## 4. 실습 과제 예측
4-1. 4주차 정답률(사람)과 Ragas Ans Correctness(자동)의 일치 정도? 어느 문항에서 차이 예상?
- 가설: 80~85%, 단답형 문제(단순 계산결과, 본인 부담률)같은 경우 거의 차이가 없겠지만, 답변이 만약 도메인 특화 용어에 대해 LLM이 생성한 답변에서 ground_truth에 있는 용어와 동음이의어라면, 이에 대해 자동 채점은 틀렸다고 답변할 가능성이 높다고 판단되어 관련 문제에 있어서는 차이가 있을 거라 생각한다.
- 실습 결과: 파일럿 5문항 기준으로 사람 채점의 정답/오답 판단과 Ragas `Answer Correctness`의 방향성은 대체로 일치했지만, 점수 수준은 완전히 같지 않았다. `q02`, `q08` 같은 단순 조회형은 자동 점수도 거의 1.0에 가깝게 나왔지만, `q03`, `q05` 같은 계산형과 `q10` 같은 cross-year 서술형은 사람 기준 정답이어도 자동 점수가 더 보수적으로 낮아졌다. 즉, 예상했던 용어 차이뿐 아니라 `계산 결과 문장이 context에 직접 없을 때`, `reference와 문장 구조가 다를 때`도 차이가 크게 발생했다.
4-2. Basic/Advanced의 네 메트릭 중 가장 크게 벌어질 메트릭은? 이유는?
- 가설: Context Precision, Advanced는 검색 성능 향상을 목적으로한 개념이고, 특히나 검색 전 쿼리 확장 에를 들어 HybrudSearch 같은 경우, dense와 sparse 비율 조정을 통해 사용자가 질의에 적합한 청크 후보들을 뽑아내는데 도움을 주며, 리랭커를 통해 더 관련성 있는 문서를 상위에 배치해두기에 Context Precision에서 크게 벌어질 것이다.
- 실습 결과: Basic vs Hybrid only 5문항 결과에서는 수치상 `Faithfulness` 차이(0.5 vs 0.6)가 가장 컸고, retrieval 품질 차이는 `Context Precision`(0.8167 vs 0.7491)에서 더 해석 가능하게 드러났다. 즉, 가설처럼 Advanced 쪽 retrieval 변화가 `Context Precision`에 영향을 주긴 했지만, 방향은 개선이 아니라 오히려 하락이었다. 반면 `Faithfulness`는 계산형 문항과 judge의 보수적 해석에 크게 흔들려서, 숫자 차이가 가장 커도 이를 그대로 검색 개선 효과로 해석하기는 어려웠다.
4-3. 년도 혼동 문제는 어느 Ragas 메트릭에 주로 반영될 것인가?
- 가설: 년도 혼동 문제의 주의점은 같은 주제에 대해 연도마다의 청크를 구분해주고 잘 가져오는 것이 핵심이다. 따라서 연도별 필요한 정보가 누락되었는지가 핵심이기에 context Recall 항목이 주로 반영될 것이다.
- 실습 결과: 실제로는 `Context Recall`이 주로 반영되지는 않았다. 파일럿 5문항에서 Basic/Hybrid only 모두 `Context Recall`은 1.0으로 유지된 반면, cross-year 위험은 `Context Precision` 하락과 `Answer Correctness` 변화에서 더 잘 드러났다. 즉, 필요한 연도 청크를 "가져왔는가"보다, `가져온 연도 청크를 얼마나 깔끔하게 분리했는가`, `최종 답변에서 연도별 기준을 정확히 대응했는가`가 더 중요했다. 그래서 최종적으로는 기본 Ragas만으로 부족했고, `연도 혼동 안 했냐`를 직접 보는 커스텀 메트릭 `Year Accuracy`를 추가하게 되었다.
4-4. Advanced에서 Faithfulness가 오히려 낮아질 시나리오가 있을까?
- 가설: 검색 성능을 향상시키는데 쓰이는 Advanced 기법이 Faithfulness를 낮추는데에는 상관성이 없어보인다. 애초에 Faithfulness LLM이 생성한 답변이 context 기반으로 했는지를 따지는 것인데, 가져온 context 자체에 청크 단위가 잘못되었거나, 질문 자체에서 context만으로 답변을 뽑아내기 힘들거나, 애초에 context 안 내용이 부족하거나 등의 문제지. 검색 성능을 향상시키는데 목적이 있는 Advanced가 Faithfulness 자체에 영향을 준다고 보기는 어렵다. 물론 해당 기법으로인해 검색 성능이 떨어져서 LLM이 참조할 context가 없어 환각을 일으킬 경우, 해당 지표에 악영향을 줄 가능성은 존재하지만, 그건 Advanced 기법을 잘못 써서 발생한 문제지. Advanced 자체에 문제가 있다고 보기는 어렵다고 판단한다.
- 실습 결과: 실제로는 그런 시나리오가 있었다. 대표적으로 `q10`에서는 Basic과 Advanced 모두 필요한 청크를 찾았고 `Context Recall`, `Context Precision`도 1.0이었지만, Advanced 답변은 `기준이 변경되었다`는 reference 구조보다 `본인부담 대상에 포함된다`는 식으로 서술이 이동하면서 `Faithfulness`가 더 낮게 평가되었다. 즉, Advanced가 retrieval을 개선하더라도 답변이 `context에 직접 적힌 표현`보다 한 단계 재서술되거나, 계산형처럼 `문맥 수치로부터 추론한 결과 문장`을 만들면 Faithfulness는 오히려 떨어질 수 있었다.

평가 관련 폴더/파일 구조
- `week5/monkana/BasicRag.py`: Basic RAG 실행 및 커스텀 judge 기반 텍스트 평가를 수행하는 기본 파이프라인 파일이다.
- `week5/monkana/HybridRerank.py`: 현재 README에서 `Hybrid only`라고 부르는 하이브리드 검색 파이프라인 파일이다. 이름은 예전 명칭이 남아 있지만, 본 실험 기준의 기본 hybrid 비교 대상이다.
- `week5/monkana/HybridCom.py`: `Hybrid Search + Contextual Compression` 조합을 실험하기 위해 추가한 파일이다.
- `week5/monkana/HybridComRerank.py`: `Hybrid Search + Contextual Compression + Rerank` 조합을 실험하기 위해 추가한 파일이다.
- `week5/monkana/ragas_eval.py`: Basic와 Hybrid 계열 파이프라인 결과를 Ragas 메트릭으로 자동 평가하는 메인 스크립트다.
- `week5/monkana/ragas_eval_with_year_accuracy.py`: 기본 Ragas 메트릭에 커스텀 `Year Accuracy`를 추가해 평가하는 전용 실행 파일이다.
- `week5/monkana/result/step2_pilot_5`: Basic vs Hybrid only 파일럿 5문항 Ragas 결과를 저장하는 폴더다.
- `week5/monkana/result/HybridTest`: `Hybrid + Compression`, `Hybrid + Compression + Rerank` 실험 결과를 저장하는 폴더다. 커스텀 txt 평가와 Ragas csv/md 결과가 함께 들어 있다.
- `week5/monkana/result/WithYearAcc`: 커스텀 메트릭 `Year Accuracy`를 포함해 다시 평가한 결과를 저장하는 폴더다.
- `evaluation_results.txt`, `evaluation_result(step2).txt`, `evaluation_hybrid_com.txt` 같은 txt 파일들은 Ragas 결과가 아니라, 각 파이프라인에서 직접 만든 커스텀 LLM judge 리포트다.

## 5. Step 1 실제 수행 내용
- 목표: 4주차 Basic RAG, Advanced RAG 실행 결과를 Ragas가 바로 평가할 수 있는 형태로 연결하는 것
- 구현 파일: `week5/monkana/ragas_eval.py`
- 설치 파일: `week5/monkana/requirements-ragas.txt`

Step 1에서 한 작업
- `goldenDataset.jsonl`의 `question`, `ground_truth`, `ground_truth_contexts`를 읽어 평가용 입력으로 사용
- Basic/Advanced RAG를 각각 실행해서 `response`, `retrieved_contexts`, `retrieved_docs`를 수집
- 수집한 값을 `SingleTurnSample` 스키마에 맞게 매핑
  - `question` -> `user_input`
  - `ground_truth` -> `reference`
  - `ground_truth_contexts` -> `reference_contexts`
  - RAG 답변 -> `response`
  - 검색 청크 -> `retrieved_contexts`
- `EvaluationDataset`을 Basic/Advanced 각각 생성
- 평가용 LLM, 임베딩을 연결하고 아래 5개 메트릭을 실행할 수 있도록 구성
  - `LLMContextRecall`
  - `LLMContextPrecisionWithReference`
  - `Faithfulness`
  - `ResponseRelevancy`
  - `AnswerCorrectness`
- 한국어 데이터셋 평가 안정성을 위해 `--korean-prompts` 옵션 사용 시 Ragas 내부 judge 프롬프트를 한국어로 적응하도록 구현

RAG 코드 쪽 보완
- `BasicRag.py`, `HybridRerank.py`에 평가용 `run_pipeline()` 인터페이스를 추가
- 이 함수가 문항 1개에 대해 아래 값을 함께 반환하도록 맞춤
  - `response`
  - `retrieved_contexts`
  - `retrieved_docs`

Step 1 결과 파일
- `basic_ragas_inputs.jsonl`, `advanced_ragas_inputs.jsonl`
  - Ragas에 넣기 직전의 실행 기록
- `basic_ragas_scores.csv`, `advanced_ragas_scores.csv`
  - 문항별 Ragas 점수
- `ragas_summary.csv`
  - Basic/Advanced 평균 비교
- `ragas_run_metadata.json`
  - 실행 설정 기록

검증 상태
- `limit 1` 파일럿으로 Basic/Advanced 모두 실제 평가 호출과 결과 저장까지 확인
- `--korean-prompts` 옵션도 별도 파일럿으로 확인
- 따라서 Step 1 기준의 "환경 구축 + 데이터 매핑 + evaluate() 동작 확인"까지 완료한 상태

## 6. Step 2 파일럿 5문항 결과 정리: Basic vs Hybrid only

2-1. 메트릭 실행 결과
- 평가 데이터셋: `golden_dataset_step2_pilot_5.jsonl` (5문항)
- 실행 파이프라인: Basic RAG, Hybrid only
- 평균 점수 요약

| 메트릭 | Basic | Hybrid only | 변화(Hybrid only - Basic) |
|--------|------:|---------:|-----------------------:|
| Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Context Precision | 0.8167 | 0.7491 | -0.0675 |
| Faithfulness | 0.5000 | 0.6000 | 0.1000 |
| Answer Relevancy | 0.5007 | 0.4758 | -0.0249 |
| Answer Correctness | 0.9483 | 0.9273 | -0.0210 |

- 해석:
Basic와 Hybrid only 모두 `Context Recall`은 1.0으로 동일했다. 즉, 이번 파일럿 5문항에서는 두 파이프라인 모두 정답에 필요한 핵심 근거 자체는 retrieval 단계에서 놓치지 않았다고 볼 수 있다. 반면 `Context Precision`은 Basic이 더 높았고, `Answer Correctness`와 `Answer Relevancy`도 Basic이 소폭 높게 나왔다. 따라서 이번 파일럿에서는 Hybrid only가 검색 상위 정렬 품질이나 최종 답변 품질을 뚜렷하게 개선했다고 보기는 어려웠다. 다만 `Faithfulness`는 Basic 0.5, Hybrid only 0.6으로 Hybrid only가 약간 높게 나왔다.

2-2. 결과 기록
- 대표 문항별 비교

| 질문 ID | 난이도 | source_year | Ctx Recall (B/H) | Ctx Precision (B/H) | Faithfulness (B/H) | Ans Relevancy (B/H) | Ans Correctness (B/H) |
|---------|--------|------------|------------------|---------------------|--------------------|---------------------|-----------------------|
| q02 | easy | 2026 | 1.0000 / 1.0000 | 0.7500 / 0.6429 | 1.0000 / 1.0000 | 0.4466 / 0.4466 | 1.0000 / 1.0000 |
| q03 | medium | 2025 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.4381 / 0.4381 | 0.8854 / 0.8854 |
| q05 | hard | 2025 | 1.0000 / 1.0000 | 1.0000 / 0.9029 | 0.0000 / 0.0000 | 0.6469 / 0.5772 | 0.8808 / 0.8805 |
| q08 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 0.3333 / 0.2000 | 0.5000 / 1.0000 | 0.4290 / 0.4290 | 1.0000 / 1.0000 |
| q10 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.5428 / 0.4880 | 0.9751 / 0.8706 |

- 대표적으로 `q03`, `q05`처럼 계산형 문항에서 `Faithfulness`가 0으로 나온 부분이 눈에 띄었다.

Faithfulness가 0으로 나온 이유 설명
- `ground_truth_contexts`가 잘못 들어가서가 아니다. `ground_truth_contexts`는 retrieval 계열 지표인 `Context Recall`, `Context Precision`에 더 직접적으로 연결되고, 실제로 `q03`, `q05`는 두 문항 모두 `Context Recall = 1.0`으로 나왔다. 즉, 정답 근거 자체는 dataset에도 있었고, retrieval도 그 핵심 근거를 가져왔다고 볼 수 있다.
- 문제는 `Faithfulness`가 `ground_truth_contexts`가 아니라 `response`와 `retrieved_contexts`의 직접 일치 여부를 본다는 점이다. 계산형 문항의 경우 검색 문맥에는 보통 `본인부담률 40%`, `본인부담률 15%` 같은 원자료만 표 형태로 들어 있고, `비용이 100,000원이므로 본인부담금은 40,000원이다` 같은 최종 계산 문장은 직접 적혀 있지 않다.
- 따라서 사람 기준으로는 정답이어도, Ragas의 `Faithfulness` judge는 이를 "문맥에 그대로 적힌 사실"이 아니라 "문맥을 바탕으로 모델이 계산해 만든 문장"으로 보수적으로 해석할 수 있다. 이번 파일럿에서 `q03`, `q05`가 바로 그런 사례였다.
- 정리하면, 단순 조회형 문항은 문맥에 있는 값을 거의 그대로 문장화하면 되기 때문에 `Faithfulness`가 높게 나오기 쉽다. 반면 계산형 문항은 `Answer Correctness`는 높게 나오더라도 `Faithfulness`가 과소평가될 수 있다. 그래서 계산형 해석에서는 `Faithfulness` 하나만 보지 말고 `Answer Correctness`와 `Context Recall`을 같이 보는 것이 더 적절하다고 판단된다.

2-3. 4주차 수동 채점 vs Ragas 비교
- 이번 파일럿 5문항은 4주차 수동 채점 기준으로 모두 정답 처리되었던 유형들에 해당한다. 따라서 이번 비교의 핵심은 "정답/오답 방향이 일치하는가"와 "자동 점수가 어떤 유형에서 상대적으로 낮아지는가"를 보는 것이다.

| 질문 ID | 4주차 수동 판정 | Ragas Ans Correctness (B/H) | 일치 | 불일치 원인 |
|---------|----------------|-----------------------------|------|-------------|
| q02 | 정답 | 1.0000 / 1.0000 | 일치 | - |
| q03 | 정답 | 0.8854 / 0.8854 | 일치 | - |
| q05 | 정답 | 0.8808 / 0.8805 | 일치 | - |
| q08 | 정답 | 1.0000 / 1.0000 | 일치 | - |
| q10 | 정답 | 0.9751 / 0.8706 | 일치 | - |

- 해석:
이번 파일럿 5문항에서는 4주차 수동 채점의 정답/오답 방향과 Ragas `Answer Correctness`의 방향은 전부 일치했다. 즉, 사람 기준으로 정답이었던 문항들이 자동 평가에서도 모두 높은 점수(약 0.88 이상)를 받았고, 파일럿 범위에서는 뚜렷한 충돌 사례는 없었다. 다만 계산형 문항인 `q03`, `q05`는 사람 기준으로는 명확한 정답이지만, 자동 점수는 1.0보다 낮게 형성되었다. 이는 계산 결과 문장이 검색 문맥에 직접 적혀 있지 않고, 모델이 문맥의 수치와 조건을 바탕으로 계산해 만든 답이기 때문에 자동 평가가 더 보수적으로 작동했기 때문으로 해석할 수 있다.

## 7. Hybrid + Compression + Rerank (Advanced) 파일럿 5문항 결과 정리

2-1. 메트릭 실행 결과
- 평가 데이터셋: `golden_dataset_step2_pilot_5.jsonl` (5문항)
- 실행 파이프라인: `Hybrid + Contextual Compression`, `Hybrid + Contextual Compression + Rerank (Advanced)`
- 결과 파일: `result/HybridTest`

| 메트릭 | Hybrid+Compression | Hybrid+Compression+Rerank | 변화(Rerank - Compression) |
|--------|-------------------:|--------------------------:|----------------------------:|
| Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Context Precision | 0.7586 | 0.6300 | -0.1286 |
| Faithfulness | 0.5000 | 0.5000 | 0.0000 |
| Answer Relevancy | 0.8449 | 0.9007 | 0.0558 |
| Answer Correctness | 0.8953 | 0.9194 | 0.0241 |

- 해석:
두 파이프라인 모두 `Context Recall`은 1.0으로 동일했다. 즉, compression을 붙인 이후에도 필요한 핵심 근거 자체를 놓치지는 않았다. 다만 `Context Precision`은 `Hybrid+Compression`이 더 높았고, rerank를 추가한 뒤에는 오히려 평균 precision이 더 낮아졌다. 반면 `Answer Relevancy`와 `Answer Correctness`는 rerank를 추가한 쪽이 소폭 높아져, 이번 파일럿에서는 rerank가 검색 결과의 정렬 깔끔함을 높였다기보다는 최종 답변 표현을 조금 더 안정화한 쪽에 가까웠다. `Faithfulness`는 두 파이프라인 모두 0.5로 동일해, 계산형 문항에서 나타난 보수적 평가 성향은 그대로 유지되었다.

2-2. 결과 기록
- 대표 문항별 비교

| 질문 ID | 난이도 | source_year | Ctx Recall (C/R) | Ctx Precision (C/R) | Faithfulness (C/R) | Ans Relevancy (C/R) | Ans Correctness (C/R) |
|---------|--------|------------|------------------|---------------------|--------------------|---------------------|-----------------------|
| q02 | easy | 2026 | 1.0000 / 1.0000 | 1.0000 / 0.9167 | 1.0000 / 1.0000 | 0.9826 / 0.9826 | 1.0000 / 1.0000 |
| q03 | medium | 2025 | 1.0000 / 1.0000 | 1.0000 / 0.5000 | 0.0000 / 0.0000 | 0.8824 / 0.8824 | 0.8050 / 0.8854 |
| q05 | hard | 2025 | 1.0000 / 1.0000 | 0.7929 / 0.7333 | 0.0000 / 0.0000 | 0.8553 / 0.8553 | 0.8828 / 0.8828 |
| q08 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 0.8953 / 0.8953 | 1.0000 / 1.0000 |
| q10 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.5000 / 0.5000 | 0.6090 / 0.8879 | 0.7887 / 0.8288 |

- 해석:
이번 비교에서는 `q03`, `q10`에서 rerank를 추가한 쪽의 `Answer Correctness`가 소폭 올라갔다. 특히 `q10`은 `Answer Relevancy`도 `0.6090 -> 0.8879`로 크게 상승해, cross-year 문항에서 rerank가 최종 답변 정리에 도움을 준 것으로 볼 수 있다. 반대로 `q03`은 정답성은 올라갔지만 `Context Precision`은 `1.0000 -> 0.5000`으로 크게 떨어졌고, 전체 평균에서도 rerank 쪽의 precision이 더 낮았다. 즉, 이번 파일럿에서는 `Hybrid + Compression + Rerank`가 항상 더 좋은 retrieval 정렬을 보장한 것은 아니었고, 일부 문항에서는 정답성 개선과 precision 저하가 동시에 나타났다.

- 추가 관찰:
계산형 문항인 `q03`, `q05`는 compression을 붙인 뒤에도 여전히 `Faithfulness = 0`으로 남았다. 이는 앞서 Basic/Advanced 비교에서 확인한 것과 같은 이유로, 검색 문맥에는 비율이나 조건이 표 형태로 들어 있지만 최종 계산 결과 문장은 직접 적혀 있지 않기 때문이다. 따라서 이 구간에서는 `Faithfulness`를 그대로 개선하기보다, `Answer Correctness`와 `Context Recall`을 함께 보는 해석 방식이 더 적절하다고 판단된다.

## 8. Basic vs Hybrid + Compression + Rerank (Advanced) 최종 비교

2-2. 결과 기록
- 비교 대상: `Basic RAG` vs `Hybrid + Contextual Compression + Rerank (Advanced)`
- 비교 기준: 동일한 `golden_dataset_step2_pilot_5.jsonl` 5문항 Ragas 결과

| 질문 ID | 난이도 | source_year | Ctx Recall (B/HCR) | Ctx Precision (B/HCR) | Faithfulness (B/HCR) | Ans Relevancy (B/HCR) | Ans Correctness (B/HCR) |
|---------|--------|------------|--------------------|-----------------------|----------------------|-----------------------|-------------------------|
| q02 | easy | 2026 | 1.0000 / 1.0000 | 0.7500 / 0.9167 | 1.0000 / 1.0000 | 0.4466 / 0.9826 | 1.0000 / 1.0000 |
| q03 | medium | 2025 | 1.0000 / 1.0000 | 1.0000 / 0.5000 | 0.0000 / 0.0000 | 0.4381 / 0.8824 | 0.8854 / 0.8854 |
| q05 | hard | 2025 | 1.0000 / 1.0000 | 1.0000 / 0.7333 | 0.0000 / 0.0000 | 0.6469 / 0.8553 | 0.8808 / 0.8828 |
| q08 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 0.3333 / 0.0000 | 0.5000 / 1.0000 | 0.4290 / 0.8953 | 1.0000 / 1.0000 |
| q10 | cross-year | 2025,2026 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 0.5000 | 0.5108 / 0.8879 | 0.9751 / 0.8288 |

- 해석:
`Context Recall`은 전 문항에서 두 파이프라인이 동일하게 1.0으로 나타나, 필요한 핵심 근거를 아예 놓치는 문제는 없었다. 그러나 `Context Precision`은 전반적으로 Basic 쪽이 더 안정적이었고, 특히 `q03`, `q08`에서 Hybrid + Compression + Rerank 쪽 precision이 크게 낮아졌다. 반면 `Answer Relevancy`는 대부분 문항에서 Hybrid + Compression + Rerank가 더 높아, 질문에 바로 맞는 문장 형태로 답변하는 경향은 더 강했다. `Answer Correctness`는 `q02`, `q03`, `q08`에서는 거의 동일했지만, `q10`에서는 Basic이 더 높아 cross-year 비교 문항의 최종 정답성은 Basic이 더 안정적이었다.

2-3. 문항별 우세 파이프라인 비교

| 질문 ID | Basic Ans Correctness | Hybrid+Compression+Rerank(Advanced) Ans Correctness | 우세 파이프라인 | 해석 |
|---------|-----------------------|-------------------------------------------|----------------|------|
| q02 | 1.0000 | 1.0000 | 동률 | 두 파이프라인 모두 정확한 단순 조회 답변을 생성했다. |
| q03 | 0.8854 | 0.8854 | 동률 | 계산형 문항이며 두 파이프라인 모두 정답이지만 완전한 1.0까지는 오르지 않았다. |
| q05 | 0.8808 | 0.8828 | Hybrid+Compression+Rerank | 차이는 매우 작지만 rerank 쪽이 소폭 높았다. |
| q08 | 1.0000 | 1.0000 | 동률 | cross-year 단순 비교형이라 두 파이프라인 모두 정확하게 응답했다. |
| q10 | 0.9751 | 0.8288 | Basic | cross-year 기준 설명 문항에서는 Basic이 더 높은 정답성을 보였다. |

- 해석:
최종 `Answer Correctness`만 기준으로 보면 이번 파일럿에서는 Basic이 전체적으로 더 우세했다. `q05`에서만 Hybrid + Compression + Rerank가 아주 소폭 앞섰고, 나머지 문항은 동률이거나 Basic이 더 높았다. 특히 `q10`처럼 기준 변화 내용을 함께 설명해야 하는 cross-year 문항에서는 Basic이 더 안정적으로 높은 점수를 받았다. 따라서 이번 5문항 결과만 놓고 보면, `Hybrid + Compression + Rerank`가 질문 적합성 측면의 표현은 좋아졌지만, 최종 정답성까지 Basic을 확실히 넘어섰다고 보기는 어려웠다.

## 9. Step 3 결과 (Basic vs Advanced 비교, 년도 혼동 재진단, 인사이트)

3-1. 다차원 비교

| 구분 | Ragas 5메트릭에서 개선/악화된 차원 |
|------|-----------------------------------|
| 개선 | `Answer Relevancy`: Hybrid + Compression + Rerank(Advanced)가 Basic보다 평균적으로 더 높았고, 특히 `q02`, `q03`, `q08`, `q10`에서 질문에 바로 맞는 문장 형태를 더 안정적으로 생성했다. |
| 악화 | `Context Precision`, `Answer Correctness`: Advanced는 평균 `Context Precision`이 더 낮았고, 최종 `Answer Correctness`도 Basic보다 낮았다. 특히 `q03`, `q08`에서는 precision이 크게 하락했고, `q10`에서는 정답성도 Basic보다 확연히 낮아졌다. |
| 보합 | `Context Recall`, `Faithfulness`: 두 파이프라인 모두 `Context Recall`은 1.0으로 동일했고, `Faithfulness` 평균도 0.5로 같았다. 즉, 핵심 근거를 아예 놓치는 문제는 없었지만, 계산형 문항에서의 faithfulness 보수성은 그대로 남아 있었다. |

- 해석:
이번 파일럿의 다차원 비교 결과를 보면, Advanced는 "답변을 질문에 더 직접적으로 맞추는 능력"에서는 개선이 있었지만, "검색 결과를 깔끔하게 정렬하는 능력"과 "최종 정답성"까지 함께 끌어올리지는 못했다. 즉, retrieval quality 자체를 확실히 개선했다기보다는 generation 표현을 일부 보정한 쪽에 더 가까웠다.

3-2. 년도 혼동 재진단

- 년도 혼동 문항에서 가장 민감하게 반응한 메트릭은 이번 파일럿 기준으로 `Answer Correctness`였다.
- 이유는 `q10`에서 두 파이프라인 모두 `Context Recall = 1.0`, `Context Precision = 1.0`이었음에도 최종 `Answer Correctness`는 `Basic 0.9751`, `Advanced 0.8288`로 차이가 났기 때문이다.
- 즉, 필요한 년도 청크를 검색해 온 것만으로는 충분하지 않았고, 그 청크를 답변에서 정확한 기준 변화로 정리했는지가 더 큰 차이를 만들었다.
- 반면 `q08`처럼 단순 cross-year 비교형에서는 두 파이프라인 모두 `Answer Correctness = 1.0`이어서, retrieval 단계보다 generation 단계 표현 차이가 더 큰 변수로 작용하지 않았다.

- 결론:
Ragas 기본 메트릭만으로도 년도 혼동의 "결과"는 어느 정도 볼 수 있지만, 년도 혼동을 직접 진단하기에는 충분하지 않았다. 특히 `q10`처럼 retrieval 메트릭은 정상인데 최종 답변에서 연도별 기준 설명이 흔들리는 경우, `Context Recall`이나 `Context Precision`만으로는 문제를 명확히 분리하기 어렵다. 따라서 실무적으로 년도별 정확성이 중요한 도메인이라면, `source_year`와 답변의 연도별 서술 일치 여부를 직접 보는 `Year Accuracy` 같은 커스텀 메트릭이 추가로 필요하다고 판단된다.

3-3. 인사이트 정리

4주차의 "Advanced가 더 낫다"는 결론은 이번 5문항 Ragas 결과만 놓고 보면 제한적으로만 유효했다. 현재 README에서 Advanced로 정의한 `Hybrid + Compression + Rerank`는 `Answer Relevancy` 측면에서는 분명 개선을 보였지만, 평균 `Answer Correctness`와 `Context Precision`에서는 Basic보다 우세하지 않았다. 즉, "질문에 바로 맞는 답변처럼 보이게 만드는 효과"는 있었지만, "실제 정답성과 retrieval 품질까지 더 좋다"라고 일반화하기에는 아직 근거가 부족하다.

도메인 임계값을 `Faithfulness ≥ 0.9`처럼 보수적으로 두면, 현재는 Basic도 Advanced도 프로덕션 가능 수준이라고 보기는 어렵다. 물론 이번 데이터셋에는 계산형 문항이 포함되어 있어 built-in `Faithfulness`가 과소평가되는 측면이 있지만, 그 점을 감안하더라도 평균 0.5 수준은 그대로 배포하기엔 불안한 상태다. 특히 의료급여처럼 연도와 조건 차이가 바로 답변 품질에 영향을 주는 도메인에서는, 환각 억제와 연도별 정확성 확인이 더 엄격해야 한다.

개선 우선순위 메트릭은 첫째 `Context Precision`, 둘째 `Year Accuracy(커스텀)`, 셋째 `Faithfulness`라고 판단된다. `Context Precision`은 rerank/압축 조합이 실제로 retrieval 정렬을 개선했는지 가장 직접적으로 보여주고, 현재는 Advanced에서 오히려 평균이 하락했다. `Year Accuracy`는 기본 Ragas로는 잘 드러나지 않는 연도별 기준 혼동을 직접 잡아낼 수 있어 이 도메인에 특히 중요하다. 마지막으로 `Faithfulness`는 계산형 문항 보정 문제를 감안하더라도 장기적으로는 반드시 끌어올려야 할 안전성 지표이므로, 이후에는 커스텀 judge나 프롬프트 개선과 함께 재측정하는 것이 필요하다.

## 10. Step 4 결과 (실패 케이스 Deep Dive)

4-1. 문항 선별

| 케이스 | 선별 기준 | 실제 선택 문항 | 선택 이유 |
|-------|----------|--------------|----------|
| Case A | Advanced가 Basic보다 악화된 문항 | `q10` | `Answer Correctness`가 `Basic 0.9751`, `Advanced 0.8288`로 가장 크게 벌어졌다. |
| Case B | 년도 혼동 발생 또는 혼동 위험이 큰 문항 | `q08` | 최종 답은 맞았지만 cross-year 청크가 섞인 상태에서 `Context Precision`이 크게 흔들려, 년도 혼동 위험을 보여주는 사례였다. |
| Case C | 메트릭 간 충돌 | `q03` | 답은 맞았지만 `Faithfulness = 0`, `Answer Correctness = 0.8854`로 지표 충돌이 명확했다. |

4-2. 케이스별 분석

### Case A: q10

질문: `2025년과 2026년에 달라진 응급의료관리료 100% 본인부담 적용 기준은 무엇인가요?`  
참고 정답: `2025년에는 응급증상 환자가 아닌 경우 응급의료관리료 100% 본인부담이 적용되고, 2026년에는 KTAS 4(경증응급) 및 KTAS 5(비응급) 환자에게 적용되는 것으로 기준이 변경되었습니다.`

[검색된 청크 — Basic]
1. `(2025, p9)` 응급증상 환자가 아닌 경우 응급실 이용 시 의료비 전액 본인부담 관련 설명
2. `(2026, p14)` `응급의료관리료 100분의100 본인부담 기준 변경('26.1.1.)` 표
3. `(2026, p9)` 2026년 본인부담률 일반 표

[검색된 청크 — Advanced]
1. `(2025, p23)` 응급증상 또는 이에 준하는 증상에 해당되지 않으면 의료비 전액 부담
2. `(2026, p14)` 개정 전/후 기준 비교 표 (`응급증상 환자가 아닌 경우 -> KTAS 4/5`)
3. `(2026, p23)` 응급실 이용 시 의료비 전액 부담 설명

[생성된 답변]
- Basic: `2025년 응급의료관리료 100% 본인부담 적용 기준은 응급증상 환자가 아닌 경우였고, 2026년에는 경증응급환자(KTAS 4) 및 비응급환자(KTAS 5)로 변경되었습니다.`
- Advanced: `2025년에는 응급증상 환자가 아닌 경우 응급의료관리료 100% 본인부담 적용 기준이 적용되었고, 2026년에는 경증응급환자(KTAS 4) 및 비응급환자(KTAS 5)도 본인부담 대상에 포함됩니다.`

[메트릭 점수]
- Basic: `Context Recall 1.0`, `Context Precision 1.0`, `Faithfulness 1.0`, `Answer Relevancy 0.5108`, `Answer Correctness 0.9751`
- Advanced: `Context Recall 1.0`, `Context Precision 1.0`, `Faithfulness 0.5`, `Answer Relevancy 0.8879`, `Answer Correctness 0.8288`

[원인 분석]
- 문제 위치: 주로 `생성 단계`
- 해석: 두 파이프라인 모두 필요한 2025/2026 청크는 가져왔고 retrieval 지표도 정상이다. 그런데 Advanced 답변은 표현상 더 자연스럽고 질문 친화적이지만, `기준이 변경되었습니다`라는 reference 문장 구조보다 `본인부담 대상에 포함됩니다` 쪽으로 서술이 이동하면서 `Answer Correctness`와 `Faithfulness`가 더 보수적으로 평가되었다.
- 가장 잘 드러낸 메트릭: `Answer Correctness`
- 조치: cross-year 기준 변화 질문에는 `변경 전/후` 또는 `2025년 기준 / 2026년 기준` 구조를 강제하는 프롬프트가 필요하다. 또한 연도별 기준 서술 일치 여부를 직접 보는 `Year Accuracy` 커스텀 메트릭을 추가하는 것이 적절하다.

### Case B: q08

질문: `2025년과 2026년에 항정신병 장기지속형 주사제 본인부담률은 각각 얼마인가요?`  
참고 정답: `항정신병 장기지속형 주사제 본인부담률은 2025년에는 5%, 2026년에는 2%입니다.`

[검색된 청크 — Basic]
1. `(2025, p12)` 장기지속형 주사제 `5%`
2. `(2025, p10)` 잠복결핵 치료 관련 본인부담 면제
3. `(2026, p12)` 장기지속형 주사제 `2%`
4. `(2026, p10)` 잠복결핵 치료 관련 본인부담 면제

[검색된 청크 — Advanced]
1. `(2025, p12)` 장기지속형 주사제 `5%`
2. `(2026, p12)` 장기지속형 주사제 `2%`
3. `(2026, p12)` 조현병 등 정신질환자 제2·3차 기관 `15%`

[생성된 답변]
- Basic: `항정신병 장기지속형 주사제 본인부담률은 2025년에는 5%, 2026년에는 2%입니다.`
- Advanced: `항정신병 장기지속형 주사제 본인부담률은 2025년에는 5%, 2026년에는 2%입니다.`

[메트릭 점수]
- Basic: `Context Recall 1.0`, `Context Precision 0.3333`, `Faithfulness 0.5`, `Answer Relevancy 0.4290`, `Answer Correctness 1.0`
- Advanced: `Context Recall 1.0`, `Context Precision 0.0`, `Faithfulness 1.0`, `Answer Relevancy 0.8953`, `Answer Correctness 1.0`

[원인 분석]
- 문제 위치: 주로 `검색 단계`, 다만 최종 답변은 우연히 안정적으로 생성됨
- 해석: 두 파이프라인 모두 정답 청크 자체는 찾았지만, cross-year 질문에서 관련 없는 청크가 같이 섞였다. 특히 Advanced는 2026년의 `15%` 정신질환 일반 청크가 함께 들어오면서 `Context Precision`이 0.0까지 떨어졌다. 결과적으로 이번에는 답은 맞았지만, 연도/항목이 비슷한 표 기반 문항에서는 같은 구조가 실제 년도 혼동으로 이어질 위험이 있다.
- 가장 잘 드러낸 메트릭: `Context Precision`
- 조치: cross-year 문항에서는 `항목명 exact match`, `source_year별 balanced retrieval`, `표 헤더 단위 청킹`, `년도별 후처리 필터`가 필요하다.

### Case C: q03

질문: `2025년 2종 수급권자가 협착증으로 복잡추나 치료를 받고 비용이 100,000원 나왔습니다. 본인부담금은?`  
참고 정답: `2025년 2종 수급권자가 협착증으로 복잡추나 치료를 받고 비용이 100,000원 나왔을 경우, 본인부담률 40%가 적용되어 본인부담금은 40,000원입니다.`

[검색된 청크 — Basic]
1. `(2025, p11)` `디스크, 협착증 / 복잡추나 / 2종 / 40%`
2. `(2025, p12)` 의료급여 2종수급권자 본인부담률 일반 표

[검색된 청크 — Advanced]
1. `(2025, p11)` `디스크, 협착증 / 복잡추나 / 2종 / 40%`
2. `(2025, p34)` 종합병원 외래 본인부담 설명
3. `(2025, p9)` 의료급여기관 이용 시 본인부담률 표

[생성된 답변]
- Basic: `2025년 2종 수급권자가 협착증으로 복잡추나 치료를 받고 비용이 100,000원인 경우 본인부담금은 40,000원입니다.`
- Advanced: `2025년 2종 수급권자가 협착증으로 복잡추나 치료를 받고 비용이 100,000원인 경우 본인부담금은 40,000원입니다.`

[메트릭 점수]
- Basic/Advanced 공통: `Context Recall 1.0`, `Faithfulness 0.0`, `Answer Correctness 0.8854`
- 차이: `Answer Relevancy`는 Advanced가 더 높고, `Context Precision`은 Basic이 더 높았다.

[원인 분석]
- 문제 위치: `검색과 생성이 모두 부분적으로 관련`, 하지만 핵심은 `평가 해석 문제`
- 해석: 답변은 사람 기준으로 정답이지만, 검색 문맥에는 `40%`라는 원자료만 있고 `100,000원 -> 40,000원` 계산 결과 문장은 직접 적혀 있지 않다. 그래서 `Answer Correctness`는 높게 나오지만 `Faithfulness`는 0으로 떨어진다.
- 가장 잘 드러낸 메트릭: `Faithfulness`와 `Answer Correctness`의 충돌
- 조치: 계산형 문항은 `Faithfulness` 단독 해석을 피하고, `단순 산술 추론 허용` 커스텀 judge 또는 `Calculation Groundedness`류 보조 메트릭을 추가하는 것이 필요하다.

4-3. 공통 교훈

- cross-year 문항에서는 `Context Recall`이 1.0이어도 실제 답변 품질이 충분히 보장되지 않았다. 필요한 청크를 가져오는 것과, 연도별 기준을 정확히 정리해 답변하는 것은 다른 문제였다.
- `Context Precision`은 년도 혼동 "위험"을 잘 보여줬지만, 실제 최종 답변에서 연도를 틀렸는지까지는 직접 말해주지 못했다.
- 계산형 문항에서는 `Faithfulness`가 실제 품질보다 더 엄격하게 낮아질 수 있었다. 이 경우 `Answer Correctness`와 함께 읽어야 해석이 가능했다.
- 4주차 수동 채점은 최종 정답 여부에는 관대하고 실용적이었고, 5주차 자동 평가는 표현 차이와 근거 직접성까지 보기 때문에 더 엄격했다.
- 따라서 실무에서는 `Ragas 기본 5메트릭 + 도메인 커스텀 메트릭(예: Year Accuracy)` 조합이 더 적절하다고 판단된다.

## 11. 심화 A: Custom Metric — YearAccuracy

구현 목적
- Ragas 기본 메트릭은 `답변이 질문에서 요구한 연도 기준을 정확히 지켰는가`를 직접 판정하지 못했다.
- 특히 cross-year 문항에서는 `Context Recall`이 1.0이어도, `2025 기준`과 `2026 기준`을 답변에서 올바르게 대응했는지는 별도 확인이 필요했다.

초기 시도와 실패 원인
- 처음에는 `question + response + source_year`를 입력으로 받는 LLM judge 방식의 `Year Accuracy`를 만들었다.
- 이 방식은 구현은 쉬웠지만, 실제로는 `ground_truth_contexts` 배열 순서를 보지 못했고, `2025 -> 2026` 대응 관계를 구조적으로 검증하지 못했다.
- 그 결과 `q08`처럼 답변이 명백히 정답인 cross-year 문항도 `year_accuracy = 0`으로 떨어지는 등, 연도 혼동 여부를 안정적으로 판정하지 못했다.
- 즉, `Year Accuracy`는 해석형 judge보다는 규칙형 검증에 더 가까운 문제였는데, 이를 LLM 자유판정에 맡긴 것이 실패 원인이었다.

최종 해결 방식
- 최종적으로는 LLM judge를 버리고 `규칙 기반 Year Accuracy`로 전환했다.
- 현재 구현은 `source_year`, `reference`, `reference_contexts`를 함께 사용한다.
- single-year 문항은 `다른 연도를 섞지 않았는지`와 `해당 연도 핵심 토큰이 답변에 포함되었는지`를 검사한다.
- cross-year 문항은 `source_year`의 순서와 `reference_contexts` 배열 순서를 기준으로 `2025 구간`, `2026 구간`을 나누고, 각 연도에 대응되는 핵심 값이 맞게 들어갔는지를 규칙적으로 비교한다.
- 즉, 이 메트릭의 의미는 결국 `연도 혼동 안 했냐`를 별도로 보는 것이다.

결과 파일
- 실행 파일: `week5/monkana/ragas_eval_with_year_accuracy.py`
- 결과 폴더: `week5/monkana/result/WithYearAcc`

결과 요약
- 이번 `WithYearAcc` 실행은 `ragas_eval.py` 기본 파이프라인 기준으로 수행했으므로 비교 대상은 `Basic` vs `Hybrid only`였다.
- 평균 결과는 다음과 같다.

| 메트릭 | Basic | Hybrid only |
|--------|------:|------------:|
| Context Recall | 1.0000 | 1.0000 |
| Context Precision | 0.7111 | 0.6141 |
| Faithfulness | 0.3000 | 0.4000 |
| Answer Relevancy | 0.9025 | 0.9034 |
| Answer Correctness | 0.8897 | 0.8829 |
| Year Accuracy | 1.0000 | 1.0000 |

- cross-year 핵심 문항인 `q08`, `q10`도 Basic/Hybrid only 모두 `year_accuracy = 1.0`으로 정상 판정되었다.

결론
- `Year Accuracy`를 LLM judge 방식으로 느슨하게 유지하는 것보다, 연도별 대응을 직접 비교하는 규칙 기반 방식이 더 안정적이고 설명 가능했다.
- 특히 이 도메인은 `2025 기준`, `2026 기준`처럼 정답 판정 규칙이 명확하므로, `연도 혼동 여부`는 LLM의 감각적 판정보다 규칙 기반 검증이 더 적합했다.
- 따라서 심화 A의 결론은, `Year Accuracy`는 실제로 유의미한 보조 메트릭이었고, cross-year 문항 품질을 보기 위한 도메인 특화 진단 지표로 채택할 가치가 있다는 것이다.
