# Week 5 — RAG 파이프라인 비교 평가 SPEC

5주차 과제: 4주차 RAG 를 재사용하여 Ragas 기반 정량 평가 파이프라인을 구축한다.
각 검색 기법의 기여도를 **단일 변수 원칙**(한 번에 한 요소만 추가)으로 분리 측정한다.

---

## 1. 버전 구성 (5개)

| 폴더 | 포트 | 파이프라인 | 해석 포인트 |
|------|-----|----------|-----------|
| `basic/` | 8000 | Dense | 베이스라인 |
| `hybrid/` | 8001 | Dense + BM25 + RRF | BM25(Sparse) 추가 기여도 |
| `rerank/` | 8002 | Dense + BM25 + Rerank(Cohere) | Cohere Rerank 기여도 |
| `metadata/` | 8003 | Dense + Year Pre-filter | Pre-filter 단독 기여도 |
| `metadata-full/` | 8004 | Dense + BM25 + Rerank + Pre-filter | 프로덕션 최종 파이프라인 |

### 비교 축

```
baseline ───────────────────────► basic
  +BM25 ────────────────────────► hybrid   (basic vs hybrid: BM25 효과)
  +Rerank ──────────────────────► rerank   (hybrid vs rerank: Rerank 효과)
  +Pre-filter ──────────────────► metadata (basic vs metadata: 필터 단독 효과)
  +BM25+Rerank+Pre-filter ──────► metadata-full (최종, 누적 효과)
```

---

## 2. 공통 구조

모든 버전은 동일한 레이아웃을 따른다.

```
<version>/
├── .env                    # API 키 (폴더 기준 자동 로드, gitignore)
├── .env.example            # 템플릿
├── .gitignore
├── requirements.txt
├── main.py                 # FastAPI 엔트리 (uvicorn)
├── data/                   # PDF, golden_dataset_v2.jsonl
└── src/
    ├── core/config.py      # pydantic-settings + dotenv (BASE_DIR/.env 우선)
    ├── domains/rag/
    │   ├── models.py       # 도메인 엔터티
    │   ├── schemas.py      # 요청/응답 Pydantic
    │   ├── service.py      # 핵심 RAG 로직
    │   └── router.py       # /api/v1/rag/{index,query,evaluate}
    └── utils/
        ├── pdf_parser.py   # PDF → 텍스트 + 표(markdown)
        └── year_extractor.py  # metadata 계열만
```

### 공통 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 메타 정보 (모델명 등) |
| GET | `/health` | 헬스체크 |
| POST | `/api/v1/rag/index` | PDF → 벡터 DB 재생성 (파괴적) |
| POST | `/api/v1/rag/query` | 단일 질의 응답 |
| POST | `/api/v1/rag/evaluate` | golden_dataset_v2.jsonl 전체 평가 → JSONL 저장 |

### 공통 응답 포맷

```json
{
  "answer": "...",
  "retrieved_contexts": ["chunk1 text", "chunk2 text", "..."],
  "sources": [{ "content": "...", "metadata": {} }],
  "filter_info": { }
}
```

- `retrieved_contexts`: Ragas `SingleTurnSample.retrieved_contexts` 로 직접 매핑
- `sources`: 출처 메타데이터 포함 (UI·디버그용)
- `filter_info`: metadata / metadata-full 에만 존재

---

## 3. 버전별 핵심 차이

### 3-1. basic
- ChromaDB `similarity_search(k=10)` 단독
- 중복 제거·재정렬 로직 없음

### 3-2. hybrid
- Dense Top 20 + BM25 Top 20 → **RRF** (`score = 1 / (rank + 60)`)
- BM25 한국어 토크나이저: `kiwipiepy` (형태소)
- **키**: `chunk_id` (다년도 청크 병합 방지)

### 3-3. rerank
- Dense Top 20 + BM25 Top 20 → union → **Cohere rerank-multilingual-v3.0** → Top 10
- Cohere 미사용·실패 시: RRF 폴백
- **키**: `chunk_id`

### 3-4. metadata (옵션 1: Basic + Pre-filter)
- 질문에서 `YearExtractor` 로 년도 추출
- `similarity_search(filter={"source_year": ...})` 로 **검색 전 단계 필터**
- 결과가 `k/2` 미만이면 필터 해제 후 재검색 (fallback)
- BM25·Rerank 없음 — 필터 단독 기여도 분리 측정용

### 3-5. metadata-full (옵션 2: 풀 파이프라인 + Pre-filter)
- metadata + BM25(년도별 corpus 로 제한 후 재계산) + Cohere Rerank
- Rerank 실패 시 `chunk_id` 기반 RRF 폴백

---

## 4. chunk_id 설계 (공통)

### 문제
`page_content` 를 문서 식별자로 쓰면, 2025/2026 같은 연례 개정본에서 동일 문장이 있을 때 **두 년도 청크가 한 항목으로 병합**되어버림. 년도 혼동 분석이 오염됨.

### 해결
청킹 시점에 고유 ID 부여:

```python
chunk.metadata["chunk_id"] = f"{source}__p{page}__s{start_index}"
# 예: "2025.pdf__p15__s0"
```

- `source`: PDF 파일명 (년도 구분)
- `page`: 페이지 번호
- `start_index`: 페이지 내 청크 시작 위치 (`RecursiveCharacterTextSplitter(add_start_index=True)`)

### 적용 지점
- RRF 점수 누적 키
- Rerank 중복 제거 세트
- 모든 `Document` 병합·정렬 연산

---

## 5. YearExtractor 로직 (metadata / metadata-full 공통)

### 추출 규칙
1. **명시적 년도**: `r"202\d"` regex
2. **상대 표현**: `REFERENCE_YEAR`(`.env`에서 설정, 기본 2026) 기준 환산
   - 작년·지난해·전년 → `-1`
   - 올해·이번 해·금년·당해 → `0`
   - 내년·다음 해·신년 → `+1`
3. **Cross-year 플래그**: `대비`, `변화`, `달라진`, `차이`, `이전`, `신설`, `개정`, `변동`, `증감`, `인상`, `인하` 중 하나라도 포함되면 ON
4. 복수 년도 감지 시 자동 cross-year ON

### Chroma filter 매핑

```python
# 단일 년도
{"source_year": {"$eq": "2025"}}

# 복수 년도 (cross-year)
{"source_year": {"$in": ["2025", "2026"]}}

# 년도 없음
None  # 필터 미적용
```

### Fallback
- 필터 결과 < `k * 0.5` → 필터 해제 후 재검색
- `filter_info.fallback = true` 로 기록

---

## 6. 평가 산출물

### 입력
`data/golden_dataset_v2.jsonl` (각 버전 공통)

필수 필드:
```json
{
  "question": "...",
  "ground_truth": "...",
  "ground_truth_contexts": ["...", "..."],
  "difficulty": "easy|medium|hard|cross-year",
  "source_year": "2025|2026|2025+2026"
}
```

### 출력
`data/<version>/<index>/evaluation_results.jsonl`

각 레코드:
```json
{
  "question": "...",
  "ground_truth": "...",
  "ground_truth_contexts": ["..."],
  "response": "...",
  "retrieved_contexts": ["..."],
  "difficulty": "...",
  "source_year": "...",
  "filter_info": {}
}
```

### Ragas 필드 매핑

| JSONL | Ragas `SingleTurnSample` |
|-------|--------------------------|
| `question` | `user_input` |
| `response` | `response` |
| `retrieved_contexts` | `retrieved_contexts` |
| `ground_truth` | `reference` |
| `ground_truth_contexts` | `reference_contexts` |

---

## 7. 측정 메트릭

Ragas 0.2+ 클래스형:

| 메트릭 | 목적 |
|--------|------|
| `ContextRecall()` | 정답 근거를 얼마나 찾았나 |
| `LLMContextPrecisionWithReference()` | 검색 결과의 정확도 |
| `Faithfulness()` | 답변이 컨텍스트를 벗어났나 (환각) |
| `ResponseRelevancy()` | 답변이 질문에 들어맞나 |
| `AnswerCorrectness()` | 최종 정확도 (의미+사실 결합) |

- 평가용 LLM: Claude Sonnet 4.5 권장 (생성 LLM 이 Gemini 이므로 패밀리 분리)
- 평가용 임베딩: `text-embedding-3-small`
- 한국어 프롬프트: `adapt_prompts(language="korean", llm=evaluator_llm)` → `set_prompts(**adapted)`

---

## 8. 실행 순서

```bash
# 각 버전 별 (basic 예시, 다른 버전도 동일 패턴)
cd week-5/basic
pip install -r requirements.txt
cp .env.example .env        # API 키 채우기
cp ../data/*.pdf data/      # PDF 배치
python main.py              # 포트 확인

# 서버 구동 중 다른 터미널에서:
curl -X POST http://localhost:8000/api/v1/rag/index
curl -X POST http://localhost:8000/api/v1/rag/evaluate
```

모든 버전 JSONL 생성 후 → `week-5/evaluate_ragas.py` (별도 구축 예정) 에서 일괄 채점.

---

## 9. 주의사항

### 재인덱싱 필요 시점
- `chunk_id` 로직 변경 후 (현재 반영 완료)
- 청킹 파라미터(`chunk_size`, `chunk_overlap`) 변경 시
- PDF 추가·교체 시

`run_indexing()` 은 `delete_collection()` 먼저 호출 → 기존 컬렉션 파괴 후 재생성.

### 포트 충돌
5개 서버를 **동시에** 기동 가능하도록 포트 분리 (8000~8004). 한 번에 하나만 돌릴 예정이면 무시 가능.

### Cohere Rate Limit
- 무료 Trial key: 10 req/min
- 20문항 평가 시 약 2분 소요. 실패 시 자동 RRF 폴백.

### `.env` 로드 방식
`src/core/config.py` 에서 `BASE_DIR / ".env"` 를 우선 검색 (`load_dotenv(override=True)`).
→ 각 버전 폴더에 독립된 `.env` 배치 가능. 시스템 환경변수와 섞이지 않음.

---

## 10. 체크리스트

- [x] 5개 폴더 구조 생성
- [x] 공통 FastAPI 라우팅
- [x] chunk_id 기반 중복 제거 (hybrid/rerank/metadata-full)
- [x] YearExtractor 규칙 기반 구현
- [x] Pre-retrieval filter (Chroma `$eq`/`$in`)
- [x] 필터 결과 부족 시 fallback
- [ ] `golden_dataset_v2.jsonl` 작성 (10~15 문항)
- [ ] 각 버전 `/index` + `/evaluate` 실행
- [ ] `evaluate_ragas.py` 스크립트 작성
- [ ] 5버전 결과 비교 표 + 인사이트 정리 (README.md)
