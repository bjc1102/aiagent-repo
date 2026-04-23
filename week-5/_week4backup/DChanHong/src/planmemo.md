# RAG 성능 테스트 시나리오 계획 (Test Scenarios)

이 문서는 Basic RAG에서 Advanced RAG로 넘어가는 과정에서 최적의 파라미터(k, n, Top-N)를 찾기 위한 실험 계획서입니다.

---

### CASE A: Small & Sharp
- **Dense (k)**: 10
- **Sparse (n)**: 10
- **Rerank (Top-N)**: 5
- **실험 목적**: 최소한의 자원(후보군 20개)으로 정답률이 유지되는지 확인하며, 비용과 속도 측면의 최적화 지점을 찾습니다.

### CASE B: Semantic Heavy
- **Dense (k)**: 30
- **Sparse (n)**: 10
- **Rerank (Top-N)**: 10
- **실험 목적**: 질문의 의미와 맥락이 복잡한 경우(Hard 난이도), 벡터 검색의 비중을 높였을 때 성능이 얼마나 개선되는지 확인합니다.

### CASE C: Keyword Heavy
- **Dense (k)**: 10
- **Sparse (n)**: 30
- **Rerank (Top-N)**: 10
- **실험 목적**: "본인부담금", "365회", "특정 질병명" 등 고유 명사나 수치가 포함된 질문에서 키워드 매칭(BM25)의 기여도를 확인합니다.

### CASE D: High Recall
- **Dense (k)**: 50
- **Sparse (n)**: 50
- **Rerank (Top-N)**: 15
- **실험 목적**: 후보군을 100개까지 대폭 늘려 리랭커(Reranker)에게 풍부한 재료를 제공했을 때, 리랭커의 정밀 분류 능력이 극대화되는지 테스트합니다.

### CASE E: Asymmetric Hybrid
- **Dense (k)**: 25
- **Sparse (n)**: 15
- **Rerank (Top-N)**: 10
- **실험 목적**: 현재의 균형(20:20)보다 약간 더 의미 중심(Dense)으로 튜닝했을 때, 다년도 문서 간의 미세한 차이를 더 잘 잡아내는지 확인합니다.
