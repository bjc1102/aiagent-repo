# Ragas Result Overview

## Run Setup

- dataset: `/Users/jms/Desktop/project/Ai-agent-PortfolioPage/week5/monkana/golden_dataset_step2_pilot_5.jsonl`
- pipelines: `basic, advanced`
- limit: `all`
- basic_k: `4`
- advanced_top_n: `8`
- korean_prompts: `False`

## Average Scores

| Metric | Basic | Advanced | Delta |
|---|---:|---:|---:|
| Answer Correctness | 0.9487 | 0.9364 | -0.0123 |
| Answer Relevancy | 0.4744 | 0.4800 | 0.0056 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Faithfulness | 0.4800 | 0.8000 | 0.3200 |
| Context Precision | 0.8167 | 0.7093 | -0.1074 |

## Per-Question Comparison

| ID | Difficulty | Year | Basic Ans | Advanced Ans | Ctx Recall B/A | Ctx Precision B/A | Faithfulness B/A | Ans Relevancy B/A | Ans Correctness B/A |
|---|---|---|---|---|---|---|---|---|---|
| q02 | easy | 2026 | 2026년 의료급여 65세 이상 2종 ... | 2026년 의료급여 65세 이상 2종 ... | 1.0000/1.0000 | 0.7500/0.6429 | 1.0000/1.0000 | 0.4467/0.4466 | 1.0000/1.0000 |
| q03 | medium | 2025 | 2025년 2종 수급권자가 협착증으로 ... | 2025년 2종 수급권자가 협착증으로 ... | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000/0.0000 | 0.4357/0.4344 | 0.8081/0.8541 |
| q05 | hard | 2025 | 2025년 2종 수급권자가 조현병으로 ... | 2025년 2종 수급권자이자 조현병 환... | 1.0000/1.0000 | 1.0000/0.8774 | 0.4000/1.0000 | 0.5752/0.5639 | 0.9823/0.9933 |
| q08 | cross-year | 2025,2026 | 2025년 항정신병 장기지속형 주사제 ... | 2025년 항정신병 장기지속형 주사제 ... | 1.0000/1.0000 | 0.3333/0.2000 | 0.5000/1.0000 | 0.4115/0.4290 | 0.9767/0.9767 |
| q10 | cross-year | 2025,2026 | 2025년에는 응급의료관리료 100% ... | 2025년과 2026년에 달라진 응급의... | 1.0000/1.0000 | 1.0000/0.8260 | 0.5000/1.0000 | 0.5028/0.5262 | 0.9764/0.8580 |
