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
| Answer Correctness | 0.9487 | 0.9492 | 0.0005 |
| Answer Relevancy | 0.4769 | 0.4726 | -0.0044 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |
| Faithfulness | 0.5000 | 0.6000 | 0.1000 |
| Context Precision | 0.8167 | 0.7491 | -0.0675 |

## Per-Question Comparison

| ID | Difficulty | Year | Basic Ans | Advanced Ans | Ctx Recall B/A | Ctx Precision B/A | Faithfulness B/A | Ans Relevancy B/A | Ans Correctness B/A |
|---|---|---|---|---|---|---|---|---|---|
| q02 | easy | 2026 | 2026년 의료급여 65세 이상 2종 ... | 2026년 의료급여 65세 이상 2종 ... | 1.0000/1.0000 | 0.7500/0.6429 | 1.0000/1.0000 | 0.4466/0.4466 | 1.0000/1.0000 |
| q03 | medium | 2025 | 2025년 2종 수급권자가 협착증으로 ... | 2025년 2종 수급권자가 협착증으로 ... | 1.0000/1.0000 | 1.0000/1.0000 | 0.0000/0.0000 | 0.4381/0.4381 | 0.8854/0.8854 |
| q05 | hard | 2025 | 2025년 2종 수급권자가 조현병으로 ... | 2025년 2종 수급권자가 조현병으로 ... | 1.0000/1.0000 | 1.0000/0.9029 | 0.0000/0.0000 | 0.5600/0.5600 | 0.8828/0.8828 |
| q08 | cross-year | 2025,2026 | 항정신병 장기지속형 주사제 본인부담률은... | 항정신병 장기지속형 주사제 본인부담률은... | 1.0000/1.0000 | 0.3333/0.2000 | 0.5000/1.0000 | 0.4290/0.4290 | 1.0000/1.0000 |
| q10 | cross-year | 2025,2026 | 2025년 응급의료관리료 100% 본인... | 2025년에는 응급증상 환자가 아닌 경... | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 0.5108/0.4891 | 0.9751/0.9777 |
