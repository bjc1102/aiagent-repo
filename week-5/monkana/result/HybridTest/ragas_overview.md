# Ragas Result Overview

## Run Setup

- dataset: `/Users/jms/Desktop/project/Ai-agent-PortfolioPage/week5/monkana/golden_dataset_step2_pilot_5.jsonl`
- pipelines: `hybrid_com, hybrid_com_rerank`
- limit: `all`
- basic_k: `4`
- advanced_top_n: `8`
- hybrid_com_top_n: `8`
- hybrid_com_rerank_top_n: `8`
- korean_prompts: `True`

## Average Scores

| Metric | hybrid_com | hybrid_com_rerank |
|---|---:|---:|
| Answer Correctness | 0.8953 | 0.9194 |
| Answer Relevancy | 0.8449 | 0.9007 |
| Context Recall | 1.0000 | 1.0000 |
| Faithfulness | 0.5000 | 0.5000 |
| Context Precision | 0.7586 | 0.6300 |

## Per-Question Comparison

| ID | Difficulty | Year | hybrid_com Ans | hybrid_com_rerank Ans | Ctx Recall hybrid_com/hybrid_com_rerank | Ctx Precision hybrid_com/hybrid_com_rerank | Faithfulness hybrid_com/hybrid_com_rerank | Ans Relevancy hybrid_com/hybrid_com_rerank | Ans Correctness hybrid_com/hybrid_com_rerank |
|---|---|---|---|---|---|---|---|---|---|
| q02 | easy | 2026 | 2026년 의료급여 65세 이상 2종 ... | 2026년 의료급여 65세 이상 2종 ... | 1.0000/1.0000 | 1.0000/0.9167 | 1.0000/1.0000 | 0.9826/0.9826 | 1.0000/1.0000 |
| q03 | medium | 2025 | 2025년 2종 수급권자가 협착증으로 ... | 2025년 2종 수급권자가 협착증으로 ... | 1.0000/1.0000 | 1.0000/0.5000 | 0.0000/0.0000 | 0.8824/0.8824 | 0.8050/0.8854 |
| q05 | hard | 2025 | 2025년 2종 수급권자가 조현병으로 ... | 2025년 2종 수급권자가 조현병으로 ... | 1.0000/1.0000 | 0.7929/0.7333 | 0.0000/0.0000 | 0.8553/0.8553 | 0.8828/0.8828 |
| q08 | cross-year | 2025,2026 | 항정신병 장기지속형 주사제 본인부담률은... | 항정신병 장기지속형 주사제 본인부담률은... | 1.0000/1.0000 | 0.0000/0.0000 | 1.0000/1.0000 | 0.8953/0.8953 | 1.0000/1.0000 |
| q10 | cross-year | 2025,2026 | 2025년에는 응급증상 환자가 아닌 경... | 2025년에는 응급증상 환자가 아닌 경... | 1.0000/1.0000 | 1.0000/1.0000 | 0.5000/0.5000 | 0.6090/0.8879 | 0.7887/0.8288 |
