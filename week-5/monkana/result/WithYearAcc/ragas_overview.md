# Ragas Result Overview

## Run Setup

- dataset: `/Users/jms/Desktop/project/Ai-agent-PortfolioPage/week5/monkana/golden_dataset_step2_pilot_5.jsonl`
- pipelines: `basic, advanced`
- limit: `all`
- basic_k: `4`
- advanced_top_n: `8`
- hybrid_com_top_n: `8`
- hybrid_com_rerank_top_n: `8`
- korean_prompts: `True`

## Average Scores

| Metric | basic | advanced |
|---|---:|---:|
| Answer Correctness | 0.8897 | 0.8829 |
| Answer Relevancy | 0.9025 | 0.9034 |
| Context Recall | 1.0000 | 1.0000 |
| Faithfulness | 0.3000 | 0.4000 |
| Context Precision | 0.7111 | 0.6141 |
| Year Accuracy | 1.0000 | 1.0000 |

## Per-Question Comparison

| ID | Difficulty | Year | basic Ans | advanced Ans | Ctx Recall basic/advanced | Ctx Precision basic/advanced | Faithfulness basic/advanced | Ans Relevancy basic/advanced | Ans Correctness basic/advanced |
|---|---|---|---|---|---|---|---|---|---|
| q02 | easy | 2026 | 2026년 의료급여 65세 이상 2종 ... | 2026년 의료급여 65세 이상 2종 ... | 1.0000/1.0000 | 0.7500/0.6429 | 1.0000/1.0000 | 0.9822/0.9826 | 1.0000/1.0000 |
| q03 | medium | 2025 | 2025년 2종 수급권자가 협착증으로 ... | 2025년 2종 수급권자가 협착증으로 ... | 1.0000/1.0000 | 1.0000/0.8333 | 0.0000/0.0000 | 0.8824/0.8826 | 0.8854/0.8050 |
| q05 | hard | 2025 | 2025년 2종 수급권자가 조현병으로 ... | 2025년 2종 수급권자가 조현병으로 ... | 1.0000/1.0000 | 1.0000/0.9029 | 0.0000/0.0000 | 0.8642/0.8642 | 0.7380/0.8828 |
| q08 | cross-year | 2025,2026 | 항정신병 장기지속형 주사제 본인부담률은... | 항정신병 장기지속형 주사제 본인부담률은... | 1.0000/1.0000 | 0.0000/0.0000 | 0.0000/0.5000 | 0.8953/0.8960 | 1.0000/1.0000 |
| q10 | cross-year | 2025,2026 | 2025년 응급의료관리료 100% 본인... | 2025년에는 응급의료관리료 100% ... | 1.0000/1.0000 | 0.8056/0.6917 | 0.5000/0.5000 | 0.8885/0.8918 | 0.8251/0.7268 |
