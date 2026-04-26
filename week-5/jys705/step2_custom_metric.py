import pandas as pd
import re
from dataclasses import dataclass
from ragas import evaluate
from ragas.metrics.base import SingleTurnMetric
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig

# ==========================================
# [심화] 커스텀 메트릭 구현: YearAccuracy
# ==========================================
@dataclass
class YearAccuracy(SingleTurnMetric):
    """
    질문에 포함된 연도(2025 또는 2026)가 답변에 정확히 반영되었는지, 
    다른 연도와 혼동하지 않았는지를 0~1점으로 평가하는 커스텀 메트릭입니다.
    """
    name: str = "year_accuracy"
    
    # 수정 1: async 제거 (Warning 해결)
    def init(self, run_config: RunConfig):
        pas
    
    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        q = sample.user_input
        ans = sample.response
        
        # 1. 질문과 답변에서 타겟 연도 추출
        q_years = set(re.findall(r'(2025|2026)', q))
        ans_years = set(re.findall(r'(2025|2026)', ans))
        
        # 2. 질문에 연도를 묻는 내용이 없다면 평가 면제 (만점)
        if not q_years:
            return 1.0
            
        # 3. 모델이 답변을 거부한 경우 (0점)
        if "정보를 찾을 수 없습니다" in ans:
            return 0.0
            
        # 4. 연도 혼동 로직 판별
        if q_years.issubset(ans_years):
            # 질문의 연도가 모두 답변에 들어있을 때
            if not ans_years.issubset(q_years):
                # 질문에 없는 엉뚱한 연도가 답변에 섞여 들어간 경우 (연도 혼용)
                return 0.5 
            # 질문과 답변의 연도가 완벽히 일치 (정상)
            return 1.0
        else:
            # 질문의 연도가 답변에 아예 빠진 경우 (심각한 년도 혼동)
            return 0.0

def create_dataset_from_csv(csv_path):
    """기존에 저장된 CSV를 Ragas 데이터셋으로 역변환"""
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        samples.append(SingleTurnSample(
            user_input=row['user_input'],
            response=row['response'],
            retrieved_contexts=eval(row['retrieved_contexts']) if isinstance(row['retrieved_contexts'], str) else [],
            reference=row['reference']
        ))
    return EvaluationDataset(samples=samples)

def run_custom_metric():
    print("기존 CSV 결과물을 불러와 [YearAccuracy] 커스텀 평가를 시작합니다...\n")
    
    # 데이터셋 로드
    basic_ds = create_dataset_from_csv("basic_ragas_scores.csv")
    adv_ds = create_dataset_from_csv("advanced_ragas_scores.csv")
    
    print("Basic RAG 연도 혼동 채점 중...")
    basic_result = evaluate(dataset=basic_ds, metrics=[YearAccuracy()])
    
    print("Advanced RAG 연도 혼동 채점 중...")
    adv_result = evaluate(dataset=adv_ds, metrics=[YearAccuracy()])
    
    # 수정 2: Pandas Dataframe으로 변환 후 확실하게 평균(mean) 계산
    basic_score = basic_result.to_pandas()['year_accuracy'].mean()
    adv_score = adv_result.to_pandas()['year_accuracy'].mean()
    
    print("\n" + "="*50)
    print("[심화] Year Accuracy (연도 정확도) 측정 결과")
    print("="*50)
    print(f"- Basic RAG Year Accuracy    : {basic_score:.3f}")
    print(f"- Advanced RAG Year Accuracy : {adv_score:.3f}")
    print("="*50)
    
    if adv_score > basic_score:
        print("분석: Advanced RAG의 메타데이터 필터링이 연도 혼동 방지에 큰 기여를 했음이 커스텀 메트릭으로 증명되었습니다.")
    else:
        print("분석: Advanced RAG 역시 연도 혼동 문제(또는 답변 거부)에서 완벽히 자유롭지 못했습니다.")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    run_custom_metric()