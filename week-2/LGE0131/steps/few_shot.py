import json
import time
import concurrent.futures
from pydantic import BaseModel, Field

# 1. 스키마 (Step 1과 동일)
class MedAidResponse(BaseModel):
    id: str = Field(description="입력받은 질문의 id")
    answer: str = Field(description="최종 본인부담률")
    reason: str = Field(description="정답을 도출한 이유")

class BatchResponse(BaseModel):
    results: list[MedAidResponse]

# 2. 배치 처리 함수 (Step 1과 완전히 동일)
def process_batch(client, img_file, prompt, batch):
    batch_input_str = json.dumps(batch, ensure_ascii=False)
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, img_file, f"다음 환자 목록의 본인부담률을 각각 계산하세요:\n{batch_input_str}"],
            config={
                "response_mime_type": "application/json",
                "response_schema": BatchResponse,
                "temperature": 0.5, # Few-shot은 일관성이 중요하므로 temperature를 살짝 낮춥니다 (0.3~0.5 추천)
            }
        )
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        result_data = json.loads(response.text) if response.text else {"results": []}
        return result_data["results"], prompt_tokens, completion_tokens
    except Exception as e:
        print(f"배치 오류 발생: {e}")
        return [], 0, 0

# 3. 메인 실행 함수
def run(client, img_file, questions_data, batch_size=5): # answer_key를 받아서 동적으로 예시를 만들 수도 있습니다.
    print(f"\n--- Step 2: Few-shot (Batch Size: {batch_size}, 병렬 처리) 실행 중 ---")
    
    # ★ 변경 포인트 1: 프롬프트에 Few-shot 예시 추가
    prompt = """당신은 의료급여 전문가입니다. 첨부된 이미지를 참조하여 다음 환자들의 본인부담률을 정확히 계산하세요.
아래의 예시를 참고하여 답변을 작성해 주세요.

[예시 1]
질문: 65세 이상 1종 수급권자가 틀니를 하면 본인부담률은 몇 %인가요?
answer: 5%
reason: 04번 표를 보면 1종 수급권자의 틀니 본인부담률은 5%로 명시되어 있습니다.

[예시 2]
질문: 15세 이하 1종 수급권자인 4세 아동이 입원하면 본인부담률은 몇 %인가요?
answer: 무료
reason: 15세 이하 아동 입원 조건에서 6세 미만은 본인부담금이 면제(무료)입니다.
"""
    
    # 이 아래로는 Step 1의 병렬 처리 로직(Thread 풀)과 100% 동일하게 복사/붙여넣기 하시면 됩니다.
    all_predictions = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()
    
    batches = [questions_data[i : i + batch_size] for i in range(0, len(questions_data), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batches)) as executor:
        future_to_batch = {executor.submit(process_batch, client, img_file, prompt, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                results, p_tokens, c_tokens = future.result()
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                
                question_map = {item["id"]: item["question"] for item in batch}
                difficulty_map = {item["id"]: item.get("difficulty", "N/A") for item in batch}
                
                for res in results:
                    res["question"] = question_map.get(res["id"], "질문 매핑 실패")
                    res["difficulty"] = difficulty_map.get(res["id"], "N/A")
                    all_predictions.append(res)
            except Exception as e:
                print(f"결과 수집 중 오류: {e}")

    elapsed_time = time.time() - start_time
    metrics = {
        "elapsed_time_sec": round(elapsed_time, 2),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens
    }
            
    return all_predictions, metrics