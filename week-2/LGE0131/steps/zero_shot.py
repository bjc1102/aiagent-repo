import json
import time
import random
from pydantic import BaseModel, Field

# 개별 응답 스키마
class MedAidResponse(BaseModel):
    id: str = Field(description="입력받은 질문의 id (원치 않게 순서가 섞일 경우를 대비)")
    answer: str = Field(description="최종 본인부담률")
    reason: str = Field(description="정답을 도출한 이유")

# 배치 응답 스키마
class BatchResponse(BaseModel):
    results: list[MedAidResponse]


def run(client, img_file, questions_data, batch_size=5):
    """Step 1: Zero-shot (배치 처리 적용)"""
    print(f"\n--- Step 1: Zero-shot (Batch Size: {batch_size}) 실행 중 ---")
    
    prompt = "당신은 의료급여 전문가입니다. 첨부된 이미지를 참조하여 다음 여러 환자들의 본인부담률을 정확히 계산하세요."

    all_predictions = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()
    
    for i in range(0, len(questions_data), batch_size):
        batch = questions_data[i : i + batch_size]
        
        batch_input_str = json.dumps(batch, ensure_ascii=False)
        print(f"{i+1} ~ {min(i+batch_size, len(questions_data))}번 문항 처리 중...")
        
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt, 
                    img_file, 
                    f"다음 환자 목록의 본인부담률을 각각 계산하세요:\n{batch_input_str}"
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": BatchResponse,
                    "temperature": 0.5,
                }
            )

            if response.usage_metadata:
                total_prompt_tokens += response.usage_metadata.prompt_token_count
                total_completion_tokens += response.usage_metadata.candidates_token_count
            
            if response.text:
                result_data = json.loads(response.text)

                question_map = {item["id"]: item["question"] for item in batch}
                difficulty_map = {item["id"]: item.get("difficulty", "N/A") for item in batch}


                for res in result_data["results"]:
                    res["question"] = question_map.get(res["id"], "질문 매핑 실패")
                    res["difficulty"] = difficulty_map.get(res["id"], "N/A")
                    all_predictions.append(res)
            else:
                print(f"배치 오류 발생: 응답 없음")
                
        except Exception as e:
            print(f"배치 오류 발생: {e}")

    elapsed_time = time.time() - start_time

    metrics = {
        "elapsed_time_sec": round(elapsed_time, 2),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens
    }
            
    return all_predictions, metrics