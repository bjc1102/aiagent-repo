import json
import time
import concurrent.futures
from pydantic import BaseModel, Field


class CoTResponse(BaseModel):
    id: str = Field(description="입력받은 질문의 id")
    reasoning_process: str = Field(description="1종/2종 여부, 의료기관 종류, 특이조건 등을 단계별로 분석한 논리적 사고 과정")
    answer: str = Field(description="최종 본인부담률")


# ★ 변경 포인트 2: CoT 배치 응답 스키마
class CoTBatchResponse(BaseModel):
    results: list[CoTResponse]

def process_batch(client, img_file, prompt, batch):
    batch_input_str = json.dumps(batch, ensure_ascii=False)
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, img_file, f"다음 환자 목록의 본인부담률을 각각 계산하세요:\n{batch_input_str}"],
            config={
                "response_mime_type": "application/json",
                "response_schema": CoTBatchResponse, # ★ 변경 포인트 3: CoT 전용 스키마 적용
                "temperature": 0.5,
            }
        )
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        result_data = json.loads(response.text) if response.text else {"results": []}
        return result_data["results"], prompt_tokens, completion_tokens
    except Exception as e:
        print(f"배치 오류 발생: {e}")
        return [], 0, 0

def run(client, img_file, questions_data, batch_size=3):
    print(f"\n--- Step 3: Chain of Thought (Batch Size: {batch_size}, 병렬 처리) 실행 중 ---")
    
    # ★ 변경 포인트 4: 단계별 사고를 유도하는 프롬프트
    prompt = """당신은 의료급여 전문가입니다. 첨부된 이미지를 참조하여 환자들의 본인부담률을 정확히 계산하세요.
단, 최종 결론을 내리기 전에 반드시 다음 단계를 거쳐 논리적으로 생각하세요(Chain of Thought):
1. 환자가 1종인지 2종인지 수급권자 유형을 먼저 파악합니다.
2. 진료받는 곳이 어떤 의료기관(1차, 2차, 3차, 약국 등)인지 확인합니다.
3. 나이, 질환(중증질환, 임산부 등) 등 예외 조건이 있는지 확인합니다.
4. 표에 맞춰 최종 본인부담률을 도출합니다."""
    
    # 이 아래로는 Step 1, 2와 완전히 동일한 병렬 처리 및 metrics 계산 로직입니다.
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