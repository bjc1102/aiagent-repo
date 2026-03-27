import json
import time
import concurrent.futures
from collections import Counter
from pydantic import BaseModel, Field
from tqdm import tqdm

# 1. 스키마 정의 (CoT 방식을 사용하여 논리적 다양성 확보)
class SCResponse(BaseModel):
    id: str = Field(description="입력받은 질문의 id")
    reasoning_process: str = Field(description="본인부담률을 도출하기 위한 논리적 사고 과정")
    answer: str = Field(description="최종 본인부담률")

class SCBatchResponse(BaseModel):
    results: list[SCResponse]

# 2. 개별 배치 처리 함수
def process_batch(client, img_file, prompt, batch, temperature):
    batch_input_str = json.dumps(batch, ensure_ascii=False)
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[prompt, img_file, f"다음 환자 목록의 본인부담률을 각각 계산하세요:\n{batch_input_str}"],
            config={
                "response_mime_type": "application/json",
                "response_schema": SCBatchResponse,
                # SC는 다양한 답변(경로)을 유도해야 하므로 온도를 살짝 높입니다. (0.7 추천)
                "temperature": temperature, 
            }
        )
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        result_data = json.loads(response.text) if response.text else {"results": []}
        return result_data["results"], prompt_tokens, completion_tokens
    except Exception as e:
        print(f"배치 오류 발생: {e}")
        return [], 0, 0

# 3. 메인 실행 함수 (다수결 투표 로직 포함)
def run(client, img_file, questions_data, batch_size=3, num_paths=3):
    """
    num_paths: 동일한 질문을 몇 번 반복해서 물어볼 것인지 (기본값 3번)
    """
    print(f"\n--- Step 4: Self-Consistency (Batch: {batch_size}, 반복: {num_paths}회) 실행 중 ---")
    
    prompt = """당신은 의료급여 전문가입니다. 첨부된 이미지를 참조하여 환자들의 본인부담률을 정확히 계산하세요.
답을 내리기 전, 환자의 수급권자 유형, 의료기관 종류, 특이사항 등을 종합적으로 고려하여 'reasoning_process'에 사고 과정을 상세히 적으세요."""

    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()
    
    # 1. 원본 데이터를 배치 크기만큼 쪼개기
    batches = [questions_data[i : i + batch_size] for i in range(0, len(questions_data), batch_size)]
    
    # 2. 작업을 num_paths(3배) 만큼 부풀리기
    # 예: 배치가 10개면, 3번씩 반복해야 하므로 총 30개의 작업(Task)이 생성됨
    tasks = [(batch, path) for batch in batches for path in range(num_paths)]
    
    all_raw_results = [] # 3번씩 물어본 모든 날것의 답변을 담을 리스트
    
    # 3. 병렬 처리로 3배 늘어난 작업 싹 다 돌리기
    # 워커(max_workers)를 넉넉히 주어 속도를 확보합니다. (너무 많으면 API 에러가 날 수 있으니 10~15 내외 권장)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {}

        for batch, path in tasks:
            time.sleep(0.5) # API 멱살 잡지 않게 숨 고르기
            future = executor.submit(process_batch, client, img_file, prompt, batch, temperature=0.7)
            future_to_task[future] = batch
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="[Step 4] 다수결 데이터 수집"):
            batch = future_to_task[future]
            try:
                results, p_tokens, c_tokens = future.result()
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                all_raw_results.extend(results)
            except Exception as e:
                print(f"결과 수집 중 오류: {e}")

    # 4. ID별로 답변 모아서 투표(Majority Vote) 진행하기
    votes_by_id = {}
    for res in all_raw_results:
        q_id = res.get("id")
        if not q_id: continue
        
        if q_id not in votes_by_id:
            votes_by_id[q_id] = []
        votes_by_id[q_id].append(res)
        
    final_predictions = []
    
    # 데이터 매핑을 위해 원본 질문을 딕셔너리로 준비
    question_map = {item["id"]: item["question"] for item in questions_data}
    difficulty_map = {item["id"]: item.get("difficulty", "N/A") for item in questions_data}

    # 5. 각 문제(ID)별로 가장 많이 나온 답 찾기
    for q_id, responses in votes_by_id.items():
        # 해당 ID에서 나온 모든 'answer'만 추출 (예: ["5%", "5%", "10%"])
        answers = [r.get("answer", "") for r in responses]
        
        # Counter를 이용해 가장 많이 나온 정답 1개(최빈값) 찾기
        majority_answer = Counter(answers).most_common(1)[0][0]
        
        # 다수결 정답을 낸 응답 중 첫 번째 응답의 사고 과정을 대표로 가져오기
        representative_reason = next((r.get("reasoning_process", "") for r in responses if r.get("answer") == majority_answer), "")
        
        # 최종 결과 조합
        final_predictions.append({
            "id": q_id,
            "question": question_map.get(q_id, "질문 매핑 실패"),
            "difficulty": difficulty_map.get(q_id, "N/A"),
            "reasoning_process": representative_reason,
            "answer": majority_answer,
            "vote_details": dict(Counter(answers)) # 어떤 투표 결과가 나왔는지 확인용 (예: {"5%": 2, "10%": 1})
        })

    elapsed_time = time.time() - start_time
    metrics = {
        "elapsed_time_sec": round(elapsed_time, 2),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens
    }
            
    return final_predictions, metrics