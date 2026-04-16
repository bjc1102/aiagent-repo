import json
import os

def evaluate_predictions(step_name, predictions_data, answer_key, results_dir, execution_time=0.0, token_usage=None):
    
    # 1. 정답지 로드
    if isinstance(answer_key, str):
        answer_dict = {}
        with open(answer_key, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                answer_dict[data["id"]] = data["expected_answer"]
    else:
        answer_dict = answer_key 

    # 2. 채점 로직 초기화
    correct_count = 0
    wrong_ids = []         # 틀린 ID만 저장할 리스트
    detailed_results = []  # 30건 전체 결과를 저장할 리스트
    total_count = len(predictions_data)

    # 3. 예측 결과와 정답 비교
    for pred in predictions_data:
        q_id = pred.get("id")
        question = pred.get("question")
        actual_answer = pred.get("answer", "") # 실제 답변
        expected_answer = answer_dict.get(q_id, "")
        
        # 난이도 가져오기 (데이터에 없으면 "N/A" 표시)
        difficulty = pred.get("difficulty", "N/A")

        # 정답 여부 체크
        is_correct = (actual_answer == expected_answer)

        if is_correct:
            correct_count += 1
        else:
            wrong_ids.append(q_id) # 틀린 경우 ID만 추가

        # 30건 전체 결과를 detailed_results에 저장
        detailed_results.append({
            "id": q_id,
            "question": question,
            "difficulty": difficulty,
            "is_correct": is_correct,
            "expected": expected_answer,
            "actual": actual_answer
        })

    # 4. 통계 및 Metrics 계산
    wrong_count = total_count - correct_count
    accuracy = correct_count / total_count if total_count > 0 else 0

    summary = {
        "step": step_name,
        "accuracy": f"{accuracy * 100:.2f}%",
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "total_count": total_count
    }

    metrics = {
        "exact_match_rate": round(accuracy, 4),
        "error_rate": round(1 - accuracy, 4),
        "total_evaluated": total_count,
        "execution_time_sec": round(execution_time, 2)
    }

    # ★ 요금 계산 로직 추가 ★
    if token_usage:
        metrics["token_usage"] = token_usage
        
        # 100만(1M) 토큰 당 단가 설정 (Gemini Flash 최신 기준 예시)
        # ※ 실제 사용하시는 요금제에 맞춰 단가를 수정해 주세요.
        PRICE_PER_1M_PROMPT = 0.25       # 입력(Prompt) 토큰 단가: $0.075 / 1M
        PRICE_PER_1M_COMPLETION = 1.50    # 출력(Completion) 토큰 단가: $0.30 / 1M

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)

        # 비용 계산 ( (토큰 수 / 1,000,000) * 단가 )
        prompt_cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_PROMPT
        completion_cost = (completion_tokens / 1_000_000) * PRICE_PER_1M_COMPLETION
        total_cost_usd = prompt_cost + completion_cost

        # 소수점이 길어질 수 있으므로 6자리까지만 반올림하여 표시
        metrics["estimated_cost_usd"] = round(total_cost_usd, 6)
    
    if token_usage:
        metrics["token_usage"] = token_usage

    # ★ 5. 최종 결과 구조 변경 ★
    final_result = {
        "summary": summary,
        "metrics": metrics,
        "wrong_ids": wrong_ids,       # 리스트 형태로 틀린 ID만! (예: ["q15"])
        "results": detailed_results   # 30건의 전체 채점 결과!
    }

    # 파일 저장 로직
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
        # 기본 파일명 설정
        base_filename = f"{step_name}_results"
        file_path = os.path.join(results_dir, f"{base_filename}.json")
        
        # 파일이 이미 존재한다면, 뒤에 _1, _2 형식으로 번호를 붙여서 빈 파일명 찾기
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(results_dir, f"{base_filename}_{counter}.json")
            counter += 1
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print(f"[{step_name}] 평가 완료! 결과가 {file_path}에 저장되었습니다.")

    return final_result