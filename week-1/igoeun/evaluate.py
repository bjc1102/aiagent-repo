import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

evaluate_prompt = "classification_results_eng_v6.json"

dataset_path = BASE_DIR / "dataset.jsonl"
result_path = BASE_DIR / evaluate_prompt


# 기대 결과값 불러와서 딕셔너리로 정리
expected_data ={}
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        ticket_id = data.get("id")
        expected_output = data.get("expected_output")
        if ticket_id and expected_output:
            expected_data[ticket_id] = expected_output


try:
    with open(result_path, "r", encoding="utf=8") as f:
        model_results = json.load(f)
except FileNotFoundError:
    print(f"{result_path} 파일 여부 확인 요망, 또는 main.py 먼저 실행")
    exit()


print("-----------정확도 평가 시작--------------\n")

print(f"평가 프롬프트: {evaluate_prompt}\n")

total_count=0
correct_count=0
mismatch_list=[]


# 정답과 모델 결과 비교
for item in model_results:
    ticket_id = item.get("ticket_id")
    model_analysis = item.get("analysis")

    expected = expected_data.get(ticket_id)

    if not expected:
        continue

    total_count += 1


    # 4개 항목(intent, urgency, needs_clarification, route_to)이 모두 일치하는지 확인
    is_correct = True
    mismatched_keys = []

    for key in ["intent", "urgency", "needs_clarification", "route_to"]:
        if model_analysis.get(key) != expected.get(key):
            is_correct = False
            mismatched_keys.append({
                "항목": key,
                "정답": expected.get(key),
                "모델결과": model_analysis.get(key)
            })
        
    if is_correct:
        correct_count += 1
    else:
        mismatch_list.append({
            "ticket_id": ticket_id,
            "차이점": mismatched_keys
        })


# 최종 정확도 계산 및 오답 출력
accuracy = (correct_count / total_count) * 100

print(f"전체 항목: {total_count}개")
print(f"정답 항목: {correct_count}개")
print(f"정확도: {accuracy:.1f}%\n")

if mismatch_list:
    print("- - - 불일치 항목- - -\n")
    for mismatch in mismatch_list:
        print(f"[{mismatch['ticket_id']}]")
        for diff in mismatch['차이점']:
            print(f" * {diff['항목']} -> 정답: {diff['정답']} / 모델분류: {diff['모델결과']}")
        print()
else:
    print("모든 분류가 정답과 일치합니다.")