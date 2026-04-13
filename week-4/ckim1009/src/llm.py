from pydantic import BaseModel, ValidationError, Field
import json

SYSTEM_PROMPT = """#당신은 의료급여 본인부담률 질문에 정확히 답해야합니다.
#아래에 정리한 의료급여 정보만을 참조하여 답해야합니다.

{md_content}
"""

class OutputSchema(BaseModel):
    answer: str = Field(..., description="주어진 의료급여제도 질문에 대한 답을 '5%', '병원급 이상 10%', '무료', '100,000원', '틀니 50,000원', '해당되지 않음'과 같이 간략히 작성합니다.")
    reason: str = Field(..., description="답에 대한 추론 과정을 작성합니다.")


def ask_llm(client, system_prompt, customer_message):
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        config={
            'system_instruction':system_prompt,
            'temperature':0,
            "max_output_tokens": 800,
            "response_mime_type": "application/json",
            "response_schema": OutputSchema
        },
        contents = customer_message
    )
    return response.text


def eval_rag_pipeline(client, golden_dataset):
    parsing_success_count=0
    validation_count=0
    exact_match_count=0
    

    total = len(golden_dataset)
    for i, item in enumerate(golden_dataset):
        if i<19:
            continue


        id = item['id']
        question = item["question"]
        expected_output = item["expected_answer"]
        md_content = item['top_chunk']

        system_prompt = SYSTEM_PROMPT.format(md_content=md_content)

        try:
            # llm 호출
            raw_output = ask_llm(client, system_prompt, question)
            print(raw_output)

            # JSON 파싱
            parsed_dict = json.loads(raw_output)
            parsing_success_count += 1

            # 스키마 검증
            validated_output = OutputSchema(**parsed_dict).model_dump()
            validation_count +=1
            
            # Exact Match 확인
            is_correct = (validated_output['answer'] == expected_output)
            if is_correct:
                exact_match_count += 1
            
            # 최종 결과 저장 (json.dump 사용)
            with open(f'output/output.jsonl', "a", encoding="utf-8") as f:
                f.write(json.dumps(parsed_dict, ensure_ascii=False) + '\n')

        except json.JSONDecodeError:
            print(f"[{id}] 에러: JSON 파싱 실패")
        except ValidationError as e:
            print(f"[{id}] 에러: 스키마 불일치\n{e}")
        except Exception as e:
            print(f"[{id}] 에러: {e}")

        

        total += 1

    print(f'Parsing 성공 횟수: {parsing_success_count}/{total}')
    print(f'Schema 규칙 준수: {validation_count}/{total}')
    print(f'일치 정확도: {exact_match_count}/{total}')