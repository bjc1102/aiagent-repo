import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import time
import os
from dotenv import load_dotenv
from pathlib import Path
from prompts.templates import get_prompt


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL  = os.getenv("GEMINI_MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")
MAX_OUTPUT_TOKENS = os.getenv("MAX_OUTPUT_TOKENS")
PROMPT_TYPE = os.getenv("PROMPT_TYPE") # zero_shot | few_shot | cot | self_consistency

client = genai.Client(api_key=GEMINI_API_KEY)

class OutputSchema(BaseModel):
    answer: str = Field(..., description="주어진 의료급여제도 질문에 대한 답을 '5%', '병원급 이상 10%', '무료', '100,000원', '틀니 75,000원', '해당되지 않음'과 같이 간략히 작성합니다.")
    reason: str = Field(..., description="답에 대한 추론 과정을 작성합니다.")




def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# def ask_llm(system_prompt, customer_message):
#     response = client.models.generate_content(
#         model = GEMINI_MODEL,
#         config={
#             'system_instruction':system_prompt,
#             'temperature': TEMPERATURE,
#             "max_output_tokens": MAX_OUTPUT_TOKENS,
#             "response_mime_type": "application/json",
#             "response_schema": OutputSchema
            
#         },
#         contents = customer_message
#     )
#     return response

def ask_llm(system_prompt, customer_message):
    response = client.models.generate_content(
        model = GEMINI_MODEL,
        config=types.GenerateContentConfig(  # 딕셔너리 대신 types 객체 권장
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=OutputSchema,
            # Thinking 기능을 0으로 제한하여 비활성화
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
        contents = customer_message
    )
    return response

def main():
    base_dir = Path(__file__).resolve().parent
    week2_dir = base_dir.parent
    data_dir = week2_dir / 'data'
    output_dir = base_dir / 'output'
    md_file_path = base_dir / "medical_payment.md"

    parsing_success_count=0
    validation_count=0
    exact_match_count=0
    total = 0

    # 데이터 로드
    query_data = load_data(str(data_dir / 'dataset.jsonl'))
    answer_data = load_data(str(data_dir / 'answer_key.jsonl'))
    
    # 질의 데이터와 정답 데이터 병합
    merged = {d["id"]: d.copy() for d in query_data}
    for d in answer_data:
        if d["id"] in merged:
            merged[d["id"]].update(d)
    data = list(merged.values())
    # print(data)
    
    # 프롬프트 구조 호출
    system_prompt = get_prompt(PROMPT_TYPE, str(md_file_path))


    # llm process
    for i, query in enumerate(data):

        id = query['id']
        question = query['question']
        difficulty = query['difficulty']
        expected_answer = query['expected_answer']
        reasoning = query['reasoning']
        save_dict = {}


        input_token = 0
        output_token = 0
        total_token = 0

        print(f"{id}: {question}")

        try:
            # LLM 질의
            response = ask_llm(system_prompt, question)
            raw_text = response.text
            usage = response.usage_metadata

            input_token = usage.prompt_token_count
            output_token = usage.candidates_token_count
            total_token = usage.total_token_count
            print(f"입력 토큰 수 (Prompt): {usage.prompt_token_count}")
            print(f"출력 토큰 수 (Candidates): {usage.candidates_token_count}")
            print(f"전체 토큰 수 (Total): {usage.total_token_count}")

            # JSON 파싱
            parsed_dict = json.loads(raw_text)
            save_dict = {'id':id, **parsed_dict}
            save_dict['input_token']=input_token
            save_dict['output_token']=output_token
            save_dict['total_token']=total_token
            

            parsing_success_count += 1

            # 스키마 검증
            validated_output = OutputSchema(**parsed_dict).model_dump()
            validation_count +=1
            
            # Exact Match 확인
            is_correct = (validated_output['answer'] == expected_answer)
            print(f"정답: {expected_answer}")
            print(f"출력값: {validated_output['answer']}")
            if is_correct:
                exact_match_count += 1


        except json.JSONDecodeError:
            print(f"[{id}] 에러: JSON 파싱 실패")
        except ValidationError as e:
            print(f"[{id}] 에러: 스키마 불일치\n{e}")
        except Exception as e:
            print(f"[{id}] 에러: {e}")
        

        with open(str( output_dir / f'{PROMPT_TYPE}_output.jsonl'), "a", encoding="utf-8") as f:
            f.write(json.dumps(save_dict, ensure_ascii=False) + '\n')

        total += 1
        # time.sleep(30)

    print(f'Parsing 성공 횟수: {parsing_success_count}/{total}')
    print(f'Schema 규칙 준수: {validation_count}/{total}')
    print(f'일치 정확도: {exact_match_count}/{total}')

if __name__ == "__main__":
    main()