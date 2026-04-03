import requests
import os

# 1. 설정
api_key = "up_VoaKpBee7NlNkh4VYEF6kBV3KMG1b" 
# 파일 경로 (현재 version2 폴더 기준 상위 data 폴더의 PDF)
filename = "../data/2024 알기 쉬운 의료급여제도.pdf" 

url = "https://api.upstage.ai/v1/document-digitization"
headers = {"Authorization": f"Bearer {api_key}"}

# 2. 파일 열기 및 파라미터 구성
# 'pages': '4' 를 추가하여 딱 4페이지만 분석하도록 제한합니다.
with open(filename, "rb") as f:
    files = {"document": f}
    data = {
        "model": "document-parse",
        "ocr": "force",
        "pages": "4,5,6,7",  # 딱 4페이지만 테스트
        "base64_encoding": "['table']" # 표 데이터를 인코딩하여 포함
    }
    
    print(f"🚀 Upstage API로 4페이지 분석 요청 중...")
    response = requests.post(url, headers=headers, files=files, data=data)

# 3. 결과 출력
if response.status_code == 200:
    result = response.json()
    print("✅ 분석 성공!")
    
    # 결과가 너무 길 수 있으므로 파일로 저장하여 확인하세요.
    import json
    with open("page4_result.json", "w", encoding="utf-8") as out:
        json.dump(result, out, indent=4, ensure_ascii=False)
    print("📄 결과가 'page4_result.json'에 저장되었습니다.")
else:
    print(f"❌ 에러 발생: {response.status_code}")
    print(response.text)