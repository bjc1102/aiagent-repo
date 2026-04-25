"""golden_dataset.json → golden_dataset_v2.jsonl 변환기.

추가 필드:
  - ground_truth: expected_answer (이미 완전한 문장 형태, 그대로 사용)
  - ground_truth_contexts: FAISS 청크에서 keyword 기반으로 발췌
"""
import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- FAISS 청크 로드 ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vs = FAISS.load_local(
    os.path.join(SCRIPT_DIR, "medical_advanced_index"),
    embeddings,
    allow_dangerous_deserialization=True,
)
all_docs = list(vs.docstore._dict.values())
docs_by_year = {"2025": [], "2026": []}
for d in all_docs:
    y = d.metadata.get("source_year")
    if y in docs_by_year:
        docs_by_year[y].append(d)

# --- 질문 ID별 ground_truth_contexts 매핑을 위한 키워드 ---
# 각 질문의 정답이 담긴 청크를 찾는 키워드들
KEYWORDS_BY_ID = {
    "q01": ["장기지속형 주사제", "2%"],           # 2026 정신질환
    "q02": ["장기지속형 주사제", "5%"],            # 2025 정신질환
    "q03": ["틀니", "15%"],                         # 2025 65세 이상
    "q04": ["임플란트", "10%"],                     # 2026 65세 이상
    "q05": ["복잡추나", "40%"],                     # 2025 추나요법
    "q06": ["고위험 임신부", "5%"],                 # 2026 분만
    "q07": ["365회", "30%"],                        # 2026 외래 적정관리
    "q08": ["KTAS", "100분의100"],                  # 2026 응급의료관리료
    "q09": ["조산아", "5년 4개월"],                 # 2026 조산아
    "q10": ["조산아", "5년"],                       # 비교용: 양 연도
    "q11": ["이상지질혈증", "면제"],                # 2026 확진검사
    "q12": ["이상지질혈증", "면제"],                # 2025 (미적용)
    "q13": ["폐쇄병동"],                             # 2025/2026 비교 (양 연도 모두 매칭)
    "q14": ["환급", "월단위"],                      # 2026 산정기간
    "q15": ["치아 홈메우기", "5%"],                 # 2025 16~18세
    "q16": ["치아 홈메우기", "3%"],                 # 2026 6~15세
    "q17": ["MRI", "15%"],                          # 2025 2종 검사
    "q18": ["장기지속형 주사제"],                    # 비교: 5%→2%
    "q19": ["노숙인", "2028"],                      # 2026 유효기간
    "q20": ["복잡추나", "PET"],                     # 2026 복합 계산
}

# cross-year 질문 (양 연도 청크 모두 필요)
CROSS_YEAR_IDS = {"q10", "q13", "q18"}


def find_contexts(q_id, source_year, keywords, top_n=2):
    """키워드로 관련 청크를 찾음. 작은 청크라 top_n=2로 답 근거 확보율 향상."""
    target_years = ["2025", "2026"] if q_id in CROSS_YEAR_IDS else source_year.split(", ")
    candidates = []
    for y in target_years:
        for d in docs_by_year.get(y, []):
            score = sum(1 for kw in keywords if kw in d.page_content)
            if score > 0:
                candidates.append((score, y, d))
    candidates.sort(key=lambda x: -x[0])

    if q_id in CROSS_YEAR_IDS:
        # 각 연도별 top_n개씩 (혹은 가능한 만큼)
        per_year = {y: [] for y in target_years}
        for score, y, d in candidates:
            if len(per_year[y]) < top_n:
                per_year[y].append(d)
        result = []
        for y in target_years:
            result.extend(per_year[y])
        return result
    else:
        return [d for _, _, d in candidates[:top_n]]


# --- 변환 실행 ---
with open(os.path.join(SCRIPT_DIR, "golden_dataset.json"), "r", encoding="utf-8") as f:
    dataset = json.load(f)

out_path = os.path.join(SCRIPT_DIR, "golden_dataset_v2.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in dataset:
        q_id = item["id"]
        kws = KEYWORDS_BY_ID.get(q_id, [])
        ctxs_docs = find_contexts(q_id, item["source_year"], kws)
        ctxs = [d.page_content for d in ctxs_docs]
        matched_years = sorted(set(d.metadata.get("source_year") for d in ctxs_docs))

        record = {
            "id": q_id,
            "question": item["question"],
            "ground_truth": item["expected_answer"],
            "ground_truth_contexts": ctxs,
            "matched_context_years": matched_years,
            "difficulty": item["difficulty"],
            "source_year": item["source_year"],
            "source_section": item.get("source_section", ""),
            "evidence_text": item.get("evidence_text", ""),
            "conditions": item.get("conditions", []),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"{q_id} ({item['difficulty']}, src={item['source_year']}): {len(ctxs)} 청크 매칭, 연도={matched_years}")

print(f"\n✅ 저장: {out_path}")
