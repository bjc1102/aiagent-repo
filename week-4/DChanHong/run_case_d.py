import asyncio
import os
from src.domains.rag.service import rag_service

async def main():
    print("Starting CASE D (High Recall) Evaluation...")
    # Dense 50, Sparse 50, Rerank 15 설정은 이미 service.py에 적용됨
    
    # 평가 실행 (경로를 rerank/test_D로 지정)
    result = await rag_service.run_evaluation(version="rerank/test_D")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Evaluation complete!")
        print(f"Total: {result['total']}, Processed: {result['processed']}")
        print(f"Results saved to: {result['output_file']}")

if __name__ == "__main__":
    asyncio.run(main())
