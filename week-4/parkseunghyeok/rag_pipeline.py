#!/usr/bin/env python3

import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader # Using TextLoader for .txt files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import shutil # Import shutil for directory removal

# Load environment variables from .env file
load_dotenv(dotenv_path='/home/park/start/AI/aiagent-repo/.env')

# --- Debugging: Check if GOOGLE_API_KEY is loaded ---
print(f"GOOGLE_API_KEY loaded: {os.getenv('GOOGLE_API_KEY') is not None}")
# ---------------------------------------------------

# Define PlaceholderLLM in global scope
class PlaceholderLLM:
    def invoke(self, prompt):
        return "LLM not initialized: Please set GOOGLE_API_KEY to get actual responses."

# Set environment variables for Cohere API (if using for re-ranking later)
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# Define document paths - Using .txt files
pdf_paths = { # Renamed to document_paths for clarity, but variable name unchanged for minimal diff
    "2025": "/home/park/start/AI/aiagent-repo/week-4/data/2025-알기-쉬운-의료급여제도.txt",
    "2026": "/home/park/start/AI/aiagent-repo/week-4/data/2026-알기-쉬운-의료급여제도.txt",
}

# 1. 문서 로드 및 청킹
def load_and_chunk_pdfs(document_paths):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "],
        length_function=len,
    )

    for year, path in document_paths.items():
        print(f"Loading and processing {year} document from {path}...")
        loader = TextLoader(path) # Use TextLoader
        documents = loader.load()
        
        # Add source_year metadata to the loaded documents before splitting
        for doc in documents:
            doc.metadata["source_year"] = year

        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"  - Generated {len(chunks)} chunks for {year}.")
    return all_chunks

# 2. 벡터 저장소 구축
def create_vectorstore(chunks, persist_directory="./chroma_db"):
    # Remove existing ChromaDB data if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Removed existing ChromaDB data from {persist_directory}")

    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Vector store created and persisted to {persist_directory}")
    return vectorstore, embeddings

# 3. Basic RAG 파이프라인 구성 (LCEL 방식)
def create_basic_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG 프롬프트 템플릿
    rag_prompt_template = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 \"정보를 찾을 수 없습니다\"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    # LCEL 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# Main execution
if __name__ == "__main__":
    # Load and chunk documents, then create vector store
    chunks = load_and_chunk_pdfs(pdf_paths) # Use pdf_paths, but it points to .txt files now
    vectorstore, embeddings = create_vectorstore(chunks)

    # Configure Gemini API and list models (for debugging, can be removed later)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        print("\n--- Available Gemini Models (generateContent supported) ---")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
        print("----------------------------------------------------------")
    else:
        print("GOOGLE_API_KEY not found. Cannot list Gemini models.")

    # Initialize LLM with the correct model name
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI. Make sure GOOGLE_API_KEY is set and model name is correct. Error: {e}")
        print("Using a placeholder LLM. Basic RAG will not function correctly without a proper LLM.")
        llm = PlaceholderLLM()

    # Create Basic RAG chain and get the retriever
    basic_rag_chain, basic_retriever = create_basic_rag_chain(vectorstore, llm)

    # Load Golden Dataset
    golden_dataset_path = "/home/park/start/AI/aiagent-repo/week-4/parkseunghyeok/golden_dataset.jsonl"
    golden_questions = []
    try:
        with open(golden_dataset_path, 'r', encoding='utf-8') as f:
            golden_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Golden dataset not found at {golden_dataset_path}. Please create it and fill in expected answers.")
        golden_questions = []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {golden_dataset_path}. Ensure it's a valid JSON array.")
        golden_questions = []

    # Evaluate Basic RAG (Step 1-3)
    print("\n--- Evaluating Basic RAG with Golden Dataset ---")
    results_basic_rag = []
    for i, q_data in enumerate(golden_questions):
        if isinstance(llm, PlaceholderLLM):
            llm_response = llm.invoke(q_data["question"])
            retrieved_docs_metadata = []
            is_correct_year_retrieved = "N/A (LLM not initialized)"
            is_chunk_included = "N/A (LLM not initialized)"
        else:
            print(f"\nProcessing question: {q_data['question']}")
            llm_response = basic_rag_chain.invoke(q_data["question"])
            
            # Manually retrieve source documents for evaluation
            retrieved_docs = basic_retriever.invoke(q_data["question"])
            retrieved_docs_metadata = [doc.metadata for doc in retrieved_docs]

            expected_year = q_data.get("source_year")
            if expected_year and expected_year != "2025+2026":
                is_correct_year_retrieved = "O" if any(doc.metadata.get("source_year") == expected_year for doc in retrieved_docs) else "X"
            elif expected_year == "2025+2026":
                has_2025 = any(doc.metadata.get("source_year") == "2025" for doc in retrieved_docs)
                has_2026 = any(doc.metadata.get("source_year") == "2026" for doc in retrieved_docs)
                is_correct_year_retrieved = "O" if has_2025 and has_2026 else "Partially O" if has_2025 or has_2026 else "X"
            else:
                is_correct_year_retrieved = "N/A"
            
            is_chunk_included = "O" if retrieved_docs else "X"

        is_correct = "수동 판정 필요"
        if not isinstance(llm, PlaceholderLLM) and q_data["expected_answer"] != "(2025년 PDF에서 찾은 값)" and q_data["expected_answer"] != "(2026년 PDF에서 찾은 값)" and q_data["expected_answer"] != "(2025, 2026년 PDF에서 찾은 비교 값)":
            if q_data["expected_answer"] in llm_response:
                is_correct = "정답"
            else:
                is_correct = "오답"

        results_basic_rag.append({
            "question_id": f"q{i+1:02d}",
            "difficulty": q_data["difficulty"],
            "source_year": q_data["source_year"],
            "is_chunk_included": is_chunk_included,
            "is_correct_year_retrieved": is_correct_year_retrieved,
            "llm_generated_answer": llm_response,
            "is_correct": is_correct,
            "retrieved_docs_metadata": retrieved_docs_metadata
        })
    
    print("\nBasic RAG Evaluation Results:")
    for res in results_basic_rag:
        print(json.dumps(res, ensure_ascii=False, indent=2))

    correct_count = sum(1 for res in results_basic_rag if res["is_correct"] == "정답")
    total_questions = len(results_basic_rag)
    basic_rag_accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    print(f"\nBasic RAG 정답률: {basic_rag_accuracy:.2f}% ({correct_count}/{total_questions})")
