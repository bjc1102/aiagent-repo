# 📚 RAG 프로젝트 구조 및 역할

이번 주차에 구축한 LangChain 기반의 RAG(Retrieval-Augmented Generation) 프로젝트 구조와 각 폴더/파일의 역할을 정리한 문서입니다. 전체적으로 유지보수와 확장이 쉬운 모듈화 구조를 채택했습니다.

---

## 📂 디렉터리 구조 및 설명

```text
DChanHong/
├── storage/            # 🗄️ FAISS 벡터 데이터베이스가 저장되는 공간 (실행 시 자동 생성)
├── logs/               # 📝 파이프라인 실행 중 발생하는 로그(app.log)가 기록되는 곳
├── src/                # ⚙️ 프로젝트의 핵심 소스 코드가 모여있는 폴더
│   ├── config/         # ⚙️ 환경 설정
│   │   └── settings.py # 환경 변수(.env)를 로드하고, 청크 사이즈, 모델명 등 공통 설정을 관리합니다.
│   │
│   ├── core/           # 🧠 RAG 핵심 모듈
│   │   ├── loader.py   # [1단계] PDF 파일을 읽어오고 파싱하여 Document 객체로 만드는 역할
│   │   ├── splitter.py # [2단계] 문서를 적절한 길이로 자르는(Chunking) 역할 (테이블 보존 로직 포함 예정)
│   │   └── embedder.py # [3단계] 자른 텍스트를 백터(숫자)로 임베딩하고 FAISS 리소스에 저장/조회하는 역할
│   │
│   ├── utils/          # 🛠️ 공통 유틸리티
│   │   └── logger.py   # 파일과 콘솔 양쪽에 로그를 출력해 주는 로거 도구
│   │
│   └── pipeline.py     # 🔄 전체 흐름 제어판. loader -> splitter -> embedder 로 이어지는 과정을 하나로 묶습니다.
│
├── .env                # 🔑 OpenAI API 키 등 외부에 노출되면 안 되는 보안 정보가 담기는 파일
├── .gitignore          # 🚫 Git에 올라가면 안될 파일(용량이 큰 스토리지, 보안 키 등)을 명시한 파일
├── requirements.txt    # 📦 이 프로젝트를 실행하기 위해 필요한 파이썬 라이브러리 목록
├── test_loader.py      # 🧪 PDF 로더 기능이 정상적으로 돌아가는지 단독으로 테스트하기 위해 만든 스크립트
└── main.py             # 🚀 프로젝트 실행 파일. 터미널에서 이 파일을 통해 파이프라인을 실행합니다.
```

## 🚀 파이프라인 데이터 흐름
RAG의 데이터베이스 구축(Indexing) 과정은 다음과 같이 흘러갑니다.

1. **`main.py` 실행**: 사용자가 터미널에 명령어를 입력해 시작
2. **`src/pipeline.py` 호출**: 전체 흐름을 관장하기 시작
3. **`src/core/loader.py`**: `--source` 로 넘겨진 PDF 파일을 읽어옴
4. **`src/core/splitter.py`**: 읽어들인 텍스트를 설정된 크기(`config.settings`)로 자름
5. **`src/core/embedder.py`**: 자른 텍스트 단위로 임베딩한 뒤, `./storage` 경로에 FAISS 인덱스로 저장함

## ▶️ 실행 방법 예시
```bash
# 특정 PDF 파싱 후 벡터 DB 저장
python main.py --source "../data/2024 알기 쉬운 의료급여제도.pdf"

# 특정 폴더 안의 모든 PDF 처리
python main.py --source "../data" --directory
```
