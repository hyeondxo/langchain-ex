# LangChain RAG 챗봇 프로젝트

## 프로젝트 개요

LangChain을 사용한 **대화형 문서 기반 챗봇** 시스템입니다. docs/ 폴더의 문서를 학습하여 사용자 질문에 정확하게 답변합니다.

**RAG(Retrieval Augmented Generation)란?**
- 문서에서 관련 정보를 검색(Retrieval)하여 LLM에게 제공함으로써 더 정확한 답변을 생성(Generation)
- 할루시네이션(거짓 정보) 방지 및 최신/전문 지식 활용 가능
- 문서 내용만을 근거로 답변하여 신뢰성 향상

## 주요 기능

✅ **대화형 인터페이스**: 사용자 프롬프트를 실시간으로 입력받아 답변
✅ **문서 자동 파싱**: docs/ 폴더의 PDF, TXT 파일 자동 인식 및 처리
✅ **벡터DB 영속화**: 한 번 처리한 문서는 재사용 가능 (처리 시간 단축)
✅ **모듈화된 구조**: 문서 처리, 벡터 저장소, 챗봇 로직 분리
✅ **오류 처리**: 다양한 PDF 파싱 전략 및 사용자 친화적 오류 메시지

## 시스템 구조

### 대화형 챗봇 아키텍처
```
사용자 ←→ [대화형 인터페이스]
              ↓
        [RAG Chatbot]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[Document Processor]  [VectorStore Manager]
    ↓                   ↓
PDF/TXT 파일      ChromaDB (영속화)
    ↓                   ↓
  청크 분할          벡터 검색
    └─────────┬─────────┘
              ↓
    [RAG Chain Pipeline]
              ↓
    질문 → 검색 → 프롬프트 → LLM → 답변
```

### 처리 흐름
```
1. 사용자 질문 입력
   ↓
2. 벡터 검색 (Retriever) → 관련 문서 청크 4개 검색
   ↓
3. 프롬프트 생성 → "컨텍스트: [검색된 청크들] + 질문: [사용자 질문]"
   ↓
4. LLM 실행 (GPT-4o-mini) → 문서 기반 답변 생성
   ↓
5. 답변 출력 → 다음 질문 대기
```

## 7단계 처리 파이프라인

### 1단계: 문서 로드
- `docs/` 폴더의 PDF, TXT 파일 읽기
- PyPDFLoader → PyMuPDFLoader 폴백 메커니즘

### 2단계: 텍스트 청크 분할
- **chunk_size=800**: 각 청크 최대 800자
- **chunk_overlap=150**: 인접 청크 간 150자 중복 (문맥 유지)

### 3단계: 임베딩 & 벡터DB 구축
- OpenAI Embeddings (1536차원 벡터)
- ChromaDB에 저장

### 4단계: Retriever 생성
- 질문과 가장 유사한 상위 **k=4**개 청크 검색

### 5단계: 프롬프트 템플릿
- 시스템 메시지: "문서 기반으로만 답변하라"
- 사용자 메시지: 실제 질문

### 6단계: RAG 체인 구성
```python
질문 → Retriever → format_docs → 프롬프트 → LLM → 파서 → 답변
```

### 7단계: 실행
```python
answer = rag_chain.invoke("질문")
```

## 빠른 시작 (Quick Start)

### 1. 환경 설정
```bash
# Python 3.12.7 이상 필요
python --version

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정
`.env` 파일 생성 및 OpenAI API 키 추가:
```env
OPENAI_API_KEY=sk-proj-your-api-key-here
```

### 3. 문서 준비
`docs/` 폴더에 학습할 문서 추가 (PDF 또는 TXT):
```bash
# 예시
docs/
├── 프로젝트_문서.pdf
├── 기술_명세서.pdf
└── 참고자료.txt
```

### 4. 챗봇 실행
```bash
# 대화형 챗봇 실행 (권장)
python chatbot.py

# 또는 기존 단일 질문 실행
python langchain_rag_ex.py
```

### 5. 챗봇 사용 예시
```
🚀 RAG 챗봇 초기화 중...

✓ PDF 로드 성공: 6조_VM_발표자료_최종.pdf
📚 총 37개의 문서 페이지 로드 완료
📄 142개의 청크로 분할 완료

🔧 벡터 데이터베이스 구축 중...
✓ 벡터DB 구축 완료 (142개 청크 저장)

============================================================
📚 문서 기반 RAG 챗봇에 오신 것을 환영합니다!
============================================================

💡 사용법:
  - 문서에 대해 질문하세요
  - 'quit', 'exit', 'q'를 입력하면 종료됩니다
  - 'help'를 입력하면 도움말을 볼 수 있습니다


[질문 1] 당신: VM이란 무엇인가요?

🤖 챗봇: VM은 Virtual Machine의 약자로...

[질문 2] 당신: 이 문서의 핵심 내용을 요약해주세요

🤖 챗봇: 이 문서는 6조의 VM 프로젝트에 대한 발표자료로...
```

## 코드 구조 상세

### 파일 구조
```
langchain-ex/
├── chatbot.py              # 대화형 챗봇 (메인 실행 파일)
├── langchain_rag_ex.py     # 단일 질문 실습 예제
├── docs/                   # 학습 문서 폴더
│   ├── *.pdf              # PDF 문서들
│   └── *.txt              # TXT 문서들
├── chroma_db/             # 벡터DB 저장소 (자동 생성)
├── .env                   # API 키 설정
├── requirements.txt       # 의존성 목록
└── README.md             # 이 문서
```

### 주요 클래스 및 컴포넌트

#### 1. `DocumentProcessor` 클래스
**역할**: 문서 로드 및 청크 분할
```python
# 사용 예시
processor = DocumentProcessor(docs_dir="docs", chunk_size=800)
documents = processor.load_documents()  # PDF/TXT 로드
chunks = processor.split_documents(documents)  # 청크 분할
```

**주요 기능**:
- PDF 파일: PyPDFLoader → PyMuPDFLoader 폴백 전략
- TXT 파일: UTF-8 인코딩 지원
- 청크 분할: RecursiveCharacterTextSplitter (중복 허용)

#### 2. `VectorStoreManager` 클래스
**역할**: 벡터 데이터베이스 구축 및 관리
```python
# 사용 예시
vs_manager = VectorStoreManager(persist_directory="./chroma_db")
vs_manager.build_vectorstore(chunks)  # 벡터DB 생성
retriever = vs_manager.get_retriever(k=4)  # Retriever 생성
```

**주요 기능**:
- ChromaDB 기반 벡터 저장소
- 디스크 영속화 (재실행 시 재사용 가능)
- OpenAI Embeddings (text-embedding-3-small)
- 유사도 검색 (Top-K)

#### 3. `RAGChatbot` 클래스
**역할**: RAG 체인 구성 및 대화 인터페이스
```python
# 사용 예시
chatbot = RAGChatbot(vectorstore_manager, model="gpt-4o-mini")
chatbot.chat_loop()  # 대화형 루프 시작
```

**주요 기능**:
- RAG 파이프라인 자동 구성
- 대화형 인터페이스 (질문 입력 → 답변 출력)
- 도움말 및 종료 명령 지원
- 오류 처리 및 예외 관리

### 구성 요소별 세부 정보

| 컴포넌트 | 역할 | 파일 |
|---------|------|------|
| **DocumentProcessor** | PDF/TXT 로드, 청크 분할 | chatbot.py:37-74 |
| **VectorStoreManager** | 벡터DB 구축/로드/관리 | chatbot.py:81-139 |
| **RAGChatbot** | RAG 체인, 대화 인터페이스 | chatbot.py:146-235 |
| **main()** | 초기화 및 실행 | chatbot.py:243-284 |

### 핵심 개념 설명

#### 임베딩 (Embedding)
```python
"고양이" → [0.2, -0.5, 0.8, ..., 0.1]  # 1536차원 벡터
"강아지" → [0.3, -0.4, 0.7, ..., 0.2]  # 유사한 벡터
"자동차" → [-0.8, 0.9, -0.2, ..., 0.5] # 먼 거리의 벡터
```

#### 청크 분할 예시
```
원본 (1500자):
"안녕하세요. LangChain은 LLM 애플리케이션 개발 프레임워크입니다. ..."

청크 1 (800자):
"안녕하세요. LangChain은 LLM 애플리케이션..."

청크 2 (800자, 150자 중복):
"...LLM 애플리케이션 개발 프레임워크입니다. RAG는..."
```

#### RAG 체인 실행 흐름
```python
# 입력
question = "VM이란 무엇인가요?"

# 내부 처리
1. retriever(question) → [청크1, 청크2, 청크3, 청크4]
2. format_docs([청크들]) → "[0] VM은...\n\n[1] 가상머신은..."
3. RAG_PROMPT.format(context=..., question=...) → 완성된 프롬프트
4. llm(프롬프트) → AIMessage("VM은 Virtual Machine의 약자로...")
5. parser(AIMessage) → "VM은 Virtual Machine의 약자로..."

# 출력
"VM은 Virtual Machine의 약자로..."
```

## 커스터마이징 가이드

### 챗봇 설정 변경

#### 1. 검색 청크 수 조정
```python
# chatbot.py의 main() 함수 수정
chatbot = RAGChatbot(
    vectorstore_manager=vectorstore_manager,
    retrieval_k=6  # 기본값 4 → 6으로 변경 (더 많은 컨텍스트)
)
```

#### 2. 청크 크기 및 중복 조정
```python
# chatbot.py의 main() 함수 수정
doc_processor = DocumentProcessor(
    docs_dir="docs",
    chunk_size=1000,    # 기본값 800 → 1000 (더 큰 청크)
    chunk_overlap=200   # 기본값 150 → 200 (더 많은 중복)
)
```

#### 3. LLM 모델 변경
```python
# chatbot.py의 main() 함수 수정
chatbot = RAGChatbot(
    vectorstore_manager=vectorstore_manager,
    model="gpt-4",       # gpt-4o-mini → gpt-4 (더 강력, 비용 증가)
    temperature=0.3      # 기본값 0.0 → 0.3 (더 창의적)
)
```

#### 4. 프롬프트 템플릿 수정
```python
# chatbot.py의 RAGChatbot.__init__() 내부 수정
self.prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 친절한 AI 어시스턴트입니다. "  # 원하는 역할로 변경
     "문서를 기반으로 상세하게 설명해주세요.\n\n"
     "컨텍스트:\n{context}"),
    ("user", "{question}")
])
```

#### 5. 벡터DB 저장 위치 변경
```python
# chatbot.py의 main() 함수 수정
vectorstore_manager = VectorStoreManager(
    collection_name="my_custom_docs",     # 컬렉션 이름
    persist_directory="./my_vector_db"    # 저장 경로 변경
)
```

## 학습 포인트

### 초보자가 이해해야 할 핵심 개념

1. **Document vs Chunk**
   - Document: 원본 파일(페이지 단위)
   - Chunk: 검색 가능한 작은 조각

2. **임베딩 (Embedding)**
   - 텍스트를 숫자 벡터로 변환
   - 의미가 비슷하면 벡터도 비슷함

3. **벡터 유사도 검색**
   - 질문 벡터와 가장 가까운 청크 벡터 찾기
   - 코사인 유사도 사용

4. **프롬프트 엔지니어링**
   - LLM에게 명확한 지시 제공
   - "문서 기반으로만 답하라" 제약 추가

5. **체인 (Chain)**
   - 여러 단계를 파이프라인으로 연결
   - LCEL (LangChain Expression Language) 문법

## 문제 해결 (Troubleshooting)

### 문서 로드 실패
**증상**: "docs/ 디렉토리에서 문서를 찾을 수 없습니다" 오류
```bash
# 해결 방법
1. docs/ 폴더가 존재하는지 확인
   ls -la docs/

2. PDF 또는 TXT 파일이 있는지 확인
   ls docs/*.pdf docs/*.txt

3. 파일 권한 확인
   chmod 644 docs/*.pdf
```

### API 키 오류
**증상**: "OPENAI_API_KEY가 설정되지 않았습니다" 오류
```bash
# .env 파일 확인
cat .env

# 환경변수 로드 테스트
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"

# API 키 형식 확인 (sk-proj-로 시작해야 함)
```

### 벡터DB 관련 문제
**증상**: ChromaDB 오류 또는 벡터 검색 실패
```bash
# 해결 방법
1. 기존 벡터DB 삭제 후 재구축
   rm -rf chroma_db/
   python chatbot.py

2. 다른 컬렉션 이름 사용
   # chatbot.py 수정: collection_name="new_collection"
```

### 청크 생성 실패
**증상**: "청크 생성 실패" 오류
```python
# 해결 방법: chunk_size 조정
doc_processor = DocumentProcessor(
    chunk_size=500,      # 800 → 500으로 감소
    chunk_overlap=100    # 150 → 100으로 감소
)
```

### 메모리 부족
**증상**: 처리 중 메모리 오류 발생
```python
# 해결 방법
1. chunk_size 증가 (청크 수 감소)
   chunk_size=1200

2. 검색 청크 수 감소
   retrieval_k=2  # 4 → 2

3. 문서를 여러 번에 나누어 처리
```

### 응답 품질 문제

**문제**: 답변이 부정확하거나 관련 없는 내용
```python
# 해결 방법
1. 검색 청크 수 증가
   retrieval_k=6  # 더 많은 컨텍스트

2. 청크 크기 조정
   chunk_size=600, chunk_overlap=200  # 더 많은 중복

3. 프롬프트 개선
   # 더 구체적인 지시사항 추가
```

**문제**: "제공된 문서에서는 해당 정보를 찾을 수 없습니다" 반복
```python
# 해결 방법
1. 문서가 실제로 관련 정보를 포함하는지 확인
2. 검색 청크 수 증가: retrieval_k=8
3. 질문을 더 구체적으로 변경
```

## 프로젝트 구조

```
langchain-ex/
├── .env                           # OpenAI API 키 설정 (보안 주의!)
├── .python-version                # Python 3.12.7
├── .venv/                         # Python 가상환경
├── docs/                          # 학습 문서 디렉토리
│   ├── *.pdf                     # PDF 문서들
│   └── *.txt                     # TXT 문서들
├── chroma_db/                     # 벡터DB 저장소 (자동 생성)
│   └── [벡터 데이터]
├── chatbot.py                     # 🆕 대화형 챗봇 (메인)
├── langchain_rag_ex.py            # 단일 질문 실습 예제 (기존)
├── requirements.txt               # Python 의존성 목록
└── README.md                     # 프로젝트 문서 (이 파일)
```

## 기능 비교

| 기능 | `chatbot.py` | `langchain_rag_ex.py` |
|------|--------------|----------------------|
| 대화형 인터페이스 | ✅ 지원 | ❌ 단일 질문만 |
| 벡터DB 영속화 | ✅ 디스크 저장 | ❌ 메모리만 |
| 모듈화 구조 | ✅ 클래스 기반 | ❌ 단일 스크립트 |
| 문서 재사용 | ✅ 가능 | ❌ 매번 재처리 |
| 오류 처리 | ✅ 포괄적 | ⚠️ 기본적 |
| 사용 목적 | 실제 사용 | 학습/실습 |

## 고급 기능 확장 아이디어

### 1. 대화 이력 추가
```python
from langchain.memory import ConversationBufferMemory

# chatbot.py에 추가
memory = ConversationBufferMemory(return_messages=True)
# 이전 대화 맥락을 유지하여 연속적인 질문 가능
```

### 2. 스트리밍 응답
```python
# 답변을 실시간으로 출력
for chunk in self.rag_chain.stream(question):
    print(chunk, end="", flush=True)
```

### 3. 웹 UI 추가
```python
# Streamlit 또는 Gradio로 웹 인터페이스 구축
import streamlit as st

st.title("RAG 챗봇")
user_input = st.text_input("질문을 입력하세요:")
if user_input:
    answer = chatbot.ask(user_input)
    st.write(answer)
```

### 4. 다국어 지원
```python
# 프롬프트에 언어 설정 추가
("system", "Answer in {language}. Context:\n{context}")
```

### 5. 소스 인용 추가
```python
# 답변에 출처 문서 정보 포함
def format_docs_with_source(docs_):
    return "\n\n".join([
        f"[문서 {i+1}] (출처: {d.metadata.get('source', 'Unknown')})\n{d.page_content}"
        for i, d in enumerate(docs_)
    ])
```

### 6. 멀티모달 지원
```python
# 이미지가 포함된 PDF 처리
from langchain_community.document_loaders import UnstructuredPDFLoader
# 이미지와 텍스트를 함께 임베딩
```

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB](https://docs.trychroma.com/)
