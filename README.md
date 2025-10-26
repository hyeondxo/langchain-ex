# LangChain RAG 실습 프로젝트

## 프로젝트 개요

이 프로젝트는 LangChain을 사용한 RAG(Retrieval Augmented Generation) 시스템 구현 예제입니다.

**RAG란?**
- 문서에서 관련 정보를 검색(Retrieval)하여 LLM에게 제공함으로써 더 정확한 답변을 생성(Generation)
- 할루시네이션(거짓 정보) 방지 및 최신/전문 지식 활용 가능

## 시스템 구조

```
질문 입력
    ↓
벡터 검색 (Retriever) → 관련 문서 청크 4개 검색
    ↓
프롬프트 생성 → "컨텍스트: [검색된 청크들] + 질문: [사용자 질문]"
    ↓
LLM 실행 (GPT-4o-mini) → 문서 기반 답변 생성
    ↓
답변 출력
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

## 설치 및 실행

### 환경 설정
```bash
# Python 3.12.7 사용
python --version

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치 (이미 설치됨)
pip install -r requirements.txt
```

### API 키 설정
`.env` 파일에 OpenAI API 키 추가:
```env
OPENAI_API_KEY=sk-proj-...
```

### 실행
```bash
python langchain_rag_ex.py
```

## 코드 구조 상세

### 주요 컴포넌트

| 컴포넌트 | 역할 | 라인 |
|---------|------|------|
| **문서 로더** | PDF/TXT 파일 읽기 | 48-77 |
| **텍스트 스플리터** | 청크 분할 | 88-105 |
| **임베딩** | 텍스트 → 벡터 변환 | 117-120 |
| **벡터DB** | 유사도 검색 | 122-132 |
| **Retriever** | 검색 인터페이스 | 138-142 |
| **프롬프트** | LLM 지시사항 | 148-166 |
| **RAG 체인** | 전체 파이프라인 | 185-213 |

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

### 다른 질문 시도
```python
# langchain_rag_ex.py:224 수정
q = "VM의 장점은 무엇인가요?"
q = "이 문서는 어떤 프로젝트에 대한 것인가요?"
```

### 검색 청크 수 변경
```python
# langchain_rag_ex.py:142 수정
retriever = vectordb.as_retriever(search_kwargs={"k": 6})  # 4 → 6
```

### 청크 크기 조정
```python
# langchain_rag_ex.py:92 수정
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 800 → 1000 (더 큰 청크)
    chunk_overlap=200  # 150 → 200 (더 많은 중복)
)
```

### LLM 모델 변경
```python
# langchain_rag_ex.py:175 수정
llm = ChatOpenAI(model="gpt-4")  # 더 강력하지만 비용 증가
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

## 문제 해결

### 청크가 0개 생성될 때
- PDF 파일이 비어있거나 파싱 실패
- `chunk_size`를 400-800 사이로 조정
- TXT 파일로 먼저 테스트

### API 키 에러
```bash
# .env 파일 확인
cat .env

# 환경변수 로드 확인
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

### 메모리 부족
- `chunk_size` 증가 (청크 수 감소)
- `k` 값 감소 (검색 청크 수 감소)

## 프로젝트 파일

```
langchain-ex/
├── .env                    # API 키 (보안 주의!)
├── .python-version         # Python 3.12.7
├── .venv/                  # 가상환경
├── docs/                   # 문서 폴더
│   └── 6조_VM_발표자료_최종.pdf
├── langchain_rag_ex.py     # 메인 코드 (상세 주석 포함)
├── requirements.txt        # 의존성 목록
└── README.md              # 이 파일
```

## 다음 단계

1. **대화 기록 추가**: ConversationBufferMemory 사용
2. **스트리밍 응답**: `llm.stream()` 사용
3. **웹 UI**: Streamlit/Gradio 연동
4. **벡터DB 영속화**: Chroma의 `persist_directory` 설정
5. **멀티턴 대화**: 이전 대화 맥락 유지

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB](https://docs.trychroma.com/)
