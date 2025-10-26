"""
LangChain RAG(Retrieval Augmented Generation) 실습 코드

RAG란?
- 문서에서 관련 정보를 검색(Retrieval)해서 LLM에게 제공하여 더 정확한 답변을 생성(Generation)하는 기술
- 할루시네이션(거짓 정보 생성) 방지 및 최신/전문 정보 활용 가능

주요 단계:
1. 문서 로드 (PDF, TXT)
2. 텍스트 청크(조각)로 분할
3. 청크를 벡터(숫자 배열)로 변환 → 벡터DB 저장
4. 질문을 벡터로 변환 → 유사한 청크 검색
5. 검색된 청크를 LLM에게 컨텍스트로 제공 → 답변 생성
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

# .env 파일에서 환경변수(API 키 등) 로드
from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY를 .env 파일에서 읽어옴

# LangChain 핵심 컴포넌트
from langchain_chroma import Chroma  # 벡터 데이터베이스 (문서의 의미를 숫자로 저장)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI 임베딩/LLM 사용
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 문서를 작은 조각으로 분할
from langchain_core.prompts import ChatPromptTemplate  # LLM에게 보낼 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser  # LLM 응답을 문자열로 파싱
from langchain_core.runnables import RunnablePassthrough  # 체인에서 입력값 그대로 전달

# 문서 로더: 다양한 형식의 파일 읽기
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader

# 유틸리티
import glob  # 파일 패턴 매칭 (예: "docs/*.pdf")
import sys  # 시스템 종료 등

# ============================================================
# 1단계: 문서 로드 (docs/ 폴더의 PDF, TXT 파일 읽기)
# ============================================================

# 로드된 문서를 저장할 빈 리스트
docs = []

# docs/ 폴더의 모든 PDF 파일 처리
# glob.glob()은 패턴에 맞는 모든 파일 경로를 리스트로 반환
for path in glob.glob("docs/*.pdf"):
    try:
        # PyPDFLoader: 기본 PDF 로더 (빠르지만 일부 PDF에서 오류 발생 가능)
        docs.extend(PyPDFLoader(path).load())  # load()는 페이지별로 Document 객체 리스트 반환
        print(f"[성공] PyPDFLoader로 로드: {path}")
    except Exception as e:
        # PyPDFLoader 실패 시 PyMuPDFLoader로 재시도
        print(f"[경고] PyPDFLoader 실패: {path} → PyMuPDFLoader로 재시도 ({e})")
        try:
            # PyMuPDFLoader: 더 강력한 PDF 파싱 라이브러리 (복잡한 PDF도 처리 가능)
            docs.extend(PyMuPDFLoader(path).load())
            print(f"[성공] PyMuPDFLoader로 로드: {path}")
        except Exception as e2:
            # 두 로더 모두 실패 시 경고만 출력하고 계속 진행
            print(f"[경고] PyMuPDFLoader도 실패: {path} ({e2})")

# docs/ 폴더의 모든 TXT 파일도 로드
# TextLoader는 일반 텍스트 파일을 읽음 (인코딩 UTF-8 지정)
for path in glob.glob("docs/*.txt"):
    docs.extend(TextLoader(path, encoding="utf-8").load())

# 문서 로드 검증: 문서가 하나도 없으면 에러 메시지 출력 후 종료
if not docs:
    print("[오류] docs/에서 읽어들인 문서가 없습니다. PDF가 깨졌거나 파일 패턴이 틀렸습니다.")
    print("      우선 docs/sample.txt를 만들어 내용 넣고 다시 실행해 보세요.")
    sys.exit(1)  # 프로그램 종료 (exit code 1 = 에러)

# 디버깅: 각 문서의 텍스트 길이를 출력 (처음 10개만)
# d.page_content는 각 Document 객체의 실제 텍스트 내용
print([len(d.page_content) for d in docs][:10])

# ============================================================
# 2단계: 텍스트 청크 분할 (긴 문서를 작은 조각으로 나누기)
# ============================================================

# 왜 청크로 나눌까?
# - LLM의 입력 토큰 제한 (GPT-4o-mini는 128K 토큰이지만 효율적 처리를 위해 작게 나눔)
# - 검색 정확도 향상 (큰 문서보다 작은 조각이 질문과 더 정확히 매칭됨)
# - 비용 절감 (필요한 부분만 LLM에게 전달)

# RecursiveCharacterTextSplitter: 재귀적으로 텍스트를 분할하는 스플리터
# - chunk_size=800: 각 청크의 최대 문자 수 (약 200-250 토큰)
# - chunk_overlap=150: 인접 청크 간 중복 문자 수 (문맥 연결을 위해 중복 허용)
#   예: "...안녕하세요. 제 이름은..." → 다음 청크도 "제 이름은..."부터 시작
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# split_documents(): 문서 리스트를 청크 리스트로 변환
# 원본: [Document(32페이지), Document(5페이지)] → 결과: [청크1, 청크2, ..., 청크N]
chunks = splitter.split_documents(docs)

# 청크 생성 결과 출력
print(f"[정보] 로드된 문서 수: {len(docs)}, 생성된 청크 수: {len(chunks)}")

# 청크 생성 검증: 청크가 없으면 문서가 비어있거나 파싱 실패
if not chunks:
    print("[오류] 청크가 0개입니다. 문서가 비어있거나 PDF 파싱이 실패했습니다.")
    print("      TXT로 먼저 테스트하거나 chunk_size를 400~800 사이로 조절해 보세요.")
    sys.exit(1)

# ============================================================
# 3단계: 임베딩 & 벡터 데이터베이스 구축
# ============================================================

# 임베딩(Embedding)이란?
# - 텍스트를 숫자 벡터(배열)로 변환하는 과정
# - 예: "고양이" → [0.2, -0.5, 0.8, ...] (실제로는 1536차원)
# - 의미가 비슷한 텍스트는 비슷한 벡터로 변환됨
#   "고양이"와 "강아지"의 벡터는 가까움 / "고양이"와 "자동차"는 멈

# OpenAIEmbeddings: OpenAI의 text-embedding-3-small 모델 사용
# - .env 파일의 OPENAI_API_KEY 필요
# - 1536차원 벡터 생성 (각 차원은 -1.0 ~ 1.0 사이 실수값)
emb = OpenAIEmbeddings()

# Chroma: 벡터 데이터베이스 (의미적으로 유사한 텍스트를 빠르게 검색)
# - collection_name: 데이터베이스 컬렉션 이름 (여러 프로젝트 관리 시 구분용)
# - embedding_function: 텍스트 → 벡터 변환 함수
vectordb = Chroma(collection_name="my_docs", embedding_function=emb)

# 청크들을 벡터DB에 저장
# 내부 동작:
#   1. 각 청크 텍스트 → OpenAI API로 임베딩 → 1536차원 벡터 생성
#   2. 벡터 + 원본 텍스트 + 메타데이터를 ChromaDB에 저장
#   3. 나중에 질문이 들어오면 질문도 벡터로 변환 → 유사한 벡터 검색
vectordb.add_documents(chunks)

# ============================================================
# 4단계: Retriever 생성 (검색 인터페이스)
# ============================================================

# Retriever: 질문(문자열)을 받아서 관련 문서(청크)를 검색하는 객체
# - as_retriever(): 벡터DB를 Retriever 인터페이스로 변환
# - search_kwargs={"k": 4}: 가장 유사한 상위 4개 청크만 반환
#   (k 값이 크면 더 많은 정보를 제공하지만 LLM 처리 시간/비용 증가)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ============================================================
# 5단계: 프롬프트 템플릿 정의
# ============================================================

# ChatPromptTemplate: LLM에게 보낼 프롬프트의 구조를 정의
# - from_messages(): 대화 형식의 프롬프트 생성 (시스템 메시지 + 사용자 메시지)
# - {context}, {question}: 나중에 실제 값으로 치환될 변수

RAG_PROMPT = ChatPromptTemplate.from_messages([
    # 시스템 메시지: LLM의 역할과 행동 지침을 정의
    ("system",
     "당신은 정확한 문서 기반 어시스턴트입니다. 다음 컨텍스트만 근거로 한국어로 간결하고 정확히 답하세요. "
     "모르면 모른다고 하세요.\n\n컨텍스트:\n{context}"),

    # 사용자 메시지: 실제 질문이 들어갈 자리
    ("user", "{question}")
])

# 실제 실행 시 예시:
# context = "VM은 Virtual Machine의 약자로..."
# question = "VM이란 무엇인가요?"
# → 시스템: "당신은 정확한... 컨텍스트: VM은 Virtual Machine..."
#   사용자: "VM이란 무엇인가요?"

# ============================================================
# 6단계: RAG 체인 구성 (전체 파이프라인 연결)
# ============================================================

# ChatOpenAI: OpenAI의 LLM 모델
# - model="gpt-4o-mini": 비용 효율적이면서 성능이 좋은 모델
# - .env 파일의 OPENAI_API_KEY 사용
llm = ChatOpenAI(model="gpt-4o-mini")

# StrOutputParser: LLM의 응답(AIMessage 객체)을 문자열로 변환
parser = StrOutputParser()

# format_docs: 검색된 청크들을 하나의 문자열로 포맷팅
# 입력: [Document1, Document2, ...] → 출력: "[0] 내용1\n\n[1] 내용2\n\n..."
def format_docs(docs_):
    return "\n\n".join([f"[{i}] " + d.page_content for i, d in enumerate(docs_)])

# RAG 체인: 전체 파이프라인을 연결 (LangChain Expression Language 사용)
# | (파이프) 연산자: 왼쪽의 출력을 오른쪽의 입력으로 전달
rag_chain = (
    # 단계 1: 입력 처리 및 검색
    {
        # "context" 키: 질문 → Retriever로 검색 → format_docs로 포맷팅
        "context": retriever | format_docs,

        # "question" 키: 원본 질문을 그대로 통과 (RunnablePassthrough)
        "question": RunnablePassthrough()
    }
    # 단계 2: 프롬프트 생성 (검색된 context + 질문을 템플릿에 삽입)
    | RAG_PROMPT

    # 단계 3: LLM 실행 (프롬프트를 받아 답변 생성)
    | llm

    # 단계 4: 응답 파싱 (AIMessage → 문자열 변환)
    | parser
)

# 전체 흐름 예시:
# 입력: "VM이란?"
# → retriever: 벡터 검색으로 관련 청크 4개 찾음
# → format_docs: "[0] VM은...\n\n[1] 가상머신은..."
# → RAG_PROMPT: 시스템+사용자 메시지 생성
# → llm: GPT-4o-mini가 답변 생성
# → parser: 문자열로 변환
# 출력: "VM은 Virtual Machine의 약자로..."

# ============================================================
# 7단계: 실행 (메인 프로그램)
# ============================================================

# if __name__ == "__main__":
# - 이 파일이 직접 실행될 때만 아래 코드 실행 (다른 파일에서 import 시 실행 안 됨)
# - Python의 관례: 테스트/데모 코드는 이 블록 안에 작성
if __name__ == "__main__":
    # 테스트 질문 정의
    q = "이 문서의 핵심 요점을 한 문장으로 말해 주세요."

    # RAG 체인 실행
    # invoke(q): 질문 문자열을 입력으로 받아 전체 파이프라인 실행
    # - 질문 임베딩 → 벡터 검색 → 관련 청크 찾기
    # - 청크 + 질문을 프롬프트에 삽입
    # - LLM 호출하여 답변 생성
    # - 답변을 문자열로 반환
    answer = rag_chain.invoke(q)

    # 답변 출력
    print(answer)

    # 다른 질문을 시도하려면:
    # print(rag_chain.invoke("VM의 장점은 무엇인가요?"))
    # print(rag_chain.invoke("이 문서는 어떤 주제를 다루나요?"))
