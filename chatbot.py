"""
LangChain RAG 기반 대화형 챗봇

이 챗봇은 docs/ 폴더의 문서들을 학습하여 사용자의 질문에 답변합니다.
- 사용자 프롬프트 입력 받기
- 문서 기반 검색 및 답변 생성
- 대화 이력 유지 (선택적)
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

from dotenv import load_dotenv
load_dotenv()

import os
import glob
import sys
from typing import List, Optional

# LangChain 핵심 컴포넌트
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# 문서 로더
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader


# ============================================================
# 문서 처리 클래스
# ============================================================

class DocumentProcessor:
    """문서 로드 및 청크 분할을 담당하는 클래스"""

    def __init__(self, docs_dir: str = "docs", chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Args:
            docs_dir: 문서가 저장된 디렉토리 경로
            chunk_size: 각 청크의 최대 문자 수
            chunk_overlap: 인접 청크 간 중복 문자 수
        """
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_documents(self) -> List[Document]:
        """docs/ 디렉토리의 모든 PDF와 TXT 파일을 로드"""
        docs = []

        # PDF 파일 로드
        pdf_pattern = os.path.join(self.docs_dir, "*.pdf")
        for path in glob.glob(pdf_pattern):
            try:
                docs.extend(PyPDFLoader(path).load())
                print(f"✓ PDF 로드 성공: {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠ PyPDFLoader 실패, PyMuPDFLoader 시도: {os.path.basename(path)}")
                try:
                    docs.extend(PyMuPDFLoader(path).load())
                    print(f"✓ PyMuPDFLoader로 로드 성공: {os.path.basename(path)}")
                except Exception as e2:
                    print(f"✗ 로드 실패: {os.path.basename(path)} - {e2}")

        # TXT 파일 로드
        txt_pattern = os.path.join(self.docs_dir, "*.txt")
        for path in glob.glob(txt_pattern):
            try:
                docs.extend(TextLoader(path, encoding="utf-8").load())
                print(f"✓ TXT 로드 성공: {os.path.basename(path)}")
            except Exception as e:
                print(f"✗ TXT 로드 실패: {os.path.basename(path)} - {e}")

        if not docs:
            raise ValueError(
                f"'{self.docs_dir}/' 디렉토리에서 문서를 찾을 수 없습니다.\n"
                f"PDF 또는 TXT 파일을 추가해주세요."
            )

        print(f"\n📚 총 {len(docs)}개의 문서 페이지 로드 완료")
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 작은 청크로 분할"""
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "청크 생성 실패. 문서가 비어있거나 파싱에 실패했습니다.\n"
                "chunk_size를 조정하거나 문서 내용을 확인해주세요."
            )

        print(f"📄 {len(chunks)}개의 청크로 분할 완료 (크기: {self.chunk_size}, 중복: {self.chunk_overlap})")
        return chunks


# ============================================================
# 벡터 저장소 관리 클래스
# ============================================================

class VectorStoreManager:
    """벡터 데이터베이스 구축 및 관리를 담당하는 클래스"""

    def __init__(
        self,
        collection_name: str = "chatbot_docs",
        persist_directory: Optional[str] = "./chroma_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Args:
            collection_name: Chroma 컬렉션 이름
            persist_directory: 벡터DB 저장 디렉토리 (None이면 메모리에만 저장)
            embedding_model: OpenAI 임베딩 모델명
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectordb = None

    def build_vectorstore(self, chunks: List[Document]) -> Chroma:
        """청크들로부터 벡터 데이터베이스 구축"""
        print("\n🔧 벡터 데이터베이스 구축 중...")

        if self.persist_directory:
            # 기존 벡터DB가 있는지 확인
            if os.path.exists(self.persist_directory):
                print(f"⚠ 기존 벡터DB 발견: {self.persist_directory}")
                response = input("기존 데이터를 삭제하고 새로 구축할까요? (y/N): ").strip().lower()
                if response == 'y':
                    import shutil
                    shutil.rmtree(self.persist_directory)
                    print("✓ 기존 데이터 삭제 완료")
                else:
                    print("⚠ 기존 벡터DB를 로드합니다.")
                    self.vectordb = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    return self.vectordb

        # 새로운 벡터DB 생성
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        print(f"✓ 벡터DB 구축 완료 ({len(chunks)}개 청크 저장)")
        if self.persist_directory:
            print(f"✓ 벡터DB 저장 위치: {self.persist_directory}")

        return self.vectordb

    def load_vectorstore(self) -> Chroma:
        """저장된 벡터 데이터베이스 로드"""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            raise ValueError(
                f"벡터DB를 찾을 수 없습니다: {self.persist_directory}\n"
                "먼저 문서를 처리하여 벡터DB를 생성해주세요."
            )

        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"✓ 기존 벡터DB 로드 완료: {self.persist_directory}")
        return self.vectordb

    def get_retriever(self, k: int = 4):
        """벡터DB로부터 Retriever 생성"""
        if not self.vectordb:
            raise ValueError("벡터DB가 초기화되지 않았습니다.")

        return self.vectordb.as_retriever(search_kwargs={"k": k})


# ============================================================
# RAG 챗봇 클래스
# ============================================================

class RAGChatbot:
    """RAG 기반 대화형 챗봇"""

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model: str = "gpt-4o-mini",
        retrieval_k: int = 4,
        temperature: float = 0.0
    ):
        """
        Args:
            vectorstore_manager: VectorStoreManager 인스턴스
            model: 사용할 LLM 모델명
            retrieval_k: 검색할 청크 개수
            temperature: LLM 응답의 창의성 (0=결정적, 1=창의적)
        """
        self.vectorstore_manager = vectorstore_manager
        self.retrieval_k = retrieval_k
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = StrOutputParser()

        # 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 문서 기반 AI 어시스턴트입니다. "
             "주어진 컨텍스트만을 근거로 정확하고 간결하게 한국어로 답변하세요. "
             "컨텍스트에 없는 정보는 '제공된 문서에서는 해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
             "컨텍스트:\n{context}"),
            ("user", "{question}")
        ])

        # RAG 체인 구성
        self.rag_chain = None
        self._build_chain()

    def _build_chain(self):
        """RAG 체인 구성"""
        retriever = self.vectorstore_manager.get_retriever(k=self.retrieval_k)

        def format_docs(docs_):
            return "\n\n".join([f"[문서 {i+1}]\n{d.page_content}" for i, d in enumerate(docs_)])

        self.rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | self.parser
        )

    def ask(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        if not question.strip():
            return "질문을 입력해주세요."

        try:
            answer = self.rag_chain.invoke(question)
            return answer
        except Exception as e:
            return f"오류 발생: {str(e)}"

    def chat_loop(self):
        """대화형 인터페이스 실행"""
        print("\n" + "="*60)
        print("📚 문서 기반 RAG 챗봇에 오신 것을 환영합니다!")
        print("="*60)
        print("\n💡 사용법:")
        print("  - 문서에 대해 질문하세요")
        print("  - 'quit', 'exit', 'q'를 입력하면 종료됩니다")
        print("  - 'help'를 입력하면 도움말을 볼 수 있습니다\n")

        conversation_count = 0

        while True:
            try:
                # 사용자 입력 받기
                user_input = input(f"\n[질문 {conversation_count + 1}] 당신: ").strip()

                # 종료 명령
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break

                # 도움말
                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                # 빈 입력 처리
                if not user_input:
                    print("⚠ 질문을 입력해주세요.")
                    continue

                # 답변 생성
                print("\n🤖 챗봇: ", end="", flush=True)
                answer = self.ask(user_input)
                print(answer)

                conversation_count += 1

            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다. (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n✗ 오류 발생: {e}")
                continue

    def _show_help(self):
        """도움말 출력"""
        help_text = """
        📖 도움말
        ─────────────────────────────────────────────────────
        이 챗봇은 docs/ 폴더의 문서를 학습하여 답변합니다.

        사용 예시:
          • "문서의 핵심 내용을 요약해주세요"
          • "VM이란 무엇인가요?"
          • "이 프로젝트의 목적은 무엇인가요?"

        명령어:
          • help  - 이 도움말 표시
          • quit  - 챗봇 종료 (또는 exit, q)

        주의사항:
          • 챗봇은 제공된 문서의 내용만을 기반으로 답변합니다
          • 문서에 없는 내용은 답변하지 않습니다
        ─────────────────────────────────────────────────────
        """
        print(help_text)


# ============================================================
# 메인 실행 함수
# ============================================================

def main():
    """챗봇 초기화 및 실행"""

    print("🚀 RAG 챗봇 초기화 중...\n")

    try:
        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ OPENAI_API_KEY가 설정되지 않았습니다.")
            print("  .env 파일에 API 키를 추가해주세요.")
            sys.exit(1)

        # 1. 문서 처리
        doc_processor = DocumentProcessor(docs_dir="docs")
        documents = doc_processor.load_documents()
        chunks = doc_processor.split_documents(documents)

        # 2. 벡터 저장소 구축 또는 로드
        vectorstore_manager = VectorStoreManager(
            collection_name="chatbot_docs",
            persist_directory="./chroma_db"
        )

        # 기존 벡터DB가 있으면 로드 여부 선택
        if os.path.exists("./chroma_db"):
            print("\n💾 기존 벡터DB 발견")
            response = input("기존 데이터를 사용할까요? (Y/n): ").strip().lower()
            if response in ['', 'y', 'yes']:
                vectorstore_manager.load_vectorstore()
            else:
                vectorstore_manager.build_vectorstore(chunks)
        else:
            vectorstore_manager.build_vectorstore(chunks)

        # 3. 챗봇 생성 및 실행
        chatbot = RAGChatbot(
            vectorstore_manager=vectorstore_manager,
            model="gpt-4o-mini",
            retrieval_k=4
        )

        print("\n✓ 챗봇 준비 완료!\n")

        # 대화형 루프 시작
        chatbot.chat_loop()

    except ValueError as e:
        print(f"\n✗ 설정 오류: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
