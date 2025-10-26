#!/usr/bin/env python3
"""
간단한 로깅 시스템 테스트 스크립트

개선된 로깅 시스템의 시각화 기능을 테스트합니다.
"""

from chatbot_logger import ChatbotLogger, LogViewer
from datetime import datetime

def create_sample_log():
    """샘플 쿼리 로그 생성"""
    query_log = {
        "query_id": "query_1",
        "timestamp": datetime.now().isoformat(),
        "user_input": "LangChain의 RAG 시스템은 어떻게 작동하나요?",
        "processing_steps": [
            {
                "step_type": "INPUT",
                "description": "User input received",
                "timestamp": datetime.now().isoformat(),
                "data": {"input": "LangChain의 RAG 시스템은 어떻게 작동하나요?"}
            },
            {
                "step_type": "RETRIEVAL",
                "description": "Retrieved top-8 documents from vector DB",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "query": "LangChain의 RAG 시스템은 어떻게 작동하나요?",
                    "num_documents": 8,
                    "search_params": {"k": 8},
                    "documents": [
                        {
                            "rank": 1,
                            "index": 0,
                            "content_preview": "RAG(Retrieval-Augmented Generation)는 검색 기반 생성 모델로, 벡터 데이터베이스에서 관련 문서를 검색하여 LLM의 응답 품질을 향상시킵니다...",
                            "content_length": 450,
                            "metadata": {"source": "docs/langchain_guide.pdf", "page": 1}
                        },
                        {
                            "rank": 2,
                            "index": 1,
                            "content_preview": "LangChain은 임베딩 모델을 사용하여 문서를 벡터로 변환하고, 유사도 검색을 통해 관련 정보를 찾습니다...",
                            "content_length": 380,
                            "metadata": {"source": "docs/vector_db.pdf", "page": 3}
                        },
                        {
                            "rank": 3,
                            "index": 2,
                            "content_preview": "Chroma DB는 LangChain과 통합되어 효율적인 벡터 검색을 제공합니다...",
                            "content_length": 320,
                            "metadata": {"source": "docs/chromadb.pdf", "page": 2}
                        }
                    ],
                    "metrics": {
                        "avg_chunk_size": 383.33,
                        "total_context_size": 1150,
                        "min_chunk_size": 320,
                        "max_chunk_size": 450
                    }
                }
            },
            {
                "step_type": "CONTEXT_PREP",
                "description": "Context formatted for LLM with chunk flow tracking",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "context_length": 1150,
                    "num_chunks_used": 3,
                    "context_preview": "[문서 1]\nRAG(Retrieval-Augmented Generation)는 검색 기반 생성 모델로..."
                }
            },
            {
                "step_type": "LLM_REQUEST",
                "description": "Requesting LLM: gpt-4o-mini",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "prompt_length": 1500
                }
            },
            {
                "step_type": "LLM_RESPONSE",
                "description": "Received LLM response",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "response_length": 520,
                    "response": "RAG 시스템은 다음과 같이 작동합니다:\n\n1. 사용자 질문 수신\n2. 벡터 데이터베이스에서 관련 문서 검색\n3. 검색된 문서를 컨텍스트로 구성\n4. LLM에 컨텍스트와 질문 전달\n5. LLM이 컨텍스트 기반 답변 생성\n\n이를 통해 최신 정보와 도메인 특화 지식을 활용한 정확한 답변이 가능합니다.",
                    "processing_time_seconds": 1.23
                }
            },
            {
                "step_type": "COMPLETE",
                "description": "Query processing completed",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "final_response": "RAG 시스템은 다음과 같이 작동합니다:\n\n1. 사용자 질문 수신\n2. 벡터 데이터베이스에서 관련 문서 검색\n3. 검색된 문서를 컨텍스트로 구성\n4. LLM에 컨텍스트와 질문 전달\n5. LLM이 컨텍스트 기반 답변 생성\n\n이를 통해 최신 정보와 도메인 특화 지식을 활용한 정확한 답변이 가능합니다.",
                    "total_time": 2.15
                }
            }
        ],
        "final_response": "RAG 시스템은 다음과 같이 작동합니다:\n\n1. 사용자 질문 수신\n2. 벡터 데이터베이스에서 관련 문서 검색\n3. 검색된 문서를 컨텍스트로 구성\n4. LLM에 컨텍스트와 질문 전달\n5. LLM이 컨텍스트 기반 답변 생성\n\n이를 통해 최신 정보와 도메인 특화 지식을 활용한 정확한 답변이 가능합니다.",
        "status": "completed",
        "metrics": {
            "total_processing_time": 2.15,
            "llm_processing_time": 1.23
        }
    }

    return query_log


def main():
    """메인 테스트 함수"""
    print("\n" + "="*70)
    print("🧪 간단한 로깅 시스템 테스트")
    print("="*70)

    # 샘플 로그 생성
    sample_log = create_sample_log()

    # 1. 간단한 흐름 보기 테스트
    print("\n📌 테스트 1: 간단한 흐름 보기 (초보자용)")
    print("-" * 70)
    LogViewer.print_simple_flow(sample_log)

    print("\n\n" + "="*70)
    print("✅ 간단한 흐름 보기 테스트 완료!")
    print("="*70)

    print("\n" + "="*70)
    print("✅ 모든 테스트 완료!")
    print("="*70)
    print("\n💡 개선사항:")
    print("  - 간단한 흐름: 5단계로 명확한 시각화")
    print("  - 이모지 사용: 직관적인 단계 구분")
    print("  - 핵심 정보만 표시: 초보자도 이해하기 쉬움")
    print("  - 상세 정보: 필요시 개발자용 상세 로그 제공")
    print()


if __name__ == "__main__":
    main()
