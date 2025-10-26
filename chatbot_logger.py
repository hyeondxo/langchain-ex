"""
챗봇 로깅 시스템

사용자 프롬프트부터 최종 답변까지의 모든 처리 과정을 추적하고 기록합니다.
- 입력 데이터 추적
- 문서 검색 결과 기록
- 컨텍스트 구성 내역
- LLM 응답 생성 과정
- 처리 시간 및 성능 메트릭
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ChatbotLogger:
    """챗봇 처리 과정을 추적하는 로거 클래스"""

    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        """
        Args:
            log_dir: 로그 파일을 저장할 디렉토리
            session_id: 세션 식별자 (None이면 자동 생성)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 세션 ID 생성 (타임스탬프 기반)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 로그 파일 경로
        self.log_file = self.log_dir / f"session_{self.session_id}.json"

        # 세션 로그 저장소
        self.session_logs: List[Dict[str, Any]] = []

        # 현재 쿼리 로그 (진행 중인 쿼리의 임시 저장소)
        self.current_query_log: Optional[Dict[str, Any]] = None

        # 세션 시작 로그
        self._init_session()

    def _init_session(self):
        """세션 초기화 로그 작성"""
        session_info = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "queries": []
        }

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, ensure_ascii=False, indent=2)

    def start_query(self, user_input: str) -> str:
        """
        새로운 쿼리 시작 로그

        Args:
            user_input: 사용자가 입력한 프롬프트

        Returns:
            query_id: 쿼리 식별자
        """
        query_id = f"query_{len(self.session_logs) + 1}"

        self.current_query_log = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "processing_steps": [],
            "metrics": {},
            "final_response": None,
            "status": "started"
        }

        self._log_step("INPUT", "User input received", {"input": user_input})

        return query_id

    def log_retrieval(self, retrieved_docs: List[Any], search_kwargs: Dict[str, Any]):
        """
        문서 검색 결과 로그

        Args:
            retrieved_docs: 검색된 문서 리스트
            search_kwargs: 검색 파라미터 (k값 등)
        """
        docs_info = []
        for i, doc in enumerate(retrieved_docs):
            doc_data = {
                "index": i,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
            docs_info.append(doc_data)

        self._log_step(
            "RETRIEVAL",
            f"Retrieved {len(retrieved_docs)} documents",
            {
                "num_documents": len(retrieved_docs),
                "search_params": search_kwargs,
                "documents": docs_info
            }
        )

    def log_context_preparation(self, formatted_context: str):
        """
        컨텍스트 준비 로그

        Args:
            formatted_context: LLM에 전달할 포맷된 컨텍스트
        """
        self._log_step(
            "CONTEXT_PREP",
            "Context formatted for LLM",
            {
                "context_length": len(formatted_context),
                "context_preview": formatted_context[:300] + "..." if len(formatted_context) > 300 else formatted_context,
                "full_context": formatted_context
            }
        )

    def log_llm_request(self, prompt: str, model: str, temperature: float):
        """
        LLM 요청 로그

        Args:
            prompt: LLM에 전달되는 전체 프롬프트
            model: 사용하는 모델명
            temperature: 온도 파라미터
        """
        self._log_step(
            "LLM_REQUEST",
            f"Requesting LLM: {model}",
            {
                "model": model,
                "temperature": temperature,
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:400] + "..." if len(prompt) > 400 else prompt
            }
        )

    def log_llm_response(self, response: str, processing_time: Optional[float] = None):
        """
        LLM 응답 로그

        Args:
            response: LLM의 응답 텍스트
            processing_time: LLM 처리 시간 (초)
        """
        data = {
            "response_length": len(response),
            "response": response
        }

        if processing_time is not None:
            data["processing_time_seconds"] = round(processing_time, 3)
            if self.current_query_log:
                self.current_query_log["metrics"]["llm_processing_time"] = round(processing_time, 3)

        self._log_step(
            "LLM_RESPONSE",
            "Received LLM response",
            data
        )

    def complete_query(self, final_response: str, total_time: Optional[float] = None):
        """
        쿼리 완료 로그

        Args:
            final_response: 사용자에게 반환되는 최종 응답
            total_time: 전체 처리 시간 (초)
        """
        if not self.current_query_log:
            return

        self.current_query_log["final_response"] = final_response
        self.current_query_log["status"] = "completed"
        self.current_query_log["completed_at"] = datetime.now().isoformat()

        if total_time is not None:
            self.current_query_log["metrics"]["total_processing_time"] = round(total_time, 3)

        self._log_step(
            "COMPLETE",
            "Query processing completed",
            {
                "final_response": final_response,
                "total_time": total_time
            }
        )

        # 세션 로그에 추가
        self.session_logs.append(self.current_query_log)

        # 파일에 저장
        self._save_to_file()

        # 현재 쿼리 로그 초기화
        self.current_query_log = None

    def log_error(self, error_type: str, error_message: str, traceback_info: Optional[str] = None):
        """
        에러 로그

        Args:
            error_type: 에러 타입
            error_message: 에러 메시지
            traceback_info: 트레이스백 정보
        """
        if not self.current_query_log:
            return

        self.current_query_log["status"] = "error"
        self.current_query_log["error"] = {
            "type": error_type,
            "message": error_message,
            "traceback": traceback_info,
            "timestamp": datetime.now().isoformat()
        }

        self._log_step(
            "ERROR",
            f"Error occurred: {error_type}",
            {
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback_info
            }
        )

        # 에러 발생해도 로그는 저장
        self.session_logs.append(self.current_query_log)
        self._save_to_file()
        self.current_query_log = None

    def _log_step(self, step_type: str, description: str, data: Dict[str, Any]):
        """
        처리 단계 로그 기록 (내부 메서드)

        Args:
            step_type: 단계 타입 (INPUT, RETRIEVAL, CONTEXT_PREP, etc.)
            description: 단계 설명
            data: 단계별 데이터
        """
        if not self.current_query_log:
            return

        step = {
            "step_type": step_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        self.current_query_log["processing_steps"].append(step)

    def _save_to_file(self):
        """로그를 파일에 저장 (내부 메서드)"""
        try:
            # 기존 세션 정보 읽기
            with open(self.log_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # 쿼리 목록 업데이트
            session_data["queries"] = self.session_logs
            session_data["last_updated"] = datetime.now().isoformat()

            # 저장
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠ 로그 저장 실패: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 정보 반환"""
        total_queries = len(self.session_logs)
        successful_queries = sum(1 for log in self.session_logs if log.get("status") == "completed")
        failed_queries = total_queries - successful_queries

        avg_processing_time = 0
        if successful_queries > 0:
            total_time = sum(
                log.get("metrics", {}).get("total_processing_time", 0)
                for log in self.session_logs
                if log.get("status") == "completed"
            )
            avg_processing_time = total_time / successful_queries

        return {
            "session_id": self.session_id,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "average_processing_time": round(avg_processing_time, 3),
            "log_file": str(self.log_file)
        }

    @staticmethod
    def load_session_log(log_file: str) -> Dict[str, Any]:
        """
        저장된 세션 로그 파일 로드

        Args:
            log_file: 로그 파일 경로

        Returns:
            세션 로그 데이터
        """
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)


class LogViewer:
    """로그를 사람이 읽기 쉬운 형태로 출력하는 유틸리티"""

    @staticmethod
    def print_query_log(query_log: Dict[str, Any], verbose: bool = False):
        """
        쿼리 로그를 포맷하여 출력

        Args:
            query_log: 쿼리 로그 딕셔너리
            verbose: 상세 출력 여부
        """
        print("\n" + "="*80)
        print(f"🔍 Query ID: {query_log.get('query_id')}")
        print(f"⏰ Timestamp: {query_log.get('timestamp')}")
        print(f"📝 Status: {query_log.get('status')}")
        print("="*80)

        # 사용자 입력
        print(f"\n💬 User Input:")
        print(f"  {query_log.get('user_input')}")

        # 처리 단계
        print(f"\n📊 Processing Steps:")
        for i, step in enumerate(query_log.get('processing_steps', []), 1):
            step_type = step.get('step_type')
            description = step.get('description')
            timestamp = step.get('timestamp', '').split('T')[1].split('.')[0] if 'T' in step.get('timestamp', '') else ''

            print(f"\n  [{i}] {step_type} ({timestamp})")
            print(f"      {description}")

            if verbose:
                data = step.get('data', {})

                if step_type == "RETRIEVAL":
                    print(f"      Retrieved {data.get('num_documents')} documents")
                    if 'documents' in data:
                        for doc in data['documents'][:3]:  # 처음 3개만 출력
                            print(f"        - Doc {doc['index']}: {doc['content_preview'][:100]}...")

                elif step_type == "CONTEXT_PREP":
                    print(f"      Context length: {data.get('context_length')} chars")
                    if 'context_preview' in data:
                        print(f"      Preview: {data['context_preview'][:150]}...")

                elif step_type == "LLM_RESPONSE":
                    print(f"      Response length: {data.get('response_length')} chars")
                    if 'processing_time_seconds' in data:
                        print(f"      Processing time: {data['processing_time_seconds']}s")

        # 최종 응답
        if query_log.get('final_response'):
            print(f"\n✅ Final Response:")
            print(f"  {query_log['final_response']}")

        # 에러 정보
        if query_log.get('error'):
            error = query_log['error']
            print(f"\n❌ Error:")
            print(f"  Type: {error.get('type')}")
            print(f"  Message: {error.get('message')}")

        # 메트릭
        if query_log.get('metrics'):
            print(f"\n📈 Metrics:")
            for key, value in query_log['metrics'].items():
                print(f"  {key}: {value}")

        print("\n" + "="*80)

    @staticmethod
    def print_session_summary(log_file: str):
        """
        세션 전체 요약 출력

        Args:
            log_file: 로그 파일 경로
        """
        session_data = ChatbotLogger.load_session_log(log_file)

        print("\n" + "="*80)
        print(f"📚 Session Summary: {session_data.get('session_id')}")
        print("="*80)

        print(f"\n⏰ Started: {session_data.get('started_at')}")
        print(f"⏰ Last Updated: {session_data.get('last_updated')}")

        queries = session_data.get('queries', [])
        print(f"\n📊 Total Queries: {len(queries)}")

        successful = sum(1 for q in queries if q.get('status') == 'completed')
        failed = len(queries) - successful

        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")

        # 평균 처리 시간
        if successful > 0:
            total_time = sum(
                q.get('metrics', {}).get('total_processing_time', 0)
                for q in queries
                if q.get('status') == 'completed'
            )
            avg_time = total_time / successful
            print(f"\n⚡ Average Processing Time: {avg_time:.3f}s")

        # 쿼리 목록
        print(f"\n📝 Query List:")
        for q in queries:
            status_icon = "✅" if q.get('status') == 'completed' else "❌"
            print(f"  {status_icon} {q.get('query_id')}: {q.get('user_input')[:50]}...")

        print("\n" + "="*80)

    @staticmethod
    def list_available_logs(log_dir: str = "logs") -> List[str]:
        """
        사용 가능한 로그 파일 목록 반환

        Args:
            log_dir: 로그 디렉토리 경로

        Returns:
            로그 파일 경로 리스트
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            return []

        return sorted([str(f) for f in log_path.glob("session_*.json")])
