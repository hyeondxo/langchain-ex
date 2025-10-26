# 챗봇 로깅 시스템 가이드

## 개요

이 챗봇은 사용자 프롬프트부터 최종 답변까지의 모든 처리 과정을 추적하고 기록하는 로깅 시스템을 제공합니다.

## 주요 기능

### 1. 자동 로깅
챗봇 실행 시 자동으로 다음 정보를 기록합니다:

- **사용자 입력**: 원본 프롬프트
- **문서 검색**: 검색된 문서 청크 정보 (내용, 메타데이터)
- **컨텍스트 준비**: LLM에 전달되는 포맷된 컨텍스트
- **LLM 요청**: 모델명, 파라미터, 프롬프트
- **LLM 응답**: 생성된 답변, 처리 시간
- **성능 메트릭**: 전체 처리 시간, LLM 응답 시간
- **에러 정보**: 발생한 에러의 타입, 메시지, 스택 트레이스

### 2. 세션 관리
- 각 챗봇 실행마다 고유한 세션 ID 생성
- 세션별로 별도의 로그 파일 생성 (`logs/session_YYYYMMDD_HHMMSS.json`)
- 세션 내 모든 쿼리를 시간순으로 추적

### 3. 로그 조회 기능

#### 챗봇 실행 중 조회
챗봇 대화 중 다음 명령어를 사용할 수 있습니다:

```
logs     - 마지막 쿼리의 상세 처리 과정 보기
summary  - 현재 세션의 통계 요약 보기
help     - 도움말 및 명령어 목록
```

#### 독립 실행형 로그 뷰어
저장된 로그 파일을 분석하는 별도 스크립트:

```bash
python view_logs.py
```

## 사용 예시

### 1. 챗봇 실행 및 로깅

```bash
python chatbot.py
```

챗봇이 시작되면 자동으로 로깅이 활성화됩니다:

```
📚 문서 기반 RAG 챗봇에 오신 것을 환영합니다!
============================================================

💡 사용법:
  - 문서에 대해 질문하세요
  - 'quit', 'exit', 'q'를 입력하면 종료됩니다
  - 'help'를 입력하면 도움말을 볼 수 있습니다
  - 'logs'를 입력하면 처리 로그를 볼 수 있습니다
  - 'summary'를 입력하면 세션 요약을 볼 수 있습니다

📝 로그 파일: logs/session_20250126_153045.json
```

### 2. 쿼리 처리 과정 확인

질문 후 `logs` 명령어 입력:

```
[질문 1] 당신: VM이란 무엇인가요?

🤖 챗봇: VM은 Virtual Machine의 약자로...

[질문 2] 당신: logs
```

출력 예시:
```
================================================================================
🔍 Query ID: query_1
⏰ Timestamp: 2025-01-26T15:30:50.123456
📝 Status: completed
================================================================================

💬 User Input:
  VM이란 무엇인가요?

📊 Processing Steps:

  [1] INPUT (15:30:50)
      User input received

  [2] RETRIEVAL (15:30:51)
      Retrieved 4 documents
      Retrieved 4 documents
        - Doc 0: VM은 Virtual Machine의 약자로, 하드웨어를...
        - Doc 1: 가상머신은 물리적 컴퓨터 위에서 실행되는...
        - Doc 2: VM의 주요 장점은 리소스 활용 효율성...

  [3] CONTEXT_PREP (15:30:51)
      Context formatted for LLM
      Context length: 2847 chars
      Preview: [문서 1]
VM은 Virtual Machine의 약자로...

  [4] LLM_REQUEST (15:30:51)
      Requesting LLM: gpt-4o-mini

  [5] LLM_RESPONSE (15:30:53)
      Received LLM response
      Response length: 156 chars
      Processing time: 1.823s

  [6] COMPLETE (15:30:53)
      Query processing completed

✅ Final Response:
  VM은 Virtual Machine의 약자로, 물리적 하드웨어 위에서...

📈 Metrics:
  llm_processing_time: 1.823
  total_processing_time: 2.156
```

### 3. 세션 요약 확인

```
[질문 3] 당신: summary
```

출력 예시:
```
============================================================
📊 세션 요약
============================================================
세션 ID: 20250126_153045
총 쿼리 수: 2
  ✅ 성공: 2
  ❌ 실패: 0
평균 처리 시간: 2.087초
로그 파일: logs/session_20250126_153045.json
============================================================
```

### 4. 저장된 로그 분석

```bash
python view_logs.py
```

대화형 메뉴:
```
📚 챗봇 로그 뷰어
================================================================================

📁 총 3개의 로그 파일 발견:

  [1] 20250126_153045 (logs/session_20250126_153045.json)
  [2] 20250126_142230 (logs/session_20250126_142230.json)
  [3] 20250125_091520 (logs/session_20250125_091520.json)

─────────────────────────────────────────────────────────────────────────────

선택할 로그 번호 (또는 'q'로 종료): 1
```

## 로그 파일 구조

### JSON 스키마

```json
{
  "session_id": "20250126_153045",
  "started_at": "2025-01-26T15:30:45.123456",
  "last_updated": "2025-01-26T15:35:12.789012",
  "queries": [
    {
      "query_id": "query_1",
      "timestamp": "2025-01-26T15:30:50.123456",
      "user_input": "VM이란 무엇인가요?",
      "processing_steps": [
        {
          "step_type": "INPUT",
          "description": "User input received",
          "timestamp": "2025-01-26T15:30:50.123456",
          "data": {
            "input": "VM이란 무엇인가요?"
          }
        },
        {
          "step_type": "RETRIEVAL",
          "description": "Retrieved 4 documents",
          "timestamp": "2025-01-26T15:30:51.234567",
          "data": {
            "num_documents": 4,
            "search_params": {"k": 4},
            "documents": [
              {
                "index": 0,
                "content_preview": "VM은 Virtual Machine의 약자로...",
                "content_length": 856,
                "metadata": {"source": "docs/vm_guide.pdf", "page": 1}
              }
            ]
          }
        },
        {
          "step_type": "CONTEXT_PREP",
          "description": "Context formatted for LLM",
          "timestamp": "2025-01-26T15:30:51.345678",
          "data": {
            "context_length": 2847,
            "context_preview": "[문서 1]\nVM은...",
            "full_context": "[문서 1]\nVM은 Virtual Machine..."
          }
        },
        {
          "step_type": "LLM_REQUEST",
          "description": "Requesting LLM: gpt-4o-mini",
          "timestamp": "2025-01-26T15:30:51.456789",
          "data": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "prompt_length": 3124,
            "prompt_preview": "messages=[SystemMessage..."
          }
        },
        {
          "step_type": "LLM_RESPONSE",
          "description": "Received LLM response",
          "timestamp": "2025-01-26T15:30:53.280156",
          "data": {
            "response_length": 156,
            "response": "VM은 Virtual Machine의 약자로...",
            "processing_time_seconds": 1.823
          }
        },
        {
          "step_type": "COMPLETE",
          "description": "Query processing completed",
          "timestamp": "2025-01-26T15:30:53.280567",
          "data": {
            "final_response": "VM은 Virtual Machine의 약자로...",
            "total_time": 2.156
          }
        }
      ],
      "metrics": {
        "llm_processing_time": 1.823,
        "total_processing_time": 2.156
      },
      "final_response": "VM은 Virtual Machine의 약자로...",
      "status": "completed",
      "completed_at": "2025-01-26T15:30:53.280567"
    }
  ]
}
```

## 로깅 비활성화

로깅을 비활성화하려면 챗봇 초기화 시 `enable_logging=False` 설정:

```python
chatbot = RAGChatbot(
    vectorstore_manager=vectorstore_manager,
    model="gpt-4o-mini",
    retrieval_k=4,
    enable_logging=False  # 로깅 비활성화
)
```

## 디렉토리 구조

```
langchain-ex/
├── chatbot.py              # 메인 챗봇 코드 (로깅 통합)
├── chatbot_logger.py       # 로깅 시스템 구현
├── view_logs.py            # 로그 뷰어 스크립트
├── logs/                   # 로그 파일 저장 디렉토리
│   ├── session_20250126_153045.json
│   ├── session_20250126_142230.json
│   └── session_20250125_091520.json
└── LOGGING_GUIDE.md        # 이 문서
```

## 문제 해결

### Q: 로그 파일이 생성되지 않습니다
A: `logs/` 디렉토리가 자동으로 생성됩니다. 파일 권한을 확인하세요.

### Q: 로그 파일이 너무 큽니다
A: 각 세션은 별도 파일로 저장됩니다. 오래된 로그는 수동으로 삭제하거나 아카이브하세요.

### Q: 로그에서 민감한 정보를 제거하고 싶습니다
A: `chatbot_logger.py`의 로깅 메서드를 수정하여 특정 필드를 필터링할 수 있습니다.

## 활용 사례

1. **디버깅**: 챗봇이 잘못된 답변을 한 경우, 어떤 문서가 검색되었는지 확인
2. **성능 분석**: 처리 시간 메트릭을 통해 병목 지점 파악
3. **품질 개선**: 검색된 문서와 실제 답변의 관계 분석
4. **사용 패턴 분석**: 사용자가 자주 묻는 질문 유형 파악
5. **에러 분석**: 실패한 쿼리의 원인 분석 및 개선
