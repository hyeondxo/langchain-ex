#!/usr/bin/env python3
"""
로그 뷰어 스크립트

챗봇이 생성한 로그 파일을 조회하고 분석하는 독립 실행형 스크립트
"""

import sys
from pathlib import Path
from chatbot_logger import ChatbotLogger, LogViewer


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("📚 챗봇 로그 뷰어")
    print("="*80)

    # 사용 가능한 로그 파일 목록
    available_logs = LogViewer.list_available_logs()

    if not available_logs:
        print("\n⚠ 사용 가능한 로그 파일이 없습니다.")
        print("  먼저 챗봇을 실행하여 로그를 생성해주세요.")
        return

    print(f"\n📁 총 {len(available_logs)}개의 로그 파일 발견:\n")

    # 로그 파일 목록 출력
    for i, log_file in enumerate(available_logs, 1):
        log_path = Path(log_file)
        session_id = log_path.stem.replace("session_", "")
        print(f"  [{i}] {session_id} ({log_file})")

    # 사용자 선택
    while True:
        try:
            print("\n" + "─"*80)
            choice = input("\n선택할 로그 번호 (또는 'q'로 종료): ").strip()

            if choice.lower() == 'q':
                print("\n👋 로그 뷰어를 종료합니다.")
                break

            log_index = int(choice) - 1

            if log_index < 0 or log_index >= len(available_logs):
                print("⚠ 잘못된 번호입니다. 다시 선택해주세요.")
                continue

            selected_log = available_logs[log_index]

            # 로그 데이터 로드
            session_data = ChatbotLogger.load_session_log(selected_log)

            # 세션 요약 출력
            LogViewer.print_session_summary(selected_log)

            # 쿼리 목록 출력
            queries = session_data.get('queries', [])

            if not queries:
                print("\n⚠ 이 세션에는 쿼리가 없습니다.")
                continue

            # 세부 쿼리 선택
            while True:
                print("\n" + "─"*80)
                print("\n옵션:")
                print("  [번호]   - 해당 쿼리의 로그 보기")
                print("  [all]    - 모든 쿼리의 로그 보기")
                print("  [simple] - 간단히 보기 모드")
                print("  [detail] - 상세히 보기 모드")
                print("  [back]   - 로그 파일 목록으로 돌아가기")

                query_choice = input("\n선택: ").strip().lower()

                if query_choice == 'back':
                    break

                # 표시 모드 설정
                view_mode = 'simple'  # 기본값: 간단히 보기

                if query_choice == 'simple':
                    view_mode = 'simple'
                    print("\n✅ 간단히 보기 모드로 설정됨")
                    continue
                elif query_choice == 'detail':
                    view_mode = 'detail'
                    print("\n✅ 상세히 보기 모드로 설정됨")
                    continue

                if query_choice == 'all':
                    # 모든 쿼리 출력
                    print("\n로그 보기 모드를 선택하세요:")
                    print("  [1] 🎯 간단히 보기 (추천)")
                    print("  [2] 📊 상세히 보기")
                    mode_choice = input("선택 (1/2, 기본값=1): ").strip() or "1"

                    for query_log in queries:
                        if mode_choice == "1":
                            LogViewer.print_simple_flow(query_log)
                        else:
                            LogViewer.print_query_log(query_log, verbose=True)
                        print("\n" + "─"*80)
                    continue

                try:
                    query_index = int(query_choice) - 1

                    if query_index < 0 or query_index >= len(queries):
                        print("⚠ 잘못된 번호입니다.")
                        continue

                    # 선택한 쿼리의 로그 출력 (모드 선택)
                    print("\n로그 보기 모드를 선택하세요:")
                    print("  [1] 🎯 간단히 보기 (추천)")
                    print("  [2] 📊 상세히 보기")
                    mode_choice = input("선택 (1/2, 기본값=1): ").strip() or "1"

                    if mode_choice == "1":
                        LogViewer.print_simple_flow(queries[query_index])
                    else:
                        LogViewer.print_query_log(queries[query_index], verbose=True)

                except ValueError:
                    print("⚠ 잘못된 입력입니다.")

        except ValueError:
            print("⚠ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n\n👋 로그 뷰어를 종료합니다.")
            break
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
