"""질문을 직접 던지는 진입점. 질문을 멀티 에이전트에게 주고 선택된 에이전트와 답을 보여 준다.

대화형에서는 앞의 질문과 최종 답을 기억해 이어지는 질문("그중에 첫 번째는?")을 이해한다. 기록은 메모리에만 있다.

사용:
  uv run python src/ask.py "2~3명이 먹을 저녁 메뉴 추천해줘"      # 질문 한 개 (기억 없음)
  uv run python src/ask.py                                        # 대화형: 이어서 질문
    종료: "종료", "끝", "이제 끝낼게", "그만" 처럼 입력 전체가 종료 말일 때, 또는 q / exit / 빈 줄 / Ctrl-C
    /new = 기억만 지우고 계속
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import multi  # noqa: E402
from agents.mcp_tools import KitchenTools  # noqa: E402

# 입력 전체가 이 중 하나일 때만 종료로 본다 (질문 속의 "끝까지 익히려면?"은 종료가 아니다)
END_WORDS = r"(종료|끝|끝내자|끝낼게|끝낼래|끝낼게요|끝낼래요|끝내겠습니다|이제 끝|이제 끝낼게|이제 끝낼게요|이제 끝낼래|이제 끝내자|이제 그만|그만|그만할게|그만할게요|그만하자|나갈게|나갈래|잘 있어|bye|quit|exit|q)"
END_PATTERN = re.compile(rf"^\s*{END_WORDS}\s*[.!~]*\s*$", re.IGNORECASE)


def is_end(text):
    """입력이 대화를 끝내자는 말인지 본다. 빈 줄도 종료로 본다."""
    return text.strip() == "" or bool(END_PATTERN.match(text))


async def show(question, tool_groups, history=None):
    """질문 하나를 처리해 화면에 보여 주고 최종 답을 돌려준다."""
    run = await multi.answer(question, tool_groups, history=history)
    selected = run["selected"]
    print("\n── 선택된 에이전트:", ", ".join(selected) if selected else "(없음 — 범위 밖으로 판단)")
    print("\n" + run["final"] + "\n")
    return run["final"]


async def main():
    # MCP 서버 6개는 이 블록이 열려 있는 동안(대화 내내) 떠 있다
    async with KitchenTools() as kitchen:
        if len(sys.argv) > 1:
            await show(" ".join(sys.argv[1:]), kitchen.groups)
        else:
            history = []  # [(질문, 최종 답)] — 프로그램이 끝나면 사라진다
            print("대화형입니다. 앞의 질문과 답을 모두 기억합니다 (길어질수록 입력 토큰이 늘어납니다).")
            print("종료: '종료' / '이제 끝낼게' 등, 또는 q / 빈 줄.  기억만 지우기: /new\n")
            while True:
                q = input(f"질문[{len(history)}]> ").strip()
                if is_end(q):
                    break
                if q == "/new":
                    history = []
                    print("기억을 지웠습니다.\n")
                    continue
                answer = await show(q, kitchen.groups, history=history)
                history.append((q, answer))
            print(f"대화를 종료합니다. (질문 {len(history)}개의 기록은 사라집니다)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (EOFError, KeyboardInterrupt):
        print("\n대화를 종료합니다.")
