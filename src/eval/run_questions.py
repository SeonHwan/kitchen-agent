"""질문 세트를 멀티 에이전트로 실행해 질문마다 실행 기록을 data/runs/<질문 id>.json 에 저장한다.

기록에는 최종 답, 선택된 에이전트, 부분 답, 노드별 기록(모델 호출 수·토큰·시간·도구 호출과 결과 원문)이 들어간다.
이미 기록이 있는 질문은 건너뛴다(중간에 끊겨도 이어서 돌릴 수 있다). 다시 돌리려면 그 파일을 지운다.
**모델을 호출한다.** 질문 하나가 대략 모델 호출 2~12회다.

사용 (프로젝트 루트에서):
  uv run python src/eval/run_questions.py Q01 Q36 Q51      # 지정한 질문만
  uv run python src/eval/run_questions.py                  # 60문항 전체
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import multi  # noqa: E402
from agents.mcp_tools import KitchenTools  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
RUNS_DIR = Path("data/runs")


async def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    wanted_ids = sys.argv[1:]
    if wanted_ids:
        questions = [q for q in questions if q["id"] in wanted_ids]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    async with KitchenTools() as kitchen:
        for q in questions:
            path = RUNS_DIR / f"{q['id']}.json"
            if path.exists():
                print(f"{q['id']} 건너뜀 (기록 있음)")
                continue
            run = await multi.answer(q["question"], kitchen.groups)
            record = {"id": q["id"], "question": q["question"], **run}
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            calls = sum(r["model_calls"] for r in run["trace"])
            print(f"{q['id']} 완료  에이전트 {run['selected']}  모델 호출 {calls}회  {run['seconds']}초")


if __name__ == "__main__":
    asyncio.run(main())
