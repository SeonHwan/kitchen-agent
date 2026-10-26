"""도구 호출 지표(ToolCallAccuracy, ToolCallF1)를 60문항에 적용해 data/ragas_tool_calls.json 에 저장한다.

채점 모델을 쓰지 않는다(모델 호출 0회). 인자는 정규화하지 않고 RAGAS 그대로 비교한다
(레시피 id와 이름은 다른 값으로 센다). ToolCallAccuracy는 strict_order=True(기본)와 False 둘 다 계산한다.

기대 호출과 실제 호출이 모두 비어 있으면(범위 밖 질문) ToolCallF1은 정의되지 않는데 RAGAS는 0.0을 돌려준다
(ToolCallAccuracy는 1.0). 이 경우 f1은 None으로 저장하고 평균에서 뺀다. RAGAS가 준 값은 f1_ragas_raw에 남긴다.

사용 (프로젝트 루트에서): uv run python src/eval/score_tool_calls.py
"""
import asyncio
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1  # noqa: E402

from eval.ragas_input import to_ragas_messages, to_reference_tool_calls  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
RUNS_DIR = Path("data/runs")
OUT_PATH = Path("data/ragas_tool_calls.json")


async def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    strict = ToolCallAccuracy(strict_order=True)
    loose = ToolCallAccuracy(strict_order=False)
    f1 = ToolCallF1()

    rows = []
    for q in questions:
        run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
        messages = to_ragas_messages(run)
        reference = to_reference_tool_calls(q)
        with warnings.catch_warnings():  # 호출 수가 다르면 RAGAS가 경고를 낸다. 행마다 호출 수를 따로 기록하므로 끈다
            warnings.simplefilter("ignore")
            accuracy_strict = (await strict.ascore(user_input=messages, reference_tool_calls=reference)).value
            accuracy_loose = (await loose.ascore(user_input=messages, reference_tool_calls=reference)).value
            f1_score = (await f1.ascore(user_input=messages, reference_tool_calls=reference)).value
        actual_count = sum(len(r["tool_calls"]) for r in run["trace"])
        both_empty = len(reference) == 0 and actual_count == 0
        rows.append({
            "id": q["id"], "type": q["type"],
            "reference_calls": len(reference), "actual_calls": actual_count,
            "accuracy_strict_order": accuracy_strict, "accuracy_any_order": accuracy_loose,
            "f1": None if both_empty else f1_score, "f1_ragas_raw": f1_score,
        })

    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'유형':8} {'문항':>4} {'accuracy(순서)':>14} {'accuracy(무순서)':>16} {'F1':>7} {'F1 문항':>7}")
    for type_name in ["단일", "2~3명", "4명 이상", "범위 밖", None]:
        group = [r for r in rows if type_name is None or r["type"] == type_name]
        n = len(group)
        mean = lambda key: sum(r[key] for r in group) / n
        f1_values = [r["f1"] for r in group if r["f1"] is not None]
        f1_text = f"{sum(f1_values) / len(f1_values):>7.3f}" if f1_values else f"{'-':>7}"
        print(f"{type_name or '전체':8} {n:>4} {mean('accuracy_strict_order'):>14.3f} {mean('accuracy_any_order'):>16.3f} {f1_text} {len(f1_values):>7}")


if __name__ == "__main__":
    asyncio.run(main())
