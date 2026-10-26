"""생성 지표(Faithfulness, FactualCorrectness)를 적용한다.

- Faithfulness: 최종 답의 문장이 도구 결과에 근거하는가. retrieved_contexts = 그 질문에서 호출한 모든 도구의 결과 원문
  (검색 지표와 달리 검색 도구만이 아니다). 도구를 부르지 않은 질문(범위 밖)은 retrieved_contexts가 비어 RAGAS가 오류를 내므로 제외한다.
- FactualCorrectness(mode f1): 최종 답을 reference(기준 답)와 대조한다. 60문항 전체.
  기준 답은 템플릿 문장에 계산 값을 채운 것이고 사람이 검증하지 않았다.

질문별로 data/ragas_generation/<질문 id>.json 에 점수와 채점 호출의 프롬프트·응답 원문을 저장하고, 있으면 건너뛴다.
nan은 JSON에 못 쓰므로 None으로 저장한다.

사용 (프로젝트 루트에서):
  uv run python src/eval/score_generation.py --sizes   # 모델 호출 없이 도구 결과 글자 수만 센다
  uv run python src/eval/score_generation.py Q11       # 지정한 질문만
  uv run python src/eval/score_generation.py           # 전부
"""
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.metrics.collections import FactualCorrectness, Faithfulness  # noqa: E402

from eval.judge import make_judge, record_calls  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
RUNS_DIR = Path("data/runs")
OUT_DIR = Path("data/ragas_generation")


def tool_results(run):
    """실행 기록에서 호출한 모든 도구의 결과 원문을 호출 순서대로 모은다. 도구 이름을 앞에 붙여 어느 도구의 결과인지 보이게 한다."""
    results = []
    for record in run["trace"]:
        for call in record["tool_calls"]:
            results.append(f"[{call['name']}] {call['result']}")
    return results


def to_none_if_nan(value):
    return None if math.isnan(value) else value


async def score_question(q, run, llm, calls):
    scores = {}
    contexts = tool_results(run)

    if contexts:
        before = len(calls)
        result = await Faithfulness(llm=llm).ascore(user_input=q["question"], response=run["final"], retrieved_contexts=contexts)
        scores["faithfulness"] = {"value": to_none_if_nan(result.value), "judge_calls": len(calls) - before}

    before = len(calls)
    result = await FactualCorrectness(llm=llm).ascore(response=run["final"], reference=q["reference"])
    scores["factual_correctness"] = {"value": to_none_if_nan(result.value), "judge_calls": len(calls) - before}
    return scores


def show_sizes(questions):
    total = 0
    biggest = []
    for q in questions:
        run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
        size = sum(len(r) for r in tool_results(run))
        total += size
        biggest.append((size, q["id"], q["type"]))
    biggest.sort(reverse=True)
    print(f"도구 결과 글자 수 합계 {total:,}자 (Faithfulness 호출 1회당 이 중 해당 질문의 결과가 들어간다)")
    print("가장 큰 질문:", [(i, t, f"{s:,}") for s, i, t in biggest[:5]])


async def main():
    args = sys.argv[1:]
    wanted_ids = [a for a in args if not a.startswith("--")]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if wanted_ids:
        questions = [q for q in questions if q["id"] in wanted_ids]

    if "--sizes" in args:
        show_sizes(questions)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q in questions:
        path = OUT_DIR / f"{q['id']}.json"
        if path.exists():
            print(f"{q['id']} 건너뜀 (기록 있음)")
            continue
        run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
        llm = make_judge()
        calls = record_calls(llm)
        scores = await score_question(q, run, llm, calls)
        record = {"id": q["id"], "type": q["type"], "question": q["question"], "scores": scores, "judge_calls": calls}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        values = {k: v["value"] for k, v in scores.items()}
        print(f"{q['id']} 완료  호출 {len(calls)}회  {values}")


if __name__ == "__main__":
    asyncio.run(main())
