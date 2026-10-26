"""검색 지표(ContextPrecision, ContextRecall, NoiseSensitivity)를 검색 도구를 부른 질문에 적용한다.

적용 대상: 실행 기록에서 search_recipes / find_substitute / storage_tip 중 하나라도 호출한 질문.
retrieved_contexts는 검색 도구가 돌려준 문서의 '임베딩 문서 문자열'이다(eval/contexts.py, decisions 15절).
response에는 최종 답(final) 전체를 넣는다. 검색 외 도구(가격·영양 등)의 문장도 포함되므로 NoiseSensitivity에서
문서에 근거가 없는 문장이 '틀린 문장'으로 셀 수 있다(해석할 때 주의).

질문별로 결과를 data/ragas_retrieval/<질문 id>.json 에 저장하고, 이미 있으면 건너뛴다(중간에 끊겨도 이어 돌 수 있다).
채점 모델 호출의 프롬프트와 응답 원문도 같은 파일에 남긴다.

사용 (프로젝트 루트에서):
  uv run python src/eval/score_retrieval.py --dry-run     # 채점 모델 호출 없이 호출 수와 프롬프트 크기만 센다
  uv run python src/eval/score_retrieval.py Q11           # 지정한 질문만
  uv run python src/eval/score_retrieval.py               # 전부
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.metrics.collections import ContextPrecision, ContextRecall, NoiseSensitivity  # noqa: E402

from eval.contexts import SEARCH_TOOLS, load_documents, retrieved_contexts  # noqa: E402
from eval.judge import make_judge, record_calls  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
RUNS_DIR = Path("data/runs")
OUT_DIR = Path("data/ragas_retrieval")


def fake_response(response_model):
    """dry-run용 가짜 응답: 지표 코드가 끝까지 돌 수 있게 최소한의 값으로 채운다. 점수는 의미 없다."""
    name = response_model.__name__
    if name == "ContextPrecisionOutput":
        return response_model(reason="dry-run", verdict=1)
    if name == "ContextRecallOutput":
        return response_model(classifications=[{"statement": "dry-run", "reason": "dry-run", "attributed": 1}])
    if name == "StatementGeneratorOutput":
        return response_model(statements=["dry-run 문장 1", "dry-run 문장 2"])
    if name == "StatementFaithfulnessOutput":
        return response_model(statements=[{"statement": "x", "reason": "dry-run", "verdict": 1},
                                          {"statement": "y", "reason": "dry-run", "verdict": 0}])
    raise ValueError(f"dry-run 가짜 응답이 없는 응답 모델: {name}")


def used_search_tool(run):
    for record in run["trace"]:
        for call in record["tool_calls"]:
            if call["name"] in SEARCH_TOOLS:
                return True
    return False


async def score_question(q, run, documents, llm, calls):
    ids, contexts = retrieved_contexts(run, documents)
    precision = ContextPrecision(llm=llm)
    recall = ContextRecall(llm=llm)
    noise = NoiseSensitivity(llm=llm)  # mode 기본값 "relevant"

    scores = {}
    for name, metric_call in [
        ("context_precision", precision.ascore(user_input=q["question"], reference=q["reference"], retrieved_contexts=contexts)),
        ("context_recall", recall.ascore(user_input=q["question"], retrieved_contexts=contexts, reference=q["reference"])),
        ("noise_sensitivity", noise.ascore(user_input=q["question"], response=run["final"], reference=q["reference"], retrieved_contexts=contexts)),
    ]:
        before = len(calls)
        scores[name] = {"value": (await metric_call).value, "judge_calls": len(calls) - before}
    return ids, scores


async def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    wanted_ids = [a for a in args if not a.startswith("--")]

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    documents = load_documents()
    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_calls = 0
    total_chars = 0
    for q in questions:
        if wanted_ids and q["id"] not in wanted_ids:
            continue
        run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
        if not used_search_tool(run):
            continue
        path = OUT_DIR / f"{q['id']}.json"
        if not dry_run and path.exists():
            print(f"{q['id']} 건너뜀 (기록 있음)")
            continue

        llm = make_judge()
        calls = record_calls(llm, dry_run, fake_response)
        ids, scores = await score_question(q, run, documents, llm, calls)
        chars = sum(len(c["prompt"]) for c in calls)
        total_calls += len(calls)
        total_chars += chars

        if dry_run:
            per_metric = {k: v["judge_calls"] for k, v in scores.items()}
            print(f"{q['id']} 문서 {len(ids)}개  채점 호출 {len(calls)}회 {per_metric}  프롬프트 {chars:,}자")
        else:
            record = {"id": q["id"], "question": q["question"], "retrieved_ids": ids,
                      "reference_docs": q["reference_docs"], "scores": scores, "judge_calls": calls}
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            values = {k: round(v["value"], 3) for k, v in scores.items()}
            print(f"{q['id']} 완료  호출 {len(calls)}회  {values}")

    label = "dry-run 합계" if dry_run else "합계"
    print(f"{label}: 채점 호출 {total_calls}회, 프롬프트 {total_chars:,}자")


if __name__ == "__main__":
    asyncio.run(main())
