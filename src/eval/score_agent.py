"""에이전트 지표(AgentGoalAccuracy, TopicAdherence)를 적용한다.

- AgentGoalAccuracy: 60문항 전체. user_input = RAGAS 메시지 목록(질문, 도구 호출·결과, 최종 답), reference = reference_outcome.
- TopicAdherence: 범위 밖 질문은 제외한다(reference_topics가 빈 목록이면 RAGAS가 오류를 낸다). reference_topics를 허용 주제로 쓴다.
  주제가 하나도 추출되면 값이 nan이다. nan은 평균에서 뺀다.

질문별로 data/ragas_agent/<질문 id>.json 에 점수와 채점 호출의 프롬프트·응답 원문을 저장하고, 있으면 건너뛴다.

사용 (프로젝트 루트에서):
  uv run python src/eval/score_agent.py --dry-run      # 채점 모델 호출 없이 호출 수와 프롬프트 크기만 센다 (주제 수 1·3·5 가정)
  uv run python src/eval/score_agent.py Q11            # 지정한 질문만
  uv run python src/eval/score_agent.py                # 전부
"""
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.metrics.collections import AgentGoalAccuracyWithReference, TopicAdherence  # noqa: E402

from eval.judge import make_judge, record_calls  # noqa: E402
from eval.ragas_input import to_ragas_messages  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
RUNS_DIR = Path("data/runs")
OUT_DIR = Path("data/ragas_agent")
OUT_OF_SCOPE = "범위 밖"


def make_fake_response(topic_count):
    """dry-run용 가짜 응답. TopicAdherence의 주제 수(topic_count)는 채점 모델이 정하는 값이라 가정으로 넣는다."""

    def fake_response(response_model):
        name = response_model.__name__
        if name == "WorkflowOutput":
            return response_model(user_goal="dry-run 목표", end_state="dry-run 결과")
        if name == "CompareOutcomeOutput":
            return response_model(reason="dry-run", verdict="1")
        if name == "TopicExtractionOutput":
            return response_model(topics=[f"dry-run 주제 {i + 1}" for i in range(topic_count)])
        if name == "TopicRefusedOutput":
            return response_model(refused_to_answer=False)
        if name == "TopicClassificationOutput":
            return response_model(classifications=[True] * topic_count)
        raise ValueError(f"dry-run 가짜 응답이 없는 응답 모델: {name}")

    return fake_response


async def score_question(q, run, llm, calls):
    messages = to_ragas_messages(run)
    scores = {}

    before = len(calls)
    goal = AgentGoalAccuracyWithReference(llm=llm)
    result = await goal.ascore(user_input=messages, reference=q["reference_outcome"])
    scores["agent_goal_accuracy"] = {"value": result.value, "judge_calls": len(calls) - before}

    if q["type"] != OUT_OF_SCOPE:
        before = len(calls)
        topic = TopicAdherence(llm=llm)  # mode 기본값 f1
        result = await topic.ascore(user_input=messages, reference_topics=q["reference_topics"])
        value = None if math.isnan(result.value) else result.value  # nan은 JSON에 못 쓰므로 None으로 저장
        scores["topic_adherence"] = {"value": value, "judge_calls": len(calls) - before}
    return scores


async def dry_run(questions):
    for topic_count in (1, 3, 5):
        total_calls = 0
        total_chars = 0
        per_metric = {"agent_goal_accuracy": 0, "topic_adherence": 0}
        for q in questions:
            run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
            llm = make_judge()
            calls = record_calls(llm, dry_run=True, fake_response=make_fake_response(topic_count))
            scores = await score_question(q, run, llm, calls)
            total_calls += len(calls)
            total_chars += sum(len(c["prompt"]) for c in calls)
            for name, score in scores.items():
                per_metric[name] += score["judge_calls"]
        print(f"주제 수 {topic_count}개 가정: 채점 호출 {total_calls}회 {per_metric}  프롬프트 {total_chars:,}자")


async def main():
    args = sys.argv[1:]
    wanted_ids = [a for a in args if not a.startswith("--")]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if wanted_ids:
        questions = [q for q in questions if q["id"] in wanted_ids]

    if "--dry-run" in args:
        await dry_run(questions)
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
