"""이전 방식 60문항 기록(data/runs/multi.jsonl)과 지금 MCP 경유 기록(data/runs/Q*.json)을 같은 기준으로 비교한다.

모델 호출 없음. 이전 Phase 4의 지표 정의(eval/metrics.py)를 그대로 쓴다: 도구 이름 F1, 인자 일치율(query·top_k 제외,
레시피 id와 이름은 같은 것으로 봄), 에이전트 선택 F1. 호출 수·토큰·시간은 기록된 값을 그대로 쓴다.

사용 (프로젝트 루트에서): uv run python src/eval/compare_runs.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import agent_scores, tool_call_scores  # noqa: E402

QUESTIONS_PATH = Path("data/questions.json")
OLD_PATH = Path("data/runs/multi.jsonl")
RUNS_DIR = Path("data/runs")

# 도구 이름 → 소유 에이전트 (kitchen-mcp TOOL_GROUPS와 같다)
TOOL_OWNER = {
    "search_recipes": "recipe", "get_recipe": "recipe", "filter_recipes": "recipe",
    "list_pantry": "pantry", "match_recipe_to_pantry": "pantry", "expiring_soon": "pantry",
    "nutrition_of": "nutrition", "check_allergens": "nutrition", "diet_rule_check": "nutrition",
    "find_substitute": "substitute", "storage_tip": "substitute",
    "price_of": "shopping", "build_shopping_list": "shopping", "budget_check": "shopping",
    "estimate_total_time": "cooking", "plan_parallel_steps": "cooking",
}


def old_row(row):
    calls = [{"name": c["name"], "args": c["args"], "agent": c["agent"]} for c in row["tool_calls"]]
    return {"calls": calls, "model_calls": row["model_calls"], "input": row["input_tokens"],
            "output": row["output_tokens"], "seconds": row["elapsed_sec"]}


def new_row(run):
    calls = []
    for record in run["trace"]:
        for c in record["tool_calls"]:
            calls.append({"name": c["name"], "args": c["args"], "agent": TOOL_OWNER[c["name"]]})
    return {"calls": calls, "model_calls": sum(r["model_calls"] for r in run["trace"]),
            "input": sum(r["input_tokens"] for r in run["trace"]),
            "output": sum(r["output_tokens"] for r in run["trace"]), "seconds": run["seconds"]}


def summarize(rows):
    n = len(rows)
    mean = lambda f: sum(f(r) for r in rows) / n
    return {
        "n": n,
        "model_calls": mean(lambda r: r["row"]["model_calls"]),
        "input": mean(lambda r: r["row"]["input"]),
        "output": mean(lambda r: r["row"]["output"]),
        "seconds": mean(lambda r: r["row"]["seconds"]),
        "tool_calls": mean(lambda r: len(r["row"]["calls"])),
        "name_f1": mean(lambda r: r["tools"]["name_f1"]),
        "arg_match": mean(lambda r: r["tools"]["arg_match"]),
        "agent_f1": mean(lambda r: r["agents"]["agent_f1"]),
    }


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    old = {}
    for line in OLD_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        old[row["id"]] = old_row(row)

    scored = {"이전(직접 호출)": [], "지금(MCP 경유)": []}
    for q in questions:
        run = json.loads((RUNS_DIR / f"{q['id']}.json").read_text(encoding="utf-8"))
        for label, row in [("이전(직접 호출)", old[q["id"]]), ("지금(MCP 경유)", new_row(run))]:
            scored[label].append({
                "id": q["id"], "type": q["type"], "row": row,
                "tools": tool_call_scores(q["reference_tool_calls"], row["calls"]),
                "agents": agent_scores(q["expected_agents"], row["calls"]),
            })

    header = f"{'구분':14} {'n':>3} {'호출':>5} {'입력':>7} {'출력':>6} {'초':>5} {'도구':>5} {'이름F1':>7} {'인자':>6} {'에이전트F1':>10}"
    for type_name in [None, "단일", "2~3명", "4명 이상", "범위 밖"]:
        print(f"== {type_name or '전체'}")
        print(header)
        for label, rows in scored.items():
            g = [r for r in rows if type_name is None or r["type"] == type_name]
            s = summarize(g)
            print(f"{label:14} {s['n']:>3} {s['model_calls']:>5.2f} {s['input']:>7,.0f} {s['output']:>6,.0f} {s['seconds']:>5.1f} "
                  f"{s['tool_calls']:>5.2f} {s['name_f1']:>7.3f} {s['arg_match']:>6.3f} {s['agent_f1']:>10.3f}")

    # 질문별로 도구 이름 집합이 같은지: 이전과 지금이 같은 도구를 불렀는가
    same_names = 0
    same_exact = 0
    for a, b in zip(scored["이전(직접 호출)"], scored["지금(MCP 경유)"]):
        if {c["name"] for c in a["row"]["calls"]} == {c["name"] for c in b["row"]["calls"]}:
            same_names += 1
        key = lambda r: sorted((c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False)) for c in r["row"]["calls"])
        if key(a) == key(b):
            same_exact += 1
    print(f"\n질문별 일치: 호출한 도구 이름 집합이 같은 질문 {same_names}/60, 호출(이름+인자)이 완전히 같은 질문 {same_exact}/60")


if __name__ == "__main__":
    main()
