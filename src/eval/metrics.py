"""Phase 6 비교용 지표(이전 Phase 4 metrics.py를 git 93bb59c에서 복원): 도구 호출 정확도, 에이전트 선택 정확도. 모델을 쓰지 않는 순수 계산이다.

정의 (docs/decisions.md에도 기록):
- 이름 정밀도/재현율/F1: 기대 도구 이름 집합 vs 실제 호출한 이름 집합. 같은 도구를 여러 번 불러도 한 번으로 센다.
- 인자 일치율: 이름이 기대에 있는 실제 호출 중, 비교 대상 인자가 기대 호출 중 하나와 같은 비율.
  검색 질의(query)와 top_k는 서술이 다양해 비교에서 뺀다. 숫자는 값이 같으면 같다(5 == 5.0).
  레시피 인자는 id(R037)와 이름이 같은 레시피를 가리키면 같다고 본다 (도구가 둘 다 받는다).
  기대에 없는 인자를 추가로 준 호출(예: servings=1)은 여전히 불일치다 — 이 한계는 의도적으로 남겼다.
- 에이전트 선택 F1: expected_agents vs 실제 호출한 도구의 소유 에이전트 집합.
- 범위 밖 질문(기대 호출 없음)은 호출이 하나도 없으면 완전 일치(1.0), 하나라도 있으면 0.0으로 본다.
"""
import json
from pathlib import Path

_RECIPES = {r["id"]: r for r in json.loads((Path.home() / "kitchen-mcp" / "data" / "recipes.json").read_text(encoding="utf-8"))}


def _recipe_by_name_or_id(key):
    """kitchen_mcp.tools.store.recipe_by_name_or_id와 같은 규칙: id 또는 이름. 없으면 None. (kitchen-agent는 kitchen_mcp를 import하지 않는다)"""
    if key in _RECIPES:
        return _RECIPES[key]
    for r in _RECIPES.values():
        if r["name"] == key:
            return r
    return None

IGNORED_ARGS = {"query", "top_k"}
RECIPE_ARGS = {"recipe", "recipes"}  # 레시피를 가리키는 인자 (이름 또는 id로 쓸 수 있다)


def _recipe_name(v):
    """'R037' 같은 id를 레시피 이름으로 바꾼다. 도구가 이름과 id를 둘 다 받으므로 같은 레시피로 본다."""
    r = _recipe_by_name_or_id(v) if isinstance(v, str) else None
    return r["name"] if r else v


def _norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return sorted(_norm(x) for x in v) if all(isinstance(x, (str, int, float)) for x in v) else v
    return v


def _key_args(args):
    out = {}
    for k, v in (args or {}).items():
        if k in IGNORED_ARGS:
            continue
        if k == "recipe":
            v = _recipe_name(v)
        elif k == "recipes" and isinstance(v, list):
            v = [_recipe_name(x) for x in v]
        out[k] = _norm(v)
    return out


def _f1(p, r):
    return 0.0 if p + r == 0 else round(2 * p * r / (p + r), 4)


def prf(expected, actual):
    """집합 비교의 (정밀도, 재현율, F1). 둘 다 비어 있으면 1.0."""
    expected, actual = set(expected), set(actual)
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    tp = len(expected & actual)
    p = tp / len(actual) if actual else 0.0
    r = tp / len(expected) if expected else 0.0
    return round(p, 4), round(r, 4), _f1(p, r)


def tool_call_scores(expected_calls, actual_calls):
    exp_names = [c["name"] for c in expected_calls]
    act_names = [c["name"] for c in actual_calls]
    p, r, f1 = prf(exp_names, act_names)
    # 인자 일치율: 기대에도 있는 이름의 실제 호출만 대상으로 한다
    graded = [c for c in actual_calls if c["name"] in set(exp_names)]
    matched = 0
    for c in graded:
        a = _key_args(c["args"])
        if any(e["name"] == c["name"] and _key_args(e["args"]) == a for e in expected_calls):
            matched += 1
    arg_rate = (matched / len(graded)) if graded else (1.0 if not expected_calls and not actual_calls else 0.0)
    return {"name_precision": p, "name_recall": r, "name_f1": f1,
            "arg_match": round(arg_rate, 4), "actual_calls": len(actual_calls), "expected_calls": len(expected_calls)}


def agent_scores(expected_agents, actual_calls):
    actual = {c["agent"] for c in actual_calls}
    p, r, f1 = prf(expected_agents, actual)
    return {"agent_precision": p, "agent_recall": r, "agent_f1": f1}
