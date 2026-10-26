"""멀티 에이전트: supervisor가 필요한 전문 에이전트를 고르고, Send로 병렬 실행한 뒤 aggregate가 하나의 답으로 취합한다.

그래프: START → supervisor ─(Send × N)→ specialist(서브그래프) … → aggregate → END
                                  └ 범위 밖이면 곧바로 aggregate (도구 호출 없음)

전문 에이전트 6명은 각자 자기 도구 묶음만 가지며 서로의 결과를 보지 않는다.
"""
import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from agents.common import make_model, text_of
from agents.trace import make_record, tool_calls_of

AGENT_ROLES = {
    "recipe": "레시피 검색·조회·필터(태그, 시간, 보유 장비)",
    "pantry": "냉장고 재고 조회, 재고로 레시피 만들 수 있는지 대조, 유통기한 임박·지난 재고",
    "nutrition": "영양 합계·1인분 영양, 알레르기 대조, 저염·저칼로리·고단백 판정",
    "substitute": "재료가 없을 때 대체 재료, 재료 보관법",
    "shopping": "재료 가격, 장보기 목록과 금액, 예산 판정",
    "cooking": "조리 총 시간, 단계를 겹쳐 할 때의 시간, 필요한 장비가 우리 집에 있는지",
}

SUPERVISOR_PROMPT = f"""당신은 주방 도우미 팀의 supervisor입니다. 사용자의 질문을 풀기 위해 필요한 전문 에이전트를 고릅니다.

전문 에이전트:
""" + "\n".join(f"- {k}: {v}" for k, v in AGENT_ROLES.items()) + """

규칙:
- 질문에 답하는 데 꼭 필요한 에이전트만 고릅니다. 필요 없는 에이전트는 고르지 않습니다.
- 주방·요리와 무관한 질문(날씨, 주식, 코딩, 의료, 여행 등)은 빈 목록으로 답합니다.
- 질문 앞에 [이전 대화]가 붙어 있으면 [현재 질문]에 필요한 에이전트를 고릅니다. 이전 대화는 '그거', '첫 번째' 같은 지시어를 이해하는 데만 씁니다.
- 반드시 아래 JSON 한 줄로만 답합니다. 다른 설명은 쓰지 않습니다.
{"agents": ["recipe", "pantry"]}
"""

SPECIALIST_PROMPT = """당신은 주방 도우미 팀의 '{role}' 담당 전문 에이전트입니다. 담당: {desc}.
사용자의 질문 중 당신 담당 부분만 맡습니다. 당신의 도구를 호출해 확인한 결과만 근거로 답합니다.
- 기억으로 추측하지 않고, 계산(영양, 가격, 시간)은 도구 결과를 그대로 옮깁니다.
- 알레르기는 도구의 대조 결과만 전하며 건강·의료 조언은 하지 않습니다.
- 담당 밖의 내용은 한 줄도 쓰지 않습니다. 다른 담당이 확인할 내용(예: 알레르기 대조는 nutrition, 재고는 pantry)을 언급하거나 추측하지 않습니다.
- 도구 결과에 없는 주의사항이나 조언은 덧붙이지 않습니다.
- 질문 앞에 [이전 대화]가 붙어 있으면 [현재 질문]만 답합니다. 이전 대화는 지시어를 이해하는 데만 쓰고, 이전 답의 수치를 그대로 믿지 말고 도구로 다시 확인합니다.
- 결과를 사실 위주로 짧고 정확하게 한국어로 정리합니다 (수치 포함).
"""

AGGREGATE_PROMPT = """당신은 주방 도우미 팀의 취합 담당입니다. 전문 에이전트들의 부분 답을 근거로 사용자의 질문에 하나의 답으로 정리합니다.
- 부분 답에 없는 사실을 만들지 않습니다. 수치는 부분 답 그대로 옮깁니다.
- 사용자가 건 조건(알레르기, 재료, 시간, 예산 등)별로 충족 여부를 분명히 밝힙니다.
- 건강·의료 판단이나 조언은 하지 않습니다.
- 질문 앞에 [이전 대화]가 붙어 있으면 [현재 질문]에 답합니다. 이전 대화의 수치는 근거로 쓰지 않고 부분 답의 수치만 씁니다.
- 부분 답이 하나도 없으면(범위 밖 질문) 이 서비스는 주방·요리 관련 질문만 도울 수 있다고 짧게 답합니다.
"""

MAX_STEPS = 8  # 전문 에이전트 하나의 도구 루프 상한(모델 호출 횟수)


class State(TypedDict):
    question: str
    selected: list
    partials: Annotated[list, operator.add]  # 병렬 에이전트의 부분 답이 여기에 쌓인다
    trace: Annotated[list, operator.add]  # 노드마다 기록 한 건(호출 수·토큰·시간·도구 호출)이 쌓인다
    final: str


class SpecialistState(TypedDict):
    question: str
    agent: str


def with_history(question, history):
    """앞 (질문, 최종 답) 기록을 질문 앞에 붙인 문자열을 돌려준다. 기록이 없으면 질문 그대로."""
    if not history:
        return question
    lines = []
    for q, a in history:
        lines.append(f"사용자: {q}")
        lines.append(f"도우미: {a}")
    return ("[이전 대화 — 지시어('그거', '첫 번째', '아까 그 요리' 등)를 이해하는 데만 쓰고, 사실은 도구로 다시 확인하세요]\n"
            + "\n".join(lines) + f"\n\n[현재 질문]\n{question}")


def parse_agents(text):
    """supervisor 응답에서 에이전트 목록을 뽑는다. 알 수 없는 이름은 버린다."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return []
    try:
        names = json.loads(m.group(0)).get("agents", [])
    except json.JSONDecodeError:
        return []
    return [n for n in names if n in AGENT_ROLES]


def build_specialist(agent_name, tools):
    """전문 에이전트 한 명의 서브그래프: 자기 도구 묶음(tools)만 쓰는 도구 루프."""
    model = make_model().bind_tools(tools)
    prompt = SPECIALIST_PROMPT.format(role=agent_name, desc=AGENT_ROLES[agent_name])

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        steps: int

    async def llm(state: S):
        response = await model.ainvoke([SystemMessage(prompt)] + state["messages"])
        return {"messages": [response], "steps": state["steps"] + 1}

    def route(state: S):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) and state["steps"] < MAX_STEPS else END

    g = StateGraph(S)
    g.add_node("llm", llm)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route, ["tools", END])
    g.add_edge("tools", "llm")
    return g.compile()


def build_graph(tool_groups):
    """tool_groups: {에이전트 이름: 그 에이전트의 LangChain 도구 목록}"""
    sup_model = make_model(max_tokens=2048)
    agg_model = make_model()
    specialists = {}
    for name, tools in tool_groups.items():
        specialists[name] = build_specialist(name, tools)

    async def supervisor(state: State):
        started = time.perf_counter()
        response = await sup_model.ainvoke([SystemMessage(SUPERVISOR_PROMPT), HumanMessage(state["question"])])
        seconds = time.perf_counter() - started
        record = make_record("supervisor", None, [response], seconds)
        return {"selected": parse_agents(text_of(response)), "trace": [record]}

    def fan_out(state: State):
        """고른 에이전트마다 Send 하나씩 → 병렬 실행. 없으면 곧바로 취합으로."""
        if not state["selected"]:
            return "aggregate"
        sends = []
        for agent_name in state["selected"]:
            specialist_input = {"question": state["question"], "agent": agent_name}
            sends.append(Send("specialist", specialist_input))
        return sends

    async def specialist(state: SpecialistState):
        name = state["agent"]
        started = time.perf_counter()
        result = await specialists[name].ainvoke({"messages": [HumanMessage(state["question"])], "steps": 0})
        seconds = time.perf_counter() - started
        messages = result["messages"]
        record = make_record("specialist", name, messages, seconds, tool_calls=tool_calls_of(messages))
        return {"partials": [{"agent": name, "answer": text_of(messages[-1])}], "trace": [record]}

    async def aggregate(state: State):
        if state["partials"]:
            body = "\n\n".join(f"[{p['agent']}]\n{p['answer']}" for p in sorted(state["partials"], key=lambda p: p["agent"]))
        else:
            body = "(전문 에이전트 부분 답 없음 — 범위 밖 질문)"
        started = time.perf_counter()
        response = await agg_model.ainvoke([SystemMessage(AGGREGATE_PROMPT),
                                            HumanMessage(f"질문: {state['question']}\n\n부분 답:\n{body}")])
        seconds = time.perf_counter() - started
        record = make_record("aggregate", None, [response], seconds)
        return {"final": text_of(response), "trace": [record]}

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("specialist", specialist)
    g.add_node("aggregate", aggregate)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", fan_out, ["specialist", "aggregate"])
    g.add_edge("specialist", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


async def answer(question, tool_groups, history=None):
    """질문 하나를 처리하고 실행 기록을 돌려준다. history는 앞선 [(질문, 최종 답), ...].

    돌려주는 딕셔너리: final(최종 답), selected(선택된 에이전트), partials(부분 답),
    trace(노드별 기록), seconds(전체 소요 시간).
    """
    graph = build_graph(tool_groups)
    started = time.perf_counter()
    result = await graph.ainvoke({"question": with_history(question, history), "selected": [],
                                  "partials": [], "trace": [], "final": ""})
    seconds = time.perf_counter() - started
    return {
        "final": result["final"],
        "selected": result["selected"],
        "partials": result["partials"],
        "trace": result["trace"],
        "seconds": round(seconds, 3),
    }
