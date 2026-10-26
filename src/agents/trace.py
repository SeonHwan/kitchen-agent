"""실행 기록 부품: 모델 응답에서 토큰을 읽고, 메시지 목록에서 도구 호출 기록을 뽑는다.

기록은 그래프 State의 `trace` 필드에 쌓인다(노드마다 한 건). 함수 인자로 끌고 다니지 않는다.
"""
from langchain_core.messages import AIMessage, ToolMessage


def usage_of(message):
    """모델 응답 하나의 (입력 토큰, 출력 토큰). 값이 없으면 0."""
    usage = message.usage_metadata
    if usage is None:
        return 0, 0
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def make_record(node, agent, messages, seconds, tool_calls=None):
    """노드 한 번의 기록 한 건. messages는 이 노드가 받은 모델 응답(AIMessage) 목록."""
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        model_calls = model_calls + 1
        tokens_in, tokens_out = usage_of(message)
        input_tokens = input_tokens + tokens_in
        output_tokens = output_tokens + tokens_out
    return {
        "node": node,
        "agent": agent,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": round(seconds, 3),
        "tool_calls": tool_calls or [],
    }


def tool_calls_of(messages):
    """서브그래프 메시지 목록에서 도구 호출 기록을 뽑는다: [{name, args, result}].

    AIMessage.tool_calls(모델이 요청한 호출)와 ToolMessage(도구가 돌려준 결과)를 tool_call_id로 짝짓는다.
    """
    results_by_id = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            results_by_id[message.tool_call_id] = message.content

    records = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            records.append({
                "name": call["name"],
                "args": call["args"],
                "result": results_by_id.get(call["id"]),  # 결과가 없으면 None(상한에서 끊긴 경우)
            })
    return records
