"""실행 기록(data/runs/<id>.json)과 질문 세트 항목을 RAGAS 입력으로 바꾸는 부품. 모델 호출 없음."""
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage


def tool_calls_in_order(run):
    """실행 기록의 trace에서 도구 호출을 모아 [{name, args, result}] 목록으로 돌려준다.

    에이전트들이 병렬로 실행되므로 실제 시각 순서는 알 수 없다.
    여기서는 trace에 쌓인 순서(supervisor, 끝난 순서대로의 specialist, aggregate)를 따른다.
    """
    calls = []
    for record in run["trace"]:
        for call in record["tool_calls"]:
            calls.append(call)
    return calls


def to_ragas_messages(run):
    """질문 → (도구 호출 + 결과) × N → 최종 답 순서의 RAGAS 메시지 목록.

    도구 호출 하나마다 AIMessage(tool_calls=[호출 한 건])와 ToolMessage(결과) 한 쌍을 만든다.
    """
    messages = [HumanMessage(content=run["question"])]
    for call in tool_calls_in_order(run):
        ragas_call = ToolCall(name=call["name"], args=call["args"])
        messages.append(AIMessage(content="", tool_calls=[ragas_call]))
        messages.append(ToolMessage(content=str(call["result"])))
    messages.append(AIMessage(content=run["final"]))
    return messages


def to_reference_tool_calls(question):
    """질문 세트 항목의 reference_tool_calls를 RAGAS ToolCall 목록으로."""
    calls = []
    for c in question["reference_tool_calls"]:
        calls.append(ToolCall(name=c["name"], args=c["args"]))
    return calls
