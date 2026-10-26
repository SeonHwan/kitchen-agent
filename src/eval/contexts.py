"""검색 도구가 돌려준 결과를 RAGAS retrieved_contexts(문자열 목록)로 바꾼다. 모델 호출 없음.

검색 지표는 '검색기가 실제로 본 문서 텍스트'를 입력으로 쓴다(decisions 15절).
- search_recipes: 결과가 {id, name, score}뿐이라 id로 레시피를 찾아, 임베딩에 쓰인 문서 문자열을 다시 만든다.
  규칙은 kitchen-mcp의 embedding.py recipe_index()와 같아야 한다: "{이름}. 태그: {태그들 공백으로}. 재료: {재료들 쉼표로}."
  (kitchen-mcp가 바뀌면 조용히 어긋난다. 그래서 check_contexts()가 임베딩 캐시 해시로 대조한다.)
- find_substitute, storage_tip: 결과에 규칙·노트 내용이 들어 있어, 임베딩 문서의 text 필드를 id로 찾아 쓴다.
"""
import json
from pathlib import Path

KITCHEN_MCP_DATA = Path.home() / "kitchen-mcp" / "data"
SEARCH_TOOLS = ("search_recipes", "find_substitute", "storage_tip")


def _load(name):
    return json.loads((KITCHEN_MCP_DATA / name).read_text(encoding="utf-8"))


def recipe_document(recipe):
    """레시피 하나의 임베딩 문서 문자열. kitchen-mcp recipe_index()의 규칙을 그대로 따른다."""
    ingredients = ", ".join(i["name"] for i in recipe["ingredients"])
    return f"{recipe['name']}. 태그: {' '.join(recipe['tags'])}. 재료: {ingredients}."


def load_documents():
    """문서 id → 임베딩 문서 문자열. 레시피(R...), 대체 규칙, 보관 노트를 한 사전에 담는다."""
    documents = {}
    for recipe in _load("recipes.json"):
        documents[recipe["id"]] = recipe_document(recipe)
    for rule in _load("substitutes.json"):
        documents[rule["id"]] = rule["text"]
    for note in _load("storage_notes.json"):
        documents[note["id"]] = note["text"]
    return documents


def parse_tool_result(text):
    """도구 결과 문자열에서 JSON 값들을 차례로 읽어 목록으로 돌려준다.

    MCP는 목록 결과를 항목마다 별도 블록으로 주고, mcp_tools.result_to_text가 줄바꿈으로 이어 붙인다.
    그래서 '{...}\n{...}' 처럼 JSON이 여러 개 붙어 있다. 결과가 [...] 한 덩어리이면 그 항목들을 쓴다.
    """
    decoder = json.JSONDecoder()
    values = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position = position + 1
            continue
        value, position = decoder.raw_decode(text, position)
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return values


def retrieved_contexts(run, documents):
    """실행 기록 한 건에서 검색 도구 결과를 모아 (문서 id 목록, 문서 문자열 목록)을 돌려준다. 호출 순서대로, 중복은 한 번만."""
    ids = []
    for record in run["trace"]:
        for call in record["tool_calls"]:
            if call["name"] not in SEARCH_TOOLS:
                continue
            hits = parse_tool_result(call["result"])
            for hit in hits:
                if hit["id"] not in ids:
                    ids.append(hit["id"])
    return ids, [documents[doc_id] for doc_id in ids]
