"""kitchen-mcp 서버 6개를 stdio로 띄워 두고, 서버가 알려 준 도구를 LangChain 도구로 바꿔 에이전트별로 돌려준다."""
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import StructuredTool
from mcp import Client
from mcp.client.stdio import StdioServerParameters

AGENT_NAMES = ["recipe", "pantry", "nutrition", "substitute", "shopping", "cooking"]
# 컨테이너에서는 KITCHEN_MCP_DIR 환경변수로 kitchen-mcp 위치를 준다. 없으면 홈의 kitchen-mcp를 쓴다
KITCHEN_MCP_DIR = Path(os.environ.get("KITCHEN_MCP_DIR") or Path.home() / "kitchen-mcp")


def result_to_text(result):
    """MCP 호출 결과의 text 블록을 이어 붙인다. 구조화 결과가 있으면 그것을 JSON 문자열로 쓴다."""
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    texts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    return "\n".join(texts)


def make_langchain_tool(client, mcp_tool):
    """MCP 도구 한 개를 LangChain StructuredTool로 감싼다. 호출은 해당 서버 클라이언트로 보낸다."""

    async def call(**arguments):
        result = await client.call_tool(mcp_tool.name, arguments)
        text = result_to_text(result)
        if result.is_error:
            return f"도구 오류: {text}"
        return text

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.input_schema,
        coroutine=call,
    )


class KitchenTools:
    """서버 6개의 연결을 대화가 끝날 때까지 유지한다. `async with`로 쓴다."""

    def __init__(self):
        self.stack = AsyncExitStack()
        self.groups = {}

    async def __aenter__(self):
        for agent_name in AGENT_NAMES:
            params = StdioServerParameters(
                command="uv",
                args=["run", "python", "-m", f"kitchen_mcp.servers.{agent_name}"],
                cwd=str(KITCHEN_MCP_DIR),
            )
            client = await self.stack.enter_async_context(Client(params))
            listed = await client.list_tools()
            tools = []
            for mcp_tool in listed.tools:
                tools.append(make_langchain_tool(client, mcp_tool))
            self.groups[agent_name] = tools
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stack.aclose()
