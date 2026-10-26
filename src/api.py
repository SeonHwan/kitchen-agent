"""FastAPI 서버: 멀티 에이전트를 HTTP로 노출한다.

MCP 서버 6개는 앱이 시작될 때 한 번 띄우고 종료될 때 닫는다(lifespan). 요청마다 띄우지 않는다.
서버는 대화 기록을 들고 있지 않는다(stateless). 이어지는 질문은 호출자가 이전 (질문, 답)을 history로 보낸다.

실행: uv run uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents import multi  # noqa: E402
from agents.mcp_tools import KitchenTools  # noqa: E402


class Turn(BaseModel):
    """이전 대화 한 번: 사용자 질문과 도우미의 최종 답."""
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Turn] = []


class AskResponse(BaseModel):
    final: str
    selected: list[str]
    seconds: float
    trace: list | None = None  # include_trace=true 일 때만 채운다


@asynccontextmanager
async def lifespan(app):
    # MCP 서버 6개를 자식 프로세스로 띄우고 앱이 살아 있는 동안 유지한다
    async with KitchenTools() as kitchen:
        app.state.tool_groups = kitchen.groups
        yield


app = FastAPI(title="kitchen-agent", lifespan=lifespan)


@app.get("/health")
async def health():
    """앱 시작 때 받아 둔 도구 묶음의 도구 수를 돌려준다. 모델은 부르지 않는다.

    MCP 서버에 다시 접속해 보지 않으므로, 시작 뒤 서버 프로세스가 죽어도 이 응답은 그대로 ok다.
    """
    groups = app.state.tool_groups
    return {"status": "ok", "agents": {name: len(tools) for name, tools in groups.items()}}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, include_trace: bool = False):
    history = []
    for turn in request.history:
        history.append((turn.question, turn.answer))

    try:
        run = await multi.answer(request.question, app.state.tool_groups, history=history or None)
    except Exception as error:  # noqa: BLE001  모델·도구 오류를 502로 알린다. 원문은 서버 로그에 남는다
        raise HTTPException(status_code=502, detail=f"에이전트 실행 실패: {type(error).__name__}") from error

    response = {"final": run["final"], "selected": run["selected"], "seconds": run["seconds"]}
    if include_trace:
        response["trace"] = run["trace"]
    return response
