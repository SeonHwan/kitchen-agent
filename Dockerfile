# 에이전트 API 이미지: kitchen-agent(FastAPI + LangGraph)와 kitchen-mcp(도구·데이터·MCP 서버 6개)를 한 이미지에 넣는다.
# MCP 서버 6개는 앱이 시작할 때 이 컨테이너 안에서 자식 프로세스(stdio)로 뜬다.
# kitchen-mcp는 별도 저장소라 compose의 additional_contexts(mcp)로 받는다.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# uv run이 서버를 띄울 때마다 환경을 다시 맞추려 하지 않게 한다(의존성은 아래에서 빌드 때 설치한다)
ENV UV_FROZEN=1 UV_NO_SYNC=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# kitchen-mcp: 코드와 작은 데이터(json, 임베딩 인덱스). 임베딩 모델 캐시(2GB)는 이미지에 넣지 않고 볼륨으로 붙인다
WORKDIR /opt/kitchen-mcp
COPY --from=mcp pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY --from=mcp src ./src
COPY --from=mcp data/*.json ./data/
COPY --from=mcp data/embeddings ./data/embeddings
RUN uv sync --frozen --no-dev

# kitchen-agent
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src

ENV KITCHEN_MCP_DIR=/opt/kitchen-mcp
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
