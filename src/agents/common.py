"""에이전트 공통 부품: 모델 생성, 응답 텍스트 추출."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

AGENT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8192


def make_model(max_tokens=MAX_TOKENS):
    return ChatAnthropic(model=AGENT_MODEL, max_tokens=max_tokens)


def text_of(message):
    """모델 응답에서 text 블록만 모아 문자열로 돌려준다 (thinking 블록은 버린다)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
