"""RAGAS 채점 모델 만들기: gpt-5.6-terra (답을 만든 claude-sonnet-5-5와 다른 계열).

RAGAS llm_factory는 모델 이름이 'gpt-<정수>'로 시작할 때만 추론 모델로 보는데 'gpt-5.6-terra'는 정수가 아니라
변환(temperature=1.0, top_p 제거, max_tokens → max_completion_tokens)이 적용되지 않는다(ROADMAP 제약 5).
그래서 만든 뒤 model_args를 직접 고친다.
"""
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

JUDGE_MODEL = "gpt-5.6-terra"
MAX_COMPLETION_TOKENS = 16384  # 구조화 출력 상한. 추론 토큰을 포함한다(ROADMAP 제약 5)


def make_judge():
    llm = llm_factory(JUDGE_MODEL, client=AsyncOpenAI())
    llm.model_args["temperature"] = 1.0
    llm.model_args.pop("top_p", None)
    llm.model_args.pop("max_tokens", None)
    llm.model_args["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    return llm


def record_calls(llm, dry_run=False, fake_response=None):
    """llm.agenerate를 감싸 채점 호출의 프롬프트와 응답 원문을 기록하고 기록 목록을 돌려준다.

    지표는 InstructorLLM 타입만 받으므로 객체는 그대로 두고 agenerate만 바꾼다.
    dry_run이면 모델을 부르지 않고 fake_response(응답 모델)가 만든 가짜 응답을 돌려준다(호출 수·프롬프트 크기 확인용).
    """
    calls = []
    original = llm.agenerate

    async def recording_agenerate(prompt, response_model):
        if dry_run:
            calls.append({"prompt": prompt, "response": None})
            return fake_response(response_model)
        result = await original(prompt, response_model)
        calls.append({"prompt": prompt, "response": result.model_dump()})
        return result

    llm.agenerate = recording_agenerate
    return calls
