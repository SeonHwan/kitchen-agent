# kitchen-agent

가정 주방의 "오늘 뭐 해 먹지"를 **전문 에이전트 6명**이 나눠 푸는 LangGraph 멀티 에이전트입니다.
supervisor가 필요한 에이전트를 고르고, `Send`로 병렬 실행한 뒤 하나의 답으로 취합합니다.
에이전트가 쓰는 도구 16개는 [kitchen-mcp](https://github.com/SeonHwan/kitchen-mcp)의 MCP 서버 6개(stdio)에 있고, 이 저장소는 그 도구를 받아 쓰는 **MCP 클라이언트**입니다.

데이터(레시피, 재료, 냉장고 재고, 가족 구성 등)는 모두 직접 만든 가상의 값입니다. 모델이 기억으로 맞힐 수 없으므로 도구를 부르지 않으면 답이 나오지 않습니다.

## AI Agent 단계별 프로젝트

AI Agent를 단계별로 만들어 가는 5개 프로젝트 중 **5번째**입니다. 앞 프로젝트에서 남은 한계가 다음 프로젝트를 시작한 이유입니다.

| 순서 | 프로젝트 | 다룬 것 | 남은 한계 → 다음 단계 |
|---|---|---|---|
| 1 | [game-wiki-rag](https://github.com/SeonHwan/game-wiki-rag) | 가상 게임 위키 4문서로 RAG 기본 파이프라인, 하이브리드·리랭커·Contextual Retrieval·GraphRAG, tool-use 에이전트, MCP, LangGraph 기반 Adaptive/Self-RAG | 데이터와 평가 질문이 너무 작아(문서 4개, 질문 12개) 오케스트레이션, 답변 품질 개선, 여러 RAG 기법의 효과를 제대로 확인하기 어려움 → 실제 방대한 데이터로 다시 |
| 2 | [dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent) | DART 사업보고서 5개사 15건(수만 개 청크): 표 행 단위 청킹, bge-m3 dense+sparse 하이브리드, 리랭커 측정, XBRL 값 조회 스토어 | 문서가 전문용어 중심이라 답변이 나아졌는지 확인하려면 용어부터 이해해야 해서 학습 진도가 더딤 → 일반 상식에 가까운 데이터로 |
| 3 | [encykorea-agent](https://github.com/SeonHwan/encykorea-agent) | 한국민족문화대백과사전 15,836건(청크 75,122개): hybrid 검색, 답변 생성 방식 비교, 직접 구현한 도구 호출 에이전트 | RAG 성격이 강해 에이전트 역량을 보여 주는 증거로 약하다고 판단 → 멀티 에이전트·오케스트레이션·LangGraph를 배우는 별도 프로젝트 |
| 4 | [logistics-agent](https://github.com/SeonHwan/logistics-agent) | 공개 물류 데이터(KTDB 화물 OD)와 정답이 계산되는 합성 시나리오: LangGraph 멀티 에이전트(supervisor, 병렬 분석가, 취합), 검증 노드, 사람 승인, Langfuse 관측, RAGAS 채점 | 병렬로 도는 에이전트를 더 늘려 멀티 에이전트의 효과를 눈으로 확인하고, MCP 분리와 서빙도 함께 다루려고 → 새 프로젝트로 |
| 5 | **[kitchen-agent](https://github.com/SeonHwan/kitchen-agent)** | 가상 주방 데이터와 6명의 전문 에이전트(LangGraph), 도구의 MCP 분리, RAGAS 평가, FastAPI 서빙 | — |

### 이전 단계([logistics-agent](https://github.com/SeonHwan/logistics-agent))에서 남은 한계
Phase 6까지 다뤘고, 도구를 MCP 서버로 분리하는 것과 서빙(FastAPI + Docker)은 이 프로젝트로 넘겼습니다. 병렬로 도는 에이전트를 더 늘려 멀티 에이전트의 효과를 눈으로 확인하려는 목적도 있었습니다.

### 이번 단계에서 바꾼 것
도구는 에이전트별 MCP 서버로 분리되어 있습니다(에이전트 하나 = 도구 묶음 하나 = MCP 서버 하나, 에이전트끼리 도구를 공유하지 않음). 이 저장소는 서버가 알려 준 도구를 LangChain 도구로 감싸 에이전트에게 주는 클라이언트이고, 그래프는 `ainvoke` 기반 비동기 호출로 동작합니다.

## 구조

```
질문 → supervisor ─┬─ (범위 밖: 빈 목록) ────────────────────────┐
                   └─ 에이전트 선택 ─(Send, 병렬)→ specialist ×N ─┴→ aggregate → 답
```

| 에이전트 | 맡는 일 | MCP 서버 |
|---|---|---|
| recipe | 레시피 검색·조회·필터(태그, 시간, 보유 장비) | `kitchen_mcp.servers.recipe` |
| pantry | 냉장고 재고, 재고로 만들 수 있는 레시피, 유통기한 임박 | `kitchen_mcp.servers.pantry` |
| nutrition | 영양 합계, 알레르기 대조, 식이 규칙 판정 | `kitchen_mcp.servers.nutrition` |
| substitute | 대체 재료, 보관법 | `kitchen_mcp.servers.substitute` |
| shopping | 재료 가격, 장보기 목록, 예산 판정 | `kitchen_mcp.servers.shopping` |
| cooking | 조리 총 시간, 병렬로 할 수 있는 단계, 필요한 장비 | `kitchen_mcp.servers.cooking` |

- **supervisor**는 JSON(`{"agents": [...]}`)으로 에이전트를 고릅니다. 파싱에 실패하거나 알 수 없는 이름이면 빈 목록(범위 밖)으로 취급합니다.
- **specialist**는 자기 도구 묶음만 쓰는 도구 루프 서브그래프이고 도구 루프는 최대 8회로 제한합니다. 서로의 결과를 보지 못합니다.
- **aggregate**는 부분 답에 없는 사실을 만들지 않고 수치를 그대로 옮기도록 지시합니다. 부분 답이 없으면 주방 관련 질문만 도울 수 있다고 답합니다.
- **대화 이어가기**: 앞의 질문과 최종 답을 질문 앞에 문자열로 붙입니다. 이전 도구 결과는 넣지 않습니다. 기록은 메모리에만 있습니다.
- **설계 원칙**: 계산(영양, 가격, 시간)은 도구가 하고 모델은 문장을 씁니다. 알레르기는 목록 대조만 하고 건강·의료 판단은 하지 않습니다.

## 평가

질문 60문항을 MCP 경유로 1회 실행했고 오류는 없었습니다. 그 실행의 도구 호출 190건(도구 16개 전부)은 같은 인자의 직접 호출 결과와 모두 같았습니다(모델이 부른 인자 조합에 한함). RAGAS는 도구 호출(ToolCallAccuracy, ToolCallF1), 검색(ContextPrecision, ContextRecall, NoiseSensitivity), 에이전트(AgentGoalAccuracy, TopicAdherence), 생성(Faithfulness, FactualCorrectness) 지표를 gpt-5.6-terra 채점 모델로 60문항에 1회 적용했습니다. 검색 지표는 검색 도구를 부른 질문에만, 범위 밖 질문은 입력이 비는 지표에서 제외했습니다.

## 한계

- **단일 에이전트 기준선이 없습니다.** 멀티 에이전트가 에이전트 하나 + 도구보다 낫다는 주장은 하지 않습니다.
- **에이전트들이 서로의 결과를 보지 못합니다.** 의존 관계가 있는 질문은 다루지 못합니다.
- **프롬프트로 담당 밖 발언을 줄였지만 남아 있습니다.** 부분 답에 "제 담당이 아니라서 답하지 않습니다"라는 문장이 남고, 취합 답에 부분 답에 없던 문장이 하나 들어간 사례가 있었으나(질문 1개의 재시험 3회) 취합 에이전트가 역할을 잘 수행했다고 판단했습니다.
- **모델이 인분 환산을 직접 계산한 사례가 있습니다**("3명이면 약 1.5배"). 계산은 도구가 한다는 원칙에 어긋나지만 영양·가격이 아닌 단순 환산이라 영향은 작다고 봤습니다.
- 데이터의 모든 수치(영양, 가격, 분량, 시간, 보관, 대체 비율)는 임의로 만든 가상 값이고, 겹쳐 할 수 있는 단계 표시는 모델에게 시켜 정했습니다. 제가 따로 검증하지는 않았습니다.

## 실행

두 저장소를 나란히 둡니다: 이 저장소(에이전트)와 [kitchen-mcp](https://github.com/SeonHwan/kitchen-mcp)(도구, 데이터, MCP 서버 6개). 호스트에서 실행하면 `~/kitchen-mcp`를 찾고, 다른 위치에 두었다면 `KITCHEN_MCP_DIR` 환경변수로 알려 줍니다.

### 준비
- **데이터 생성**: 데이터(`kitchen-mcp/data/`)는 저장소에 없습니다. `kitchen-mcp`에서 `docs/data.md`의 순서대로 `scripts/ingredients.py` → `recipes.py` → `household.py` → `substitutes.py` → `storage_notes.py`를 실행합니다. 질문 세트(60문항)는 `kitchen-agent` 루트에서 `uv run --project ~/kitchen-mcp python ~/kitchen-mcp/scripts/build_questions.py`로 만듭니다.
- **임베딩 모델**: 검색 도구 3개가 `intfloat/multilingual-e5-large`(약 2.1GB)를 `kitchen-mcp/data/fastembed_cache`에서 읽습니다. 처음에는 내려받아야 하고, 컨테이너는 오프라인(`HF_HUB_OFFLINE=1`)으로 읽기 전용 볼륨에서 읽습니다.
- **`.env`**: 필요한 값은 `ANTHROPIC_API_KEY`(에이전트, `claude-sonnet-5-5`)와 `OPENAI_API_KEY`(RAGAS 채점, `gpt-5.6-terra`) 두 개이고, `.env.example`은 아직 없습니다. 시연만 하려면 `ANTHROPIC_API_KEY`만 있으면 됩니다.
- 모델 호출은 비용이 듭니다. 60문항 실행은 에이전트 모델 호출 375회였고(질문당 평균 약 6회, 범위 밖 질문은 2회, 에이전트가 많이 선택되면 12회 안팎), RAGAS 채점은 지표 묶음마다 224~431회가 더 듭니다(`docs/decisions.md` 13·16~18절).

### 터미널에서 질문
```bash
uv sync

uv run python src/ask.py "2~3명이 먹을 저녁 메뉴 추천해줘"   # 질문 한 개
uv run python src/ask.py                                       # 대화형: 이어서 질문 ('종료', '/new' 지원)
```

### API 서버
```bash
docker compose up --build        # 포트 8000. kitchen-mcp는 ../kitchen-mcp 에 있어야 하고 그 안에 data/가 있어야 합니다
# 호스트에서: uv run uvicorn api:app --app-dir src --port 8000
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 모델을 부르지 않고 앱 시작 때 받아 둔 도구 수를 돌려줍니다. 서버에 다시 접속해 보지 않으므로 서버 프로세스가 죽어도 200일 수 있습니다 |
| `POST /ask` | `{"question": "...", "history": [{"question": "...", "answer": "..."}]}` → `{final, selected, seconds}`. `?include_trace=true`이면 노드별 호출 수·토큰·도구 호출을 함께 돌려줍니다 |

서버는 대화 기록을 들고 있지 않아서(stateless) 이어지는 질문은 호출자가 이전 질문과 답을 `history`로 보냅니다. **인증과 요청 제한이 없습니다.** 모델 호출 키가 서버에 있으므로 외부에 공개하지 않고 로컬에서만 씁니다. 이미지는 2.36GB이고 크기의 대부분은 RAGAS가 끌어온 의존성입니다.

**주의**: `docker compose config`는 `.env` 값을 평문으로 풀어 출력합니다. 설정을 확인할 때는 `docker compose config --quiet`를 씁니다.

자세한 결정과 측정 근거는 `docs/decisions.md`, 단계별 계획은 `docs/ROADMAP.md`에 있습니다.
