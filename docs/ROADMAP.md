# kitchen-agent 로드맵

가정 주방의 "오늘 뭐 해 먹지"를 여러 전문 에이전트가 나눠 푸는 **멀티 에이전트(LangGraph) 학습 프로젝트**.
레시피·재료·냉장고 재고·장보기·조리 계획처럼 서로 다른 데이터를 다루는 에이전트 6명이 한 질문을 함께 풀고,
이 구조의 호출 수·토큰·시간·도구 호출 정확도를 숫자로 재고, RAGAS로 검색·에이전트·생성 품질을 평가하고, 도구를 MCP 서버로 분리한다.

## 1. 학습 목표

| 목표 | 확인 방법 |
|---|---|
| LangGraph로 supervisor, 병렬 분기(`Send`), 취합 흐름 구현 | 복합 질문 1개가 에이전트 여러 명을 동시에 호출하는 trace |
| 멀티 에이전트의 비용과 도구 호출 정확도를 수치로 확인 | 같은 질문 세트의 호출 수·토큰·시간·정확도 요약표 |
| Langfuse로 호출 흐름·토큰·시간 관측 | trace 화면 |
| RAGAS의 검색·에이전트·생성 지표를 각각 의미 있게 적용 | 지표별 점수와 실패 사례 읽기 |
| 도구를 MCP 서버로 분리 | 에이전트가 MCP로 도구를 호출해 같은 질문에 답함 |
| FastAPI + Docker 서빙 | `docker compose up` 후 API 호출 |

## 2. 설계 원칙

1. **모델이 기억으로 맞힐 수 없는 데이터만 쓴다.** 일상 주제이지만 데이터는 이 프로젝트가 직접 만든 가상의 값(우리 집 냉장고 재고, 이 DB의 영양·가격)이다. 도구를 부르지 않으면 답을 맞힐 수 없어야 에이전트와 평가가 의미를 가진다.
2. **데이터는 작게, 에이전트는 다양하게.** 전체 데이터는 수백 KB, 에이전트 6명, 에이전트당 도구 2~3개.
3. **정답을 아는 질문 세트를 데이터와 함께 만든다.** 데이터를 직접 구성하므로 질문마다 정답 근거 문서, 기대 도구 호출, 기준 답을 처음부터 기록한다(Phase 0). 평가를 나중에 붙이느라 질문을 다시 만들지 않는다.
4. **에이전트 하나 = 도구 묶음 하나 = (나중에) MCP 서버 하나.** 에이전트끼리 도구를 공유하지 않는다. 데이터는 읽기 전용이므로 여러 에이전트가 같은 파일을 읽을 수 있다(레시피는 recipe, pantry, cooking이, 냉장고 상태는 pantry와 shopping이 읽는다). 어떤 도구도 데이터 파일을 수정하지 않는다.
5. **계산은 도구가, 문장은 모델이.** 영양 합계, 가격 합계, 소요 시간은 도구가 계산하고 모델은 계산하지 않는다.
6. **가정값은 코드에 숨기지 않는다.** 가정값(1인분 기준량, 단가, 조리 시간 보정 등)은 파라미터로 두고 `docs/decisions.md`에 근거 또는 "근거 없음"을 적는다.
7. **건강·의료 판단은 하지 않는다.** 알레르기는 "재료의 알레르기 분류 항목과 가족 알레르기 목록이 겹치는가"만 대조한다. 영양 수치는 학습용 가상 값이다.
8. **확인한 것과 추정을 구분해 기록한다.** 측정 근거와 한계(질문 수, 가정값)를 함께 쓴다. 모델이 한 판정을 "사람이 검증했다"고 쓰지 않는다.

## 3. 데이터 명세

모두 생성 스크립트(`kitchen-mcp`의 `scripts/`)가 만든다. 같은 입력이면 같은 결과가 나오도록 난수 시드를 고정한다. 결과는 `data/`에 두고 커밋하지 않는다(생성 스크립트와 설명서만 커밋).

| 데이터 | 규모(목표) | 형태 | 내용 |
|---|---|---|---|
| 레시피 | 60개 | JSON 문서 | 이름, 재료와 분량, 단계(소요 시간 포함), 필요 장비, 태그(한식·양식·간편 등), 인분 |
| 재료 사전 | 120개 | 표 | 이름, 분류, 100g당 영양(열량·단백질·탄수화물·지방·나트륨), 알레르기 분류, 단위당 가격, 보관법 |
| 대체 재료 규칙 | 40개 | 문서 | 어떤 재료를 무엇으로, 어떤 비율로, 맛·식감 차이 |
| 보관·위생 노트 | 40개 | 문서 | 재료·상황별 보관 기간과 주의점 |
| 우리 집 상태 | 1세트 | JSON | 냉장고 재고(재료·수량·유통기한), 보유 장비(에어프라이어 있음, 오븐 없음 등), 가족 구성(인원·연령·알레르기), 주간 예산 |

- 레시피 재료는 재료 사전의 이름과 반드시 일치해야 한다(Phase 0 검증 항목).
- 현재 날짜는 코드에 박지 않고 "기준 날짜" 파라미터로 둔다(유통기한 계산이 재현되도록).
- 데이터 설명서(`docs/data.md`)에 모든 값이 학습용 가상 값임을 명시한다.

## 4. 에이전트와 도구

에이전트 6명, 에이전트당 도구 2~3개. 도구는 순수 함수로 짜서 모델 없이 시험할 수 있어야 한다.

| 에이전트 | 맡는 질문 | 도구 | 데이터 |
|---|---|---|---|
| recipe (레시피) | 무슨 요리가 좋을지 찾기 | `search_recipes`(임베딩 검색), `get_recipe`, `filter_recipes`(태그·시간·장비) | 레시피 |
| pantry (냉장고) | 지금 있는 재료로 뭘 할 수 있나, 곧 상하는 건 | `list_pantry`, `match_recipe_to_pantry`, `expiring_soon` | 우리 집 상태, 레시피 |
| nutrition (영양·식이) | 영양 합계, 알레르기 충돌 | `nutrition_of`, `check_allergens`, `diet_rule_check` | 재료 사전, 가족 구성 |
| substitute (대체·보관) | 재료가 없을 때 대체, 보관법 | `find_substitute`(임베딩 검색), `storage_tip`(임베딩 검색) | 대체 규칙, 보관 노트 |
| shopping (장보기·예산) | 부족한 재료와 비용 | `price_of`, `build_shopping_list`, `budget_check` | 재료 사전, 냉장고, 예산 |
| cooking (조리 계획) | 총 시간, 병렬로 할 수 있는 단계 | `estimate_total_time`, `plan_parallel_steps` | 레시피, 보유 장비 |

예시 복합 질문: "금요일 저녁 4인 가족, 아이 하나가 땅콩 알레르기, 냉장고 재료 위주로, 40분 안에, 예산 2만 원 안으로 추천해 줘."
→ recipe, pantry, nutrition, shopping, cooking이 각자 자기 도구로 부분 답을 만들고 취합이 하나로 정리한다.

## 5. 질문 세트 스키마

Phase 0에서 질문과 함께 아래 필드를 기록한다. RAGAS 지표가 요구하는 입력 형식(설치 버전 0.4.3의 소스에서 확인)에서 거꾸로 정했다.

| 필드 | 쓰이는 곳 |
|---|---|
| `id`, `question` | 모든 평가 |
| `type` | 단일 에이전트로 풀리는 질문 / 에이전트 2~3명 / 에이전트 4명 이상 / 범위 밖 질문 |
| `expected_agents` | supervisor 선택 확인 |
| `reference_tool_calls` | Tool Call Accuracy·F1 (도구 이름과 인자) |
| `reference_docs` | 검색 평가(정답 근거 문서 id 목록) |
| `reference` | 기준 답 문장. Context Precision·Recall, Noise Sensitivity, Factual Correctness |
| `reference_outcome` | Agent Goal Accuracy |
| `reference_topics` | Topic Adherence (허용 주제) |

- 규모: 60문항 목표. 유형별로 균형을 맞춘다(단일 15, 2~3명 20, 4명 이상 15, 범위 밖 10).
- 기준 답과 정답 근거는 데이터 생성 스크립트가 만든 값에서 계산해 기록한다(모델이 쓰지 않는다).
- `ToolCall`의 정확한 필드와 `retrieved_contexts`로 쓸 문서 단위는 Phase 5 시작 때 설치된 소스를 읽고 맞춘다.

## 6. Phase 계획

한 Phase는 "실행 직전까지 코드를 짜고, 사용자가 실행하고, 호출 순서대로 짚는다" 한 흐름으로 진행한다.

| Phase | 내용 | 끝났다는 증거 |
|---|---|---|
| 0 | **데이터와 질문 세트**: 생성 스크립트, 데이터 설명서, 질문 60문항(스키마 전체), 정합성 검증 | 레시피 재료가 모두 재료 사전에 있음, 질문별 기준 값이 데이터와 일치 |
| 1 | **도구와 검색 기반**: 도구 함수 16개, 임베딩 인덱스(레시피·대체·보관), 모델 없이 동작 확인 | 도구 단위 시험 통과, 검색이 정답 근거 문서를 상위에 가져옴 |
| 2 | ~~단일 에이전트 기준선~~ **제거됨**: 번호는 유지한다. 코드는 git 이력 `3cc8e61`에 남아 있다 | - |
| 3 | **멀티 에이전트**: 에이전트 6명(서브그래프), supervisor 선택, 병렬 분기, 취합 | 복합 질문에서 에이전트 병렬 실행 trace |
| 4 | **관측성과 실행 요약**: Langfuse 연결, 질문 세트를 멀티 에이전트로 실행, 호출 수·토큰·시간·도구 호출 정확도 요약 | 요약표 |
| 5 | **RAGAS**: 검색 지표(Context Precision·Recall, Noise Sensitivity), 에이전트 지표(Tool Call Accuracy·F1, Agent Goal Accuracy, Topic Adherence), 생성 지표(Faithfulness 등), 지표마다 실패 사례 읽기 | 지표별 점수표와 사례 분석 |
| 6 | **MCP 서버 분리**: 에이전트별 도구 묶음을 MCP 서버로, 에이전트는 MCP 클라이언트로 호출 | MCP 경유 실행이 직접 호출과 같은 도구 호출 결과 |
| 7 | **서빙**: FastAPI + Docker Compose | `docker compose up` 후 API 호출 성공 |

- 사람 승인(HITL), 체크포인트는 범위에 넣지 않는다.
- Phase 0~4는 연속으로 진행해도 된다. 코드 리뷰는 Phase 4가 끝난 뒤 사용자가 요청할 때 한다.
- Phase 5는 Phase 4 이후에 시작한다.

### Phase 4 측정 항목

| 항목 | 멀티 에이전트 |
|---|---|
| 모델 호출 수, 입력·출력 토큰, 소요 시간 | 측정 |
| 도구 호출 정확도(기대 도구 호출과 비교) | 측정 |
| 질문 유형별 결과(단일 / 2~3명 / 4명 이상 / 범위 밖) | 측정 |

### Phase 5 지표와 질문 유형의 대응

| 지표 묶음 | 적용 대상 | 질문 세트에서 쓰는 필드 |
|---|---|---|
| 검색 | `search_recipes`, `find_substitute`, `storage_tip`이 가져온 문서 | `reference_docs`, `reference` |
| 에이전트 | 도구 호출 기록과 supervisor·에이전트 선택 | `reference_tool_calls`, `reference_outcome`, `reference_topics` |
| 생성 | 최종 답 | `reference`, 도구 결과를 근거로 한 충실도 |

- 채점 모델은 답을 만든 모델과 다른 계열을 쓴다.
- 지표 구현이 두 갈래(새 방식 `ragas.metrics.collections`, 옛 방식)로 나뉘어 있으므로 시작 때 설치된 소스에서 필요한 지표가 어느 쪽에 있는지부터 확인한다.

## 7. 환경과 기술 선택

| 항목 | 선택 |
|---|---|
| 위치 | WSL 홈 `~/kitchen-agent`. Windows 폴더(`/mnt/c/...`)는 쓰지 않는다 |
| Python | 3.13, 패키지 관리 `uv`, 실행은 `uv run python src/...` |
| 에이전트 | LangGraph, LangChain |
| 모델 | `.env`의 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. 에이전트 `claude-sonnet-5-5`, 채점·검증 `gpt-5.6-terra` |
| 임베딩 | `fastembed`의 `intfloat/multilingual-e5-large`(1024차원, 약 2.2GB, `data/fastembed_cache`에 둠). 벡터 DB는 쓰지 않고 메모리의 `numpy` 행렬로 검색한다(문서 총 수백 개) |
| 관측성 | Langfuse 셀프 호스팅(Docker Compose) |
| 평가 | RAGAS |
| MCP·서빙 | Python MCP SDK, FastAPI, Docker Compose |

### 동작이 확인된 라이브러리 조합

아래 버전에서 LangGraph 도구 루프, 병렬 분기, Langfuse 연결, RAGAS 지표 실행이 동작했다. 새 환경은 같은 버전으로 시작한다.

| 패키지 | 버전 |
|---|---|
| langgraph | 1.2.15 |
| langchain-core | 1.6.10 |
| langchain-anthropic | 1.7.6 |
| mcp | 2.3.0 |
| python-dotenv | 1.2.4 |

RAGAS(0.4.3)와 서빙의 `fastapi`(0.162.0), `uvicorn`(0.54.0)은 `pyproject.toml`에 고정했다. `fastembed`는 `kitchen-mcp`에 있다. `langfuse`(클라이언트 5.0.2, 서버는 4.x)는 걷어냈고 다시 쓸 때 추가한다.

### 알려진 제약 (모두 실행으로 확인함)

1. **RAGAS를 다시 쓸 때 `langchain-community`를 0.3.31로 고정한다.** 0.4.2는 `ragas 0.4.3`이 import하는 `vertexai` 모듈이 없어 `import ragas`가 실패한다.
2. **`claude-sonnet-5-5`는 thinking이 기본으로 켜져 있다.** `max_tokens`가 작으면 텍스트가 빈 응답이 나올 수 있다. 강제 도구 호출(`tool_choice`의 `any`/`tool`)은 400 오류가 난다. 도구 호출은 `auto`로 두고 프롬프트로 유도한다.
3. **`ChatAnthropic`은 `ANTHROPIC_API_KEY`를 환경 변수에서 읽는다.** 인증 헤더나 주소를 코드에서 따로 넘기지 않는다.
4. **모델 응답 `content`는 문자열이거나 블록 목록**(`thinking`, `text`)이다. text 블록만 모아 쓰는 헬퍼를 둔다. thinking 블록은 내용이 비어 있고 `signature`만 온다.
5. **`gpt-5.6-terra`는 추론 모델이다.** `temperature`는 1만 받고 `top_p`는 받지 않는다. 출력 상한은 추론 토큰을 포함한다. Chat Completions 형식(`/v1/chat/completions`)과 Responses 형식 모두 호출된다. RAGAS의 `llm_factory`는 모델 이름이 `gpt-<정수>`로 시작할 때만 추론 모델로 인식해서 `gpt-5.6-terra`에는 `temperature=0.01`, `top_p=0.1`을 그대로 보내 400이 난다. 생성 뒤 `llm.model_args`를 직접 고쳐야 하고(`temperature=1.0`, `top_p` 제거, `max_tokens`를 `max_completion_tokens`로, 구조화 출력은 상한 16384), 이 처리 없이는 호출되지 않는다.
6. 일부 사이트는 `self-signed certificate` 오류로 막힌다. 인증서 검증을 끄는 방식으로 우회하지 않는다.
7. `draw_mermaid_png()`는 외부 서버를 불러 막힌다. LangGraph 그래프는 노드와 엣지를 Graphviz DOT으로 바꿔 `dot`으로 PNG를 만든다(`dot`이 설치되어 있어야 함). 한글 폰트가 없으니 노드 이름은 영어로 쓴다.
8. `add_conditional_edges`의 세 번째 인자는 도착 노드 목록(또는 `{라벨: 노드}` 딕셔너리)이다. 생략하면 실행은 되지만 그림에 갈림길이 안 그려진다. `Send`로 병렬 분기하는 필드는 `Annotated[list, operator.add]`로 합치고, 되돌려 다시 쓸 때 쌓인 값을 비우려면 `Overwrite`를 쓴다.
9. Langfuse 서버 4.x는 `/api/public/traces`가 404이고 `/api/public/v2/observations`로 조회한다. 클라이언트 5.x와 서버 4.x 조합이 정상 기록됐다. 클라우드(`cloud.langfuse.com`)와 일부 이미지 저장소는 접속이 막혀 셀프 호스팅한다. 막히는 이미지는 다른 경로로 받아 `docker save`/`docker load`로 옮긴다.

## 8. 폴더 구조 (목표)

```
~/kitchen-agent/                  # 에이전트와 평가
├── docs/
│   ├── ROADMAP.md            # 이 문서
│   └── decisions.md          # 설계 결정과 측정 근거 (진행하면서)
├── src/
│   ├── agents/               # 멀티 에이전트, MCP 클라이언트 (Phase 3, 6)
│   ├── ask.py                # 대화 루프
│   ├── api.py                # FastAPI 서버 (Phase 7)
│   └── eval/                 # 평가, RAGAS (Phase 4~5)
├── Dockerfile                # api 이미지: kitchen-agent + kitchen-mcp (Phase 7)
├── compose.yaml              # docker compose up
├── infra/langfuse/           # Langfuse Docker Compose
├── data/                     # 질문 세트, 로그, 결과 (커밋 안 함)
├── .env                      # 키 (커밋 안 함)
└── pyproject.toml

~/kitchen-mcp/                    # 데이터, 도구, MCP 서버 (별도 저장소)
├── docs/data.md              # 데이터 설명서 (Phase 0)
├── scripts/                  # 데이터 생성 스크립트와 질문 세트 생성기 build_questions.py
├── src/kitchen_mcp/tools/    # 순수 함수 도구 16개 (Phase 1)
├── src/kitchen_mcp/servers/  # MCP 서버 6개 (Phase 6)
└── data/                     # 생성 데이터 (커밋 안 함)
```

- 질문 세트 생성기는 `kitchen-mcp`에 있고, 기준 값을 도구 함수 호출로 계산하기 때문이다(decisions 12절). 결과(`questions.json`, `questions.csv`)는 `kitchen-agent` 루트에서 실행해 `kitchen-agent/data/`에 만든다.
  `uv run --project ~/kitchen-mcp python ~/kitchen-mcp/scripts/build_questions.py`
- 서빙: `docker compose up --build`로 API가 뜬다(포트 8000). `kitchen-mcp`가 `../kitchen-mcp`에 있어야 하고(다르면 `KITCHEN_MCP_DIR` 환경변수), 그 안의 `data/`(생성 데이터와 임베딩 모델 캐시)가 있어야 한다. 모델 캐시는 이미지에 넣지 않고 읽기 전용 볼륨으로 붙인다(decisions 20절).
  호스트에서는 `uv run uvicorn api:app --app-dir src --port 8000`. 주의: `docker compose config`는 `.env` 값을 평문으로 출력하므로 `--quiet`로만 쓴다.

## 9. 진행 상황

| Phase | 상태 |
|---|---|
| 0. 데이터와 질문 세트 | 완료 (질문 60문항, 기준 답은 사람이 검증하지 않음). 생성기는 `kitchen-mcp/scripts/`로 옮겨 다시 생성함(docs/decisions.md 12절) |
| 1. 도구와 검색 기반 | 완료 (도구 단위 시험 35건 통과, 검색 확인은 docs/decisions.md 5절) |
| 2. 단일 에이전트 기준선 | 제거됨 |
| 3. 멀티 에이전트 | 완료. 병렬 5개 실행은 Langfuse trace로 한 번 확인했다(과거 기록, 지금은 trace 코드가 없어 같은 방식으로 재확인 불가). 코드 로직은 사람이 읽고 확인했다(2026-10-23) |
| 4. 관측성과 실행 요약 | 한 번 완료했으나(요약표 data/compare.md, 해석은 docs/decisions.md 6절) 이후 Langfuse 연결 코드와 RunLog 기반 평가 코드를 모두 걷어냈다(순수 로직 확인 우선). 측정 수치는 과거 기록이고 현재 저장소에서 다시 만들 수 없다. `infra/langfuse/`는 남아 있다 |
| 5. RAGAS | 완료. 설치, 도구 호출·검색·에이전트·생성 지표 4묶음을 한 번씩 적용했다(docs/decisions.md 14~18절). 점수는 모델 판정이고 사람이 검증하지 않았다. 지표마다 입력 범위·기준 문장·기준 답 길이 등의 한계가 있어 점수를 에이전트 품질의 직접 증거로 읽을 수 없다. 반복 실행 편차는 재지 않았다 |
| 6. MCP 서버 분리 | 완료. 도구는 별도 저장소 `kitchen-mcp`의 서버 6개로 옮겼고 에이전트가 MCP로 호출한다(docs/decisions.md 11절). 60문항을 MCP 경유로 1회 실행해 기록된 도구 호출 190건을 직접 호출과 대조했고 모두 같았다. 이전 방식과 같은 지표로 비교했으나 호출 방식과 프롬프트 수정이 함께 바뀐 비교다(decisions 19절) |
| 7. 서빙 | 완료. FastAPI(`/health`, `/ask`)를 stateless로 만들고 Docker Compose로 띄웠다. 컨테이너에서 `/health`(서버 6개, 도구 16개)와 임베딩 도구를 쓰는 `/ask`(Q01)가 동작했고 호스트와 같은 검색 결과를 냈다(docs/decisions.md 20절). 동시 요청은 2건만, 컨테이너 재시작·502 경로는 시험하지 않았다. 이미지는 2.36GB(RAGAS 의존성이 큼) |
