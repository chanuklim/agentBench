# scieval — 과학기술 에이전트 LLM 평가 플랫폼 설계 (v1)

- 작성일: 2026-07-12
- 상태: 사용자 승인 대기
- 근거 문서: [`science_agent_llm_evaluation_strategy.md`](../../../science_agent_llm_evaluation_strategy.md) (평가 전략), [`docs/research/2026-07-12-benchmark-recon-synthesis.md`](../../research/2026-07-12-benchmark-recon-synthesis.md) (벤치마크·harness 실사), [`docs/research/2026-07-12-benchmark-recon-verification.md`](../../research/2026-07-12-benchmark-recon-verification.md) (핵심 주장 교차검증)

---

## 1. 목표

과학기술 특화 200B+ agentic 언어모델(참조: `upstage/Solar-Open-100B` — MoE 102B/12B active, 128k context, vLLM 0.12 서빙, `solar_open` tool/reasoning 파서)과 상용 frontier 모델(GPT/Claude/Gemini)을 **동일 harness·동일 예산·동일 scaffold**로 비교 측정하는 재현 가능한 평가 플랫폼을 만든다.

성공 기준:

1. 자사 모델 checkpoint와 상용 모델을 같은 명령으로 실행해 6개 역량 축 점수 + 최종 지수 + 비용을 산출하는 리더보드 리포트가 나온다.
2. 모든 run이 manifest로 완전히 고정(pin)되어 재현 가능하다.
3. 로컬(macOS 개발 머신)에서 개발·검증한 코드가 수정 없이 서버(Docker 가능 Linux GPU 서버)로 이전된다.

## 2. 비범위 (v2 이후로 명시적 연기)

- ScienceOps-τ (상태형 비공개 과학 프로젝트 환경)
- 최신 논문 기반 rolling holdout, Sci-LCR 자체 제작
- Product Agent 레이어 평가, Internal Release Score (공개 60% + 비공개 40% 합산)
- 실제 연구자 longitudinal study
- CC-BY-NC 라이선스 벤치마크 (MLGym-Bench, ResearchBench) — 법무 검토 전 제외

## 3. 확정된 아키텍처 결정

**Inspect AI 단일 코어** (2026-07-12 사용자 승인). 근거는 교차검증된 4개 사실:

1. 타겟 벤치마크 17개 중 13개가 `inspect_evals`에 1st-party 포팅 완료 (검증됨)
2. MIT 라이선스, UK AISI 프로덕션 사용, 유일하게 달러 단위 `cost_limit` 강제 내장 (v0.3.180+, 커스텀 provider 단가 등록 지원 — 검증됨)
3. AstaBench는 Inspect 위에 얹힌 층이므로 Inspect 채택 시 추가 비용 없이 병용 가능 (검증됨)
4. HELM은 2026-06-01 유지보수 모드 진입, lm-eval-harness는 agentic 미지원 → 대안 부재 (검증됨)

알려진 리스크: `inspect_harbor` 브리지는 서드파티(Meridian Labs) 0.x 패키지다. Terminal-Bench 실행이 여기에 의존하므로, parity 검증(§12)을 통과하지 못하면 Harbor CLI를 사이드 트랙으로 강등한다.

## 4. 시스템 아키텍처

```
configs/*.yaml ──→ runner (CLI: scieval) ──→ Inspect AI eval_set
                                               ├─ providers ─ 자사 vLLM endpoint (openai-api)
                                               │              상용 API (네이티브 provider)
                                               ├─ scaffolds ─ base | reference-agent
                                               ├─ tasks ────  inspect_evals 임포트 (핀 고정)
                                               │              자체 포팅 task
                                               └─ sandbox ──  Docker
                        .eval 로그 ──→ store (run manifest + parquet/DuckDB)
                                        ├──→ scoring engine (정규화 → 합산 → gate)
                                        └──→ reporting (정적 HTML 리더보드)
   사이드 트랙 러너 (별도 venv) ──→ 점수 JSON ──→ store로 임포트 (스키마 검증)
```

### 4.1 컴포넌트 정의

| 컴포넌트 | 책임 | 의존 |
|---|---|---|
| `providers` | 모델 레지스트리(YAML), endpoint health-check, 모델별 단가표(자사 모델은 GPU-시간 환산), Inspect 커스텀 cost 등록 | Inspect AI |
| `tasks` | 벤치마크 래퍼 + `BenchmarkSpec` 메타데이터(역량 축, 가중치, required_modalities, 채점 방식, 반복 횟수, 예산 클래스, judge 핀) | inspect_evals(커밋 핀), 자체 포트 |
| `scaffolds` | `base`(도구 없음·고정 프롬프트), `reference-agent`(전 모델 동일 ReAct + bash/python 샌드박스 도구 + 통일 retry 정책) | Inspect solver API |
| `runner` | CLI(`scieval run --suite … --model … --budget … --profile …`), 계층 suite(smoke/weekly/monthly/release), eval_set resume | Inspect eval_set |
| `store` | run manifest(JSON) + 결과(parquet, DuckDB 조회), 사이드 트랙 점수 임포트 API | — |
| `scoring` | human-normalize → 축 내 가중 산술평균 → 6축 가중 기하평균 → gate 판정 | store |
| `reporting` | 리더보드 HTML, score-vs-cost Pareto, CI, 실패 분류표 | scoring |
| `external/` | 사이드 트랙 러너: astabench(문헌·E2E·SUPER·DiscoveryBench), CritPt 제출 게이트웨이, BFCL V4 공식 러너, (조건부) SWE-bench-Live | 각자 pinned venv |

각 컴포넌트는 "무엇을 하는가 / 어떻게 쓰는가 / 무엇에 의존하는가"가 위 표로 닫혀 있어야 하며, 컴포넌트 간 통신은 파일(manifest, parquet, 점수 JSON)로만 한다. 사이드 트랙이 파일 경계로 분리되는 이유: astabench는 inspect-ai 버전 핀이 코어와 충돌하고, CritPt·BFCL V4는 외부 채점/자체 러너를 쓰기 때문이다.

### 4.2 실행 환경 프로파일 (로컬 → 서버 이전 요구사항 반영)

모든 실행은 `--profile local|server`로 구동되며, 프로파일은 configs에서 선언한다. **코드는 프로파일 간 동일하고 설정만 다르다.**

| 항목 | `local` (macOS 개발 머신) | `server` (Linux GPU 서버) |
|---|---|---|
| 용도 | 파이프라인 개발·검증, task 래퍼 테스트, smoke run | 공식 suite 실행 |
| 모델 | `mockllm`, 상용 API 소량, 또는 서버 vLLM endpoint 원격 접속 | vLLM 서빙 자사 모델 + 상용 API |
| 샌드박스 | Docker Desktop (linux/amd64 에뮬레이션 — 느림, 기능 검증용) | Docker (네이티브) |
| 데이터 | `SCIEVAL_HOME/cache`에 HF 데이터셋·이미지 캐시 | `scieval fetch`로 사전 다운로드 |
| suite | `smoke`만 기본 허용 (그 외는 `--force`) | 전체 |

이전 가능성을 보장하는 규칙:

1. **경로·endpoint 하드코딩 금지** — 모든 경로는 `SCIEVAL_HOME`(env var) 기준 상대 경로, 모든 endpoint·API key는 env var 참조. manifest에도 절대 경로를 기록하지 않는다.
2. **파일 기반 저장소** — 결과는 run 디렉토리 단위로 자기완결(parquet + manifest + .eval 로그). 서버 DB 없음. `rsync`로 로컬↔서버 결과 이동이 그대로 성립한다.
3. **재현 환경** — `uv` workspace + lockfile로 파이썬 환경 고정. 서버 이전 = `git clone && uv sync && scieval fetch`.
4. **플랫폼 요구 태깅** — 각 suite/task에 `requires: [linux, high-ram, gpu, network]`를 선언하고, runner가 프로파일과 대조해 미충족 시 실행 거부(로컬에서 Terminal-Bench 전체 실행 같은 실수 방지).

## 5. v1 벤치마크 포트폴리오

md의 6축 가중치(22/20/15/23/10/10)를 유지하고, 2026-07 실사 결과로 실행 방식과 축 내 가중치를 확정한다. 아래 "기여 가중치"는 최종 지수 100 중 절대 기여분이며 `configs/weights.yaml`에서 조정 가능한 시작값이다.

### 축 1 — 고난도 과학 추론 (22%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| FrontierScience-Olympiad (gold 100) | 8 | inspect_evals `frontierscience` | 답 동등성 model-grader, judge 핀 |
| FrontierScience-Research (gold 60) | 6 | inspect_evals `frontierscience` | 10점 rubric judge — judge 핀 필수 |
| HLE text-only STEM subset | 5 | inspect_evals `hle` (`include_multi_modal=False` + 수학·자연과학 category 필터) | bio/chem 정답 유효성 논란(~18–30% 지적) → 공개 유효성 분석 기반 제외 문항 목록을 configs에 버전 관리 |
| CritPt | 3 | 사이드 트랙: 자체 harness 생성 + AA 서버 채점(일 10회, 70문항 일괄) | 점수 비동기 도착 — 미도착 시 축 내 재정규화 |
| GPQA Diamond | 0 | inspect_evals `gpqa_diamond` | 회귀·타 모델 비교 전용, 지수 제외(포화·오염) |

FrontierMath v2는 **파이프라인에서 제외** — 비공개(338문항 중 12개만 공개), Epoch 대행 평가만 가능. 필요 시 Epoch 제출을 별도 외부 트랙으로 관리한다.

### 축 2 — 과학 코딩·데이터 분석 (20%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| SciCode | 8 | inspect_evals `scicode` (Docker) | subproblem 통과율 + 전체 문제 완결률 동시 보고 |
| ScienceAgentBench | 7 | **자체 포팅** (M3) | 2026-04 verified 아티팩트만, GPT-4o judge 핀 |
| DiscoveryBench | 5 | 사이드 트랙: astabench 포트 | 전면 LLM-judge — judge 핀 |
| DS-1000 | 0 | inspect_evals `ds1000` | smoke 전용 바닥 지표(포화) |

### 축 3 — 문헌 검색·근거 종합 (15%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| ScholarQA-CS2 | 6 | 사이드 트랙: astabench | judge 2026-04 교체 이력 — scorer 버전 기록 |
| LitQA2-FullText | 5 | 사이드 트랙: astabench | Ai2 호스팅 인덱스 의존(~4 req/s) — 장애 시 축 결측 처리 |
| LitQA2-FullText-Search | 4 | 사이드 트랙: astabench | 85문항 소규모 — CI 넓음을 리포트에 명시 |

PaperFindingBench는 v1 제외(recall 분모가 저자 추정치, Ai2 에이전트 실패 사례 기반 선별 편향).

### 축 4 — 연구 실행·재현성 (23%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| CORE-Bench | 9 | inspect_evals `core_bench` (45과제 전체 레벨 구성) | astabench 37과제 CPU 변형과 비호환 — 이 구성으로 고정·문서화 |
| SUPER-Expert | 6 | 사이드 트랙: astabench | 40GB+ RAM, server 전용 태그 |
| E2E-Bench | 5 | 사이드 트랙: astabench | 고비용 — monthly/release suite에서만 |
| E2E-Bench-Hard | 3 | 사이드 트랙: astabench | scorer 2026-04 강화 이력 — scorer 핀 |

### 축 5 — 도구 사용·기술 에이전트 (10%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| Terminal-Bench 2.1 | 4 | `inspect_harbor` 어댑터 | 서드파티 0.x — parity 실패 시 Harbor CLI 사이드 트랙으로 강등 |
| BFCL V1–V3 | 2 | inspect_evals `bfcl` | V4 미포함 확인됨(issue #1026) |
| BFCL V4 agentic | 2 | 사이드 트랙: 공식 `bfcl-eval` | web-search 카테고리는 라이브 인터넷 비결정성 유의 |
| SWE-bench-Live | 2 | 조건부: inspect_evals `swe_bench` 커스텀 데이터셋 경로 호환 테스트 → 실패 시 공식 RepoLaunch 사이드 트랙 | monthly 전용, Docker 대량 필요(server 전용) |

### 축 6 — 신뢰성·장문맥·보정 (10%)

| 벤치마크 | 기여 | 실행 경로 | 비고 |
|---|---:|---|---|
| AA-Omniscience STEM (공개 600문항) | 5 | 자체 Inspect task | 공식 AA 수치와 **비교 불가 근사치**로 라벨 — 정확도·비환각률 분리 보고 |
| AA-LCR | 5 | 자체 Inspect task | judge 핀(공식은 Qwen3-235B 계열) — 3회 반복 채점 |

### 안전 gate (지수 합산 제외, 별도 보고)

| 벤치마크 | 실행 경로 | 해석 원칙 |
|---|---|---|
| AgentDojo | inspect_evals `agentdojo` (AgentDojo v1.2.1 핀) | utility + 공격 성공률 분리 보고. 상당히 포화된 벤치마크 → **통과는 필요조건이지 충분조건 아님** |
| WMDP (bio/chem/cyber) | inspect_evals `wmdp_*` | 전체 공개 → 오염·sandbagging 위험. 낮은 점수를 unlearning 증명으로 해석 금지. 최신 HF revision 사용 |

v1의 gate는 report-only(수치·경고 표시)이며, 차단 기준치 설정은 자사 안전 정책 수립 후 config로 활성화한다.

### 분야별 선택 모듈 (v1.5 후보, 지수 제외)

실사 결과 사용 가능 확인: LAB-Bench 2(gated, 자체 harness), BixBench(aviary harness), ChemBench(inspect_evals 포트), MatTools(Docker harness). 사용 불가 확인: LifeSciBench(비공개·53% 멀티모달), GeneBench-Pro(10/129만 공개), MolQuest(코드·데이터 미공개). 명칭 충돌 주의: OpenCompass의 "ChemBench"는 별개 벤치마크(ChemBench4K)다.

## 6. 평가 프로토콜

### 6.1 Run manifest (모든 run에 JSON으로 고정 기록)

model checkpoint id·tokenizer / endpoint 설정(서빙 이미지·vLLM 버전·파서 플래그) / system prompt / scaffold 이름·버전 / inspect-ai·inspect_evals 버전(커밋) / task 버전·데이터셋 revision / Docker image digest / timeout·retry 정책 / 예산 클래스 / seed / judge 모델·버전 / grader 버전 / 프로파일(local|server). md §6.2의 고정 목록 전체를 포괄한다.

### 6.2 예산 클래스 (`configs/budgets.yaml`)

| 클래스 | 용도 | 시작값 (config로 조정) |
|---|---|---|
| small | smoke·회귀 | 샘플당 token_limit 200k, message_limit 30, working_limit 10분, cost_limit 적용 |
| standard | 공식 비교 점수 | 샘플당 token_limit 1M, message_limit 100, working_limit 40분 |
| extended | 능력 상한 탐색 | 샘플당 token_limit 4M, message_limit 250, working_limit 2시간, 공식 순위 제외 |

시작값은 reasoning 토큰 포함 총 토큰 기준이며, M1에서 자사 모델의 실측 토큰 소비 분포를 보고 재조정한다(md §6.3의 도구 호출 수 20~30/100/200~250 기준을 message_limit에 반영).

release suite는 small/standard/extended 3점을 모두 실행해 성공률-예산 곡선과 Cost@Success를 산출한다. 자사 모델의 달러 비용은 GPU-시간 환산 단가표로 계산하고 Inspect 커스텀 cost 등록으로 `cost_limit`까지 연결한다.

### 6.3 반복 실행과 통계

- agentic 과제 k=3, release candidate k=5. primary metric은 **k회 독립 실행 성공률 평균**(최고점 금지) + bootstrap 95% CI.
- 정적 QA는 k=1 기본, 분산 큰 과제(CritPt 등)는 벤치마크 명세를 따름(CritPt는 5회).

### 6.4 Judge 정책

- 벤치마크별 judge 모델·버전을 `BenchmarkSpec`에 핀 고정하고 manifest에 기록.
- judge 교체 시: 고정 canary transcript 50개를 신·구 judge로 재채점해 드리프트를 수치화한 후 교체. 교체 전후 점수는 리더보드에서 비교 불가로 표시.
- **자사 모델 출력을 자사 모델이 판정하지 않는다.** 비교 대상 상용 모델이 judge인 벤치마크(FrontierScience-Research 등)는 judge 독립성 한계를 리포트에 명시.

### 6.5 Modality 계약

모든 task에 `required_modalities` 메타데이터를 부여하고 text-only 필터를 기본 적용한다(HLE ~14% 이미지 문항 제외 등). 멀티모달 모델의 full-suite 점수와의 직접 비교를 리포트에서 금지 표기한다.

## 7. 점수 합산 엔진

1. **정규화**: `N_b = clip((S_model − S_naive) / (S_expert − S_naive), 0, 1.2)`. `configs/anchors.yaml`에 벤치마크별 naive(무작위/다수결/비에이전트) 및 expert anchor와 출처를 기록. expert anchor가 없는 벤치마크는 naive 대비 min-max로 대체하고 `anchor_quality: fallback`으로 표시.
2. **축 점수**: 축 내 가중 산술평균. 점수 미도착/결측 벤치마크(CritPt 비동기, Ai2 인덱스 장애 등)는 축 내 재정규화 후 리포트에 결측 플래그.
3. **최종 지수**: 6축 가중 기하평균. 기하평균의 0 소멸 방지를 위해 정규화 점수를 ε=0.01로 하한 처리(raw 점수는 별도 보고).
4. **Gate**: 축별 최저 기준·반복 신뢰성 기준·안전 gate를 `configs/gates.yaml`로 선언, 지수와 별도로 pass/fail 보고.
5. **공개 항목**: 최종 지수 + 6축 점수 + 전체 raw 점수 + CI + 비용·시간 + 실패 분류. v1은 Public Comparability Index만 산출한다.

## 8. 리포팅

정적 HTML 리더보드(서버 불필요, 파일로 공유 가능): 모델×suite 표, 축별 레이더, score-vs-cost Pareto 플롯(AstaBench 방법론 차용), run manifest 링크, 실패 사례 분류표(도구 오류/루프/timeout/채점 실패), judge·scorer 버전 표기. 입력은 store의 parquet만 사용한다.

## 9. 에러 처리

- **preflight**: run 시작 전 endpoint health-check(1 completion + usage 응답 확인), 데이터셋·이미지 캐시 확인, 프로파일 요구사항 대조.
- **실행 중**: Inspect eval_set의 retry·resume 사용. 샘플 실패는 기록하되 silent drop 금지.
- **run 유효성**: 완료율 <95%면 run은 `invalid` — 리더보드 편입 금지(부분 결과는 디버그 뷰에만).
- **비용 폭주**: 샘플 단위 cost_limit + run 단위 wall-clock 상한. 사이드 트랙(자체 예산 강제 없는 러너)은 wrapper에서 timeout·비용 사후 검증.
- **사이드 트랙 임포트**: 점수 JSON은 스키마 검증(벤치마크 id·버전·모델 id·점수 범위) 통과 시에만 store에 수용.

## 10. 테스트 전략

1. **scoring 단위 테스트**: 정규화·기하평균·재정규화·gate 로직에 golden fixture.
2. **task 래퍼 CI**: 모든 래퍼를 `mockllm`으로 end-to-end 구동(로컬 프로파일에서 실행 가능).
3. **Parity 검증 (suite 편입 조건)**: 각 포팅 벤치마크를 상용 모델 1개로 실행해 공개 리더보드 수치와 비교. 기준: 공개 수치의 95% CI 이내 또는 ±3%p 이내(둘 중 넓은 쪽), agentic 벤치마크는 반복 3회 평균 기준. 특히 BFCL·CORE-Bench·inspect_harbor 경로(포트 충실도 리스크 지적됨). 불일치 시 원인 규명 전 suite 편입 금지.
4. **smoke suite**: 전체의 10–15% 층화 표본, checkpoint마다 실행 가능한 비용으로 유지.

## 11. 마일스톤

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| **M1 골격** | repo 골격(uv workspace), providers+단가표, base scaffold, 저노력 4개(GPQA-D, SciCode, FrontierScience, HLE-STEM), scoring 코어, 최소 HTML 리포트, local/server 프로파일 | 로컬 mockllm smoke 통과 + 상용 모델 1개 GPQA parity 확인 |
| **M2 agentic** | reference-agent scaffold, Terminal-Bench(inspect_harbor), BFCL V1–V3, CORE-Bench, 안전 gate(AgentDojo·WMDP), 반복 실행 통계·CI | 서버에서 weekly suite 1회 완주 + agentic 반복 3회 통계 산출 |
| **M3 확장** | 사이드 트랙(astabench 문헌·E2E·SUPER·DiscoveryBench, CritPt 게이트웨이, BFCL V4), ScienceAgentBench 자체 포팅, SWE-bench-Live 호환 판정, 예산 곡선, 전체 리더보드 | monthly suite 완주 + 6축 지수·Pareto 리포트 산출 |

## 12. 리스크 및 미해결 항목

| 리스크 | 대응 |
|---|---|
| `inspect_harbor` 서드파티 0.x 의존 | M2 parity 검증으로 판정, 실패 시 Harbor CLI 사이드 트랙 강등 |
| SWE-bench-Live × inspect_evals 호환 미검증 (Live 이미지가 Epoch ghcr.io 레지스트리에 없음) | M3 초에 스파이크 테스트로 판정 |
| astabench의 inspect-ai 버전 핀 충돌 | 사이드 트랙 분리(별도 venv)로 격리 — 이미 설계에 반영 |
| LLM-judge scorer 드리프트(AstaBench 2026-04 재채점 사례) | judge·scorer 핀 + canary 재채점 절차(§6.4) |
| CritPt eval repo LICENSE 파일 부재 | 사용 전 저자 문의 또는 법무 확인 |
| gated 데이터셋 약관(HLE, GPQA, AstaBench, LAB-Bench 2) | 온보딩 문서에 약관 준수 절차 명시, 재배포 금지 |
| WMDP sandbagging·오염 / AgentDojo 포화 | gate 해석 원칙(§5)으로 문서화 — 정량 기준은 안전 정책 수립 후 |
| 자사 모델 GPU-시간 단가 미확정 | M1에서 서빙 처리량 측정 후 단가표 확정 (그 전까지 토큰 수만 보고) |

## 13. 저장소 구조

```
Benchmark/
  science_agent_llm_evaluation_strategy.md   # 전략 원본 (기존)
  docs/
    superpowers/specs/                       # 본 설계 문서
    research/                                # 실사·검증 보고서
  scieval/                                   # 플랫폼 (M1에서 생성)
    pyproject.toml                           # uv workspace
    configs/
      models/*.yaml  suites/*.yaml  profiles/{local,server}.yaml
      budgets.yaml  weights.yaml  anchors.yaml  gates.yaml
    src/scieval/
      providers/  tasks/  scaffolds/  runner/  store/  scoring/  reporting/
    external/                                # 사이드 트랙 (각자 pinned venv)
      astabench_track/  critpt_track/  bfcl_v4_track/
    tests/
```
