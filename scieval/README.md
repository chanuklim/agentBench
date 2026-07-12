# scieval

과학기술 에이전트 LLM 평가 플랫폼 (Inspect AI 코어).
설계: ../docs/superpowers/specs/2026-07-12-scieval-platform-design.md

## 설치

    uv sync

## 환경 변수

| 변수 | 용도 |
|---|---|
| SCIEVAL_HOME | run 출력 루트 (기본 ~/.scieval) |
| SOLAR_BASE_URL / SOLAR_API_KEY | 자사 모델 vLLM endpoint |
| OPENAI_API_KEY | 상용 비교 모델·judge |
| HF_TOKEN | gated 데이터셋 (HLE) |

## 실행 (local 프로파일, macOS 개발 머신)

    uv run scieval health solar-open-100b
    uv run scieval fetch --suite smoke
    uv run scieval run --suite smoke --model gpt-5 --budget standard --profile local
    uv run scieval score <run_dir>   # → results.parquet, scores.json, report.html

## 서버 이전

서버(Linux, Docker)에서: git clone → uv sync → env var 설정 →
`scieval fetch --suite m1` → `--profile server`로 실행.
run 디렉토리는 자기완결이므로 rsync로 로컬↔서버 이동 가능.
