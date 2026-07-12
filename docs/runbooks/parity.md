# Parity 검증 runbook (M1 완료 기준)

목적: inspect_evals 포트가 공개 리더보드 수치를 재현하는지 확인 (스펙 §10).

1. `export OPENAI_API_KEY=...` 후:
   `cd scieval && uv run scieval run --suite smoke --model gpt-5 --budget standard --profile local`
2. `uv run scieval score <run_dir>`
3. results.parquet의 gpqa_diamond headline_value(n=198×epochs)와
   공개 수치(Epoch AI Benchmarking Hub 또는 모델 카드)를 비교:
   `python -c "from scieval.scoring.parity import parity_check; print(parity_check(<observed>, <n>, <published>))"`
4. ok=False면: 프롬프트/epochs/샘플링 차이를 원인 규명 전까지 해당 벤치마크를
   공식 suite에 편입하지 않는다. 결과를 docs/runbooks/parity-log.md에 기록.
