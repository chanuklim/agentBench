# Fact-check of load-bearing claims

- **Claim 1: inspect_evals contains a first-party `frontierscience` port (plus GPQA-Diamond, HLE, SciCode, DS-1000, CORE-Bench, AgentDojo, WMDP, MLE-bench, LAB-Bench, ChemBench, SWE-bench, BFCL).**
  VERDICT: **confirmed**. The inspect_evals catalog lists `frontierscience` (Knowledge) and every other eval named, matching the report's "13/17 first-party" coverage pillar.
  Source: https://github.com/UKGovernmentBEIS/inspect_evals
  ⚠ **Decision-changing if false** — the coverage count is the primary quantitative argument for Option A.

- **Claim 2: inspect_evals' PaperBench port is WIP, and there is no ScienceAgentBench or SWE-bench-Live port.**
  VERDICT: **confirmed**. The catalog marks PaperBench "Work In Progress" (issue #334); ScienceAgentBench and SWE-bench-Live are absent from the catalog.
  Source: https://github.com/UKGovernmentBEIS/inspect_evals

- **Claim 3: The inspect_evals BFCL port covers v1–v3 only; v4 (agentic) is open issue #1026 with no PR.**
  VERDICT: **confirmed**. The port's README states it "implements V1 (original categories), V2 (live datasets), and V3 (multi-turn)"; issue #1026 (opened Feb 2026) requesting v4 agentic support is still open with no linked PR or branch.
  Sources: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/bfcl/README.md ; https://github.com/UKGovernmentBEIS/inspect_evals/issues/1026

- **Claim 4: Inspect AI enforces real dollar cost limits (`cost_limit`), alongside token/message/time/working limits.**
  VERDICT: **confirmed**. The repo's docs include `_cost_limits.md` ("Cost is computed from token usage and model cost data... Cost limits are checked whenever `generate()` is called") alongside `_token_limits.md`, `_message_limits.md`, `_working_limits.md`, `_turn_limits.md`; CHANGELOG v0.3.180 (20 Feb 2026) adds a `cost_limit()` context manager, and later entries reference validating cost limits and custom cost registrations for routed/custom providers.
  Sources: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/_cost_limits.md ; https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/CHANGELOG.md
  ⚠ **Decision-changing if false** — "only surveyed core with real dollar-budget enforcement" is an explicit pillar of the Option A verdict. It holds.

- **Claim 5: AstaBench is built on Inspect AI (so adopting it is additive to an Inspect core), covers the 11 listed sub-benchmarks, and is Apache-2.0; its docs exercise only hosted API providers.**
  VERDICT: **confirmed** (docs-only-hosted-APIs partially confirmed). Repo states "AstaBench builds on top of the InspectAI framework"; the 11 benchmarks match the report's list (PaperFindingBench, ScholarQABench2, LitQA2 variants, ArxivDIGESTables-Clean, CORE-Bench-Hard, DS-1000, SUPER-Expert, DiscoveryBench, E2E-Bench, E2E-Bench-Hard); code is Apache-2.0. Quickstart docs indeed emphasize OpenAI/Anthropic/Google API keys with local-model paths only implied via Inspect inheritance — consistent with the report's caveat.
  Source: https://github.com/allenai/asta-bench
  ⚠ **Decision-changing if false** — the "AstaBench is an Inspect layer, so it's additive rather than a second architecture" argument is central to Option A. It holds.

- **Claim 6: An `inspect_harbor` bridge exists to run Harbor/Terminal-Bench tasks from Inspect.**
  VERDICT: **confirmed**. `meridianlabs-ai/inspect_harbor` ("Inspect AI interface to Harbor tasks") exists with docs and a PyPI release (inspect-harbor 0.4.5), including a default ReAct agent scaffold for Harbor tasks. Caveat not in the report: it is a third-party (Meridian Labs) 0.x package, not UK-AISI first-party — its fidelity/maintenance is an additional dependency risk for Option A.
  Sources: https://github.com/meridianlabs-ai/inspect_harbor ; https://meridianlabs-ai.github.io/inspect_harbor/agents.html
  ⚠ **Decision-changing if false** — Option A's claim to "capture Harbor's strengths without inheriting its gaps" rests entirely on this bridge. It exists, but its third-party 0.x status slightly weakens the argument versus how the report presents it.

- **Claim 7: Stanford HELM has been in maintenance mode since 2026-06-01 (disqualifying it as a foundation).**
  VERDICT: **confirmed**. HELM's README states: "HELM entered maintenance mode on June 1, 2026," linking a Maintenance Mode Policy.
  Sources: https://github.com/stanford-crfm/helm/blob/main/README.md ; https://crfm-helm.readthedocs.io/en/latest/maintenance_mode/

- **Claim 8: MLGym-Bench is CC-BY-NC 4.0, a hard blocker for commercial use.**
  VERDICT: **confirmed with nuance**. The README license section says "The majority of this code is licensed under CC-BY-NC 4.0"; however, vendored components carry MIT (SWE-Agent, Modded-NanoGPT) and Apache-2.0 (Gymnax) licenses. The NonCommercial blocker on the framework/benchmark itself stands.
  Source: https://github.com/facebookresearch/MLGym

- **Claim 9: CritPt ground-truth answers are withheld; grading is server-side only, rate-limited to 10 submissions per account per 24h, accepting only complete all-70 batches.**
  VERDICT: **confirmed**. The CritPt paper and repo state answers to the 70 test challenges are kept private, the online grading server accepts only complete 70-problem batches, and submissions are limited to 10 per account per 24 hours.
  Sources: https://arxiv.org/abs/2509.26574 ; https://github.com/CritPt-Benchmark/CritPt ; https://artificialanalysis.ai/evaluations/critpt

**Summary:** All nine load-bearing claims verified against primary sources; none were wrong. The four flagged claims (1, 4, 5, 6) all hold, so the report's Option A recommendation survives adversarial checking. The only material nuance found: the `inspect_harbor` bridge is a young third-party Meridian Labs package (0.x), not a first-party UK-AISI or Harbor-org component — worth stating explicitly before the architecture decision, since Option A's Harbor-coverage argument depends on it.