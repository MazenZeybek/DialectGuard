# DialectGuard — Final Experiment and Researcher B Handoff

## 1. Overview

This document summarizes Researcher B's implementation, experimental execution, reproducibility information, and final deliverables for DialectGuard.

DialectGuard investigates factual reliability across four Arabic varieties: Modern Standard Arabic (MSA), Jordanian, Egyptian, and Saudi Arabic.

The system compares baseline generation, dialect normalization, retrieval-augmented generation (RAG), and risk-triggered intervention.

All experimental generation has been completed. Factual correctness, hallucination rates, and statistical significance remain the responsibility of Researcher A, using independently verified gold answers.

## 2. Models and Experimental Configuration

**Language models**
- Qwen3-4B-4bit — baseline and target model.
- Gemma-3-4B-IT-4bit — risk-detection ensemble.
- Granite-4.0-H-3B-AR — risk detection, normalization, and grounded generation.

**Retrieval models**
- paraphrase-multilingual-MiniLM-L12-v2 — embeddings.
- mmarco-mMiniLMv2-L12-H384-v1 — reranking.

All five model revisions are recorded in `config/model_revisions.json`.

The recorded configuration uses:
- Temperature: 0.0
- Baseline generation limit: 80 tokens
- Normalization limit: 100 tokens
- Grounded generation limit: 120 tokens
- Risk threshold: 1
- Prompt version: `week2-prompts-v2`
- Corpus version: `week2-37-passages-screened-v1`

The configuration is recorded in `config/final_experiment.json`. No explicit random seed was recorded, and temperature zero alone does not guarantee complete determinism.

## 3. Experimental Conditions

**Baseline:** Qwen answers the original question without normalization or retrieval.

**Normalization:** The question is normalized to MSA where applicable, followed by Qwen generation.

**Always RAG:** Supporting evidence is retrieved and supplied to Granite for grounded answer generation.

**DialectGuard:** Qwen, Gemma, and Granite produce baseline responses, which are used for disagreement-based risk assessment. High-risk questions undergo conditional normalization, evidence retrieval, and Granite grounded generation. Low-risk questions retain Qwen's original response.

The standard comparison includes a generator-model confound: baseline uses Qwen, whereas grounded generation uses Granite. A separate controlled experiment was conducted to investigate RAG using the same generator.

## 4. Completed Experiments

| Experiment | Question variants | Conditions | Result rows |
|---|---:|---:|---:|
| Development experiments | 260 | 4 | 1,040 |
| Controlled Qwen RAG comparison | 260 | 2 | 520 |
| Ablation experiments | 260 | 3 | 780 |
| Efficiency benchmark | 20 | 4 | 80 |
| Held-out final experiments | 400 | 4 | 1,600 |
| **Total** | | | **4,020** |

Development, controlled comparison, and ablation data were used before final held-out inference. The locked held-out dataset was not used for prompt selection, risk-threshold calibration, or question-specific evidence collection.

### Controlled RAG

Qwen was used for both the no-RAG and RAG conditions on the 260 development variants.

Average measured latency:
- Qwen without RAG: 0.49 seconds.
- Qwen with RAG: 2.30 seconds.

These results describe measured processing latency, not factual accuracy.

### Ablation Studies

Three development configurations were compared:
- Full DialectGuard.
- DialectGuard without normalization.
- DialectGuard without risk-based routing.

All 780 result rows completed without reported execution errors or duplicate keys.

Average logical model calls:
- Full DialectGuard: 4.51.
- Without normalization: 3.86.
- Without risk routing: 1.75.

The ablation results reuse existing generation outputs where behavior is unchanged. Their recorded latency coverage is incomplete, so the ablation CSV is not a reliable standalone end-to-end latency comparison.

### Efficiency Benchmark

A separate benchmark measured 20 development questions, balanced between historically high-risk and low-risk categories.

| Condition | Average total latency | Average model-switching time | Average model calls |
|---|---:|---:|---:|
| Baseline | 1.86 s | 1.40 s | 1.00 |
| Normalization | 6.33 s | 3.50 s | 1.70 |
| Always RAG | 8.35 s | 3.56 s | 1.00 |
| DialectGuard | 13.46 s | 7.34 s | 3.85 |

These timings were measured on an Apple M2 MacBook with 16 GB RAM using sequential single-model loading. The fixed execution order affects model-switching overhead. They should not be generalized to production server performance.

## 5. Evidence Corpus and Provenance

The frozen retrieval corpus contains:
- 37 Arabic Wikipedia passages.
- 31 unique source URLs.
- Passages associated with 36 development base questions.

The corpus was prepared using development questions only. It was relevance-screened but not independently fact-verified. All 37 passages retained `needs_review` status in the corpus audit.

Frozen corpus SHA-256:

`a1f913ad427ec75b23b91e1eea3014fec8f5c64959876dc195b771fd8224bf0a`

The same corpus was preserved for the final held-out experiment. Its development-oriented selection creates a coverage limitation for unseen questions.

## 6. Final Held-Out Experiment

Dataset version: `heldout-v0.1-review-locked`.

The dataset contains 100 base questions across four Arabic varieties, resulting in 400 question variants. The four frozen conditions produced 1,600 output rows.

The experiment was executed using frozen Git commit:

`12cd956c5a1afc11b3a156c1e64f8d863e97a09a`

The original input split `heldout_review_locked` was mapped to `test` in the output while preserving question identifiers.

**Integrity verification**
- Total result rows: 1,600.
- Unique question IDs: 400.
- Rows per experimental condition: 400.
- Rows per Arabic variety: 400.
- Duplicate question-condition keys: 0.
- Reported execution errors: 0.
- Empty final answers: 8.
- Results SHA-256 matches the recorded run manifest.

Eight Granite-generated responses were empty despite having `run_status=ok`. Five occurred under Always RAG and three under DialectGuard. Each had one recorded evidence retrieval and a grounded generation call.

These eight responses have been preserved without selective regeneration and are documented in `results/heldout_quality_control.json`. Their treatment during scoring should be specified by Researcher A.

### Held-Out Risk and Intervention Statistics

| Variety | High risk | Low risk |
|---|---:|---:|
| MSA | 93 | 7 |
| Jordanian | 94 | 6 |
| Egyptian | 97 | 3 |
| Saudi | 94 | 6 |
| **Total** | **378** | **22** |

DialectGuard triggered intervention on 378/400 variants (94.5%). This describes risk routing behavior, not demonstrated hallucination prevalence or factual error detection accuracy.

Average held-out logical generation calls:
- Baseline: 1.00.
- Normalization: 1.75.
- Always RAG: 1.00.
- DialectGuard: 4.66.

## 7. Reproducibility

Final experiment configuration and exact model revisions are committed to the `final-experiments` Git branch.

The held-out runner reads the locked question-only handoff ZIP locally, verifies hashes, and produces the canonical 25-column output.

Model initialization and retrieval-index construction are excluded from per-question latency measurements. DialectGuard's recorded latency includes three detector-model generations, risk computation, and conditional intervention. Results are saved incrementally.

The held-out question file and private gold answers are not included in the GitHub branch.

## 8. Deliverables

**Primary held-out outputs**
- `results/heldout_experiment_results.csv`
- `results/heldout_run_manifest.json`
- `results/heldout_quality_control.json`

**Supporting experimental outputs**
- `results/week2_development_experiments.csv`
- `results/controlled_qwen_rag.csv`
- `results/ablation_experiments.csv`
- `results/efficiency_benchmark.csv`

**Implementation and configuration**
- `src/week2_pipeline.py`
- `src/heldout_experiment.py`
- `src/controlled_rag_experiment.py`
- `src/ablation_experiment.py`
- `src/efficiency_benchmark.py`
- `config/final_experiment.json`
- `config/model_revisions.json`

**Documentation**
- `WEEK2_METHODS.md`
- `FINAL_EXPERIMENT_HANDOFF.md`

Repository: `https://github.com/MazenZeybek/DialectGuard`

Branch: `final-experiments`

## 9. Limitations and Researcher A Follow-Up

- The held-out dataset is a review-locked candidate, without documented independent linguistic signoff or final semantic-overlap certification.
- Evidence passages were relevance-screened rather than independently fact-verified.
- The primary RAG comparison uses a different generator model from baseline.
- Eight empty model responses require explicit scoring treatment.
- The high intervention rate limits the practical selectivity of risk-based routing.
- Model-switching costs depend on the local hardware and execution schedule.
- Gold-answer verification and blind factual-accuracy scoring have not yet been completed.

Researcher A should score the final results against verified gold answers, determine how empty responses are handled, compare the conditions, assess hallucination and mitigation effectiveness, and conduct appropriate statistical analyses.

No held-out error-driven prompt tuning or selective rerunning should be performed after this final evaluation.

## 10. Handoff Status

Researcher B has completed implementation, development experiments, controlled RAG, ablation experiments, efficiency benchmarking, provenance auditing, final held-out inference, and output integrity verification.

The final experimental artifacts are ready for Researcher A's blind evaluation and scientific analysis.