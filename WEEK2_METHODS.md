
# DialectGuard — Researcher B Week 2

## Overview

Implemented and executed the Week 2 development experiments
for DialectGuard, comparing four approaches to factual
reliability in Arabic language models.

## Development Dataset

- 65 base questions
- 4 Arabic varieties: MSA, Jordanian, Egyptian, Saudi
- 260 question variants
- Development split only
- Held-out evaluation data not used

## Experimental Conditions

1. Baseline — original Qwen3 answer.
2. Normalization — dialect-to-MSA normalization followed
   by Qwen3 generation.
3. Always-RAG — retrieval followed by grounded generation.
4. DialectGuard — risk detection followed by selective
   normalization and grounded generation.

## Models

- Qwen: mlx-community/Qwen3-4B-4bit
- Gemma: mlx-community/gemma-3-4b-it-4bit
- Granite: ibm-research/granite-4.0-h-3b-ar

Embedding model:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Reranker:
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

## Configuration

- Temperature: 0.0
- Baseline generation: 80 tokens maximum
- Grounded generation: 120 tokens maximum
- Normalization: 100 tokens maximum
- Risk threshold: 1
- Prompt version: week2-prompts-v2
- Pipeline version: week2-development-v2
- Corpus version: week2-37-passages-screened-v1

## Evidence Corpus

The development RAG corpus contains 37 Arabic Wikipedia
passages from 31 unique source URLs.

Passages were screened for relevance to development
questions but were not independently fact-verified.

The corpus should not be treated as an authoritative
fact-checked evidence collection.

## Results

- 1,040 experimental rows
- 260 rows per condition
- 260 rows per Arabic variety
- 25 columns matching the experiment template
- Zero execution errors
- Zero missing final answers
- Zero duplicate question-condition pairs

Dial​​​ectGuard risk classification:
- High risk: 224 / 260
- Low risk: 36 / 260

## Reproducibility and Limitations

- Cached historical baseline answers were reused.
- Historical baseline generation latency was not available.
- Some DialectGuard latency measurements are therefore
  incomplete and should not be used for direct end-to-end
  latency comparisons.
- Generation counts represent logical model calls.
- Temperature zero does not guarantee deterministic output.
- No explicit random seed was recorded.
- Exact model revision hashes were not recorded.
- The retrieved evidence was relevance-screened, not
  independently fact-verified.
- Development results do not establish factual accuracy,
  precision, recall, or mitigation effectiveness.
- A high risk flag does not prove the corresponding answer
  is incorrect.
- Final evaluation requires independently verified labels
  and held-out test data.

## Running the Pipeline

Activate the project's Python environment and run:

    python -m src.week2_pipeline --bases 65

The script supports resuming from successfully completed
question-condition pairs and retrying failed conditions.

## Output Files

- results/week2_development_experiments.csv
- results/week2_generated_baselines.csv
- results/week2_evidence_final.csv

## Research Responsibilities

Researcher B:
- Local model integration
- Experimental condition execution
- Normalization and risk-routing integration
- RAG corpus preparation
- Development experiment export

Researcher A:
- Gold-answer verification
- Held-out test evaluation
- Final scientific performance assessment
