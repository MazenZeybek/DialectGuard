# DialectGuard

**Adaptive Detection and Mitigation of Factual Reliability Failures in Arabic Large Language Models**

DialectGuard is a research prototype investigating factual reliability in Arabic-capable large language models (LLMs), including responses to questions written in Modern Standard Arabic (MSA) and Arabic dialects.

The project explores whether potential factual reliability failures can be detected before retrieval and mitigated selectively using dialect normalization and retrieval-augmented generation (RAG).

## Week 1 Prototype

The Week 1 implementation includes:

- A reproducible Arabic question-generation and inference pipeline.
- Baseline responses from three Arabic-capable open-weight LLMs.
- A dialect-to-MSA normalization prototype.
- A multilingual semantic retrieval system using SentenceTransformers and FAISS.
- Evidence reranking using a multilingual cross-encoder.
- A pre-retrieval risk detector based on cross-model answer disagreement and non-answer signals.
- An adaptive mitigation policy.
- An integrated 25-question development demonstration.

## Models Tested

The initial model candidates are:

1. `mlx-community/Qwen3-4B-4bit`
2. `mlx-community/gemma-3-4b-it-4bit`
3. `ibm-research/granite-4.0-h-3b-ar`

The prototype uses the Granite model for normalization and evidence-grounded regeneration.

Models are run locally using MLX on Apple Silicon.

## Pipeline

The integrated pipeline follows these steps:

1. **Baseline generation:** Collect model responses to Arabic questions under recorded inference settings.
2. **Risk detection:** Compare model responses using semantic similarity and identify non-answer signals.
3. **Adaptive decision:** Leave lower-risk responses unchanged and send higher-risk cases to the mitigation pipeline.
4. **Normalization:** Rewrite the question into MSA while attempting to preserve its meaning.
5. **Retrieval:** Retrieve relevant passages using multilingual sentence embeddings and FAISS.
6. **Reranking:** Rank candidate evidence passages with a multilingual cross-encoder.
7. **Grounded regeneration:** Generate an answer using the retrieved evidence, allowing the model to abstain when the evidence is insufficient.
8. **Logging:** Save baseline responses, risk decisions, retrieved evidence identifiers, and final outputs.

The risk detector is a heuristic prototype, not a calibrated probability of factual correctness.

## Environment

The initial implementation was developed on an Apple Silicon Mac with 16 GB of unified memory.

Recommended setup for reproducing the current local prototype:

- macOS on Apple Silicon
- Python virtual environment
- Dependencies from `requirements.txt`
- Internet access for the initial model downloads

## Installation

Clone the repository:

```bash
git clone https://github.com/MazenZeybek/DialectGuard.git
cd DialectGuard
```

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The initial run may download model weights from Hugging Face. These weights are not included in the repository.

## Running the Integrated Demo

From the repository root, run:

```bash
python -m src.dialectguard_pipeline
```

The script reads:

- `results/all_baseline_results.csv` — previously generated model responses.
- `results/evidence_passages.csv` — the small evidence collection.

It writes:

- `results/dialectguard_demo_results.csv` — integrated demo results.

The current script selects the first 25 distinct MSA question entries from the baseline dataset. Dialect normalization is also implemented and tested separately; this integrated run should not be interpreted as a full evaluation across all dialects.

## Week 1 Demonstration

The integrated development demonstration processed **25 questions**:

| Outcome | Count |
|---|---:|
| Total questions | 25 |
| High-risk decisions | 18 |
| Low-risk decisions | 7 |
| Evidence passages available | 5 |

Example successful grounded corrections included:

- Jamal Khashoggi's birthplace: Medina.
- Capital of Morocco: Rabat.
- Beginning of World War II in Europe: September 1, 1939.

These are illustrative examples, not a measured accuracy or hallucination-reduction score.

## Limitations

This is an early research prototype.

- Risk detection can miss incorrect answers when multiple models produce semantically similar responses.
- The five-passage evidence collection covers only a small subset of questions.
- Some high-risk questions produce abstentions because relevant evidence is unavailable.
- The integrated 25-question demonstration uses MSA inputs, while dialect normalization has been tested separately.
- The demonstration is exploratory and should not be treated as held-out final evaluation.
- Systematic accuracy, calibration, latency, and cross-dialect performance comparisons remain future evaluation work.

## Research Collaboration

The project is developed collaboratively. Week 1 work includes implementation of the model inference, normalization, retrieval, risk detection, and integrated demonstration components, alongside the separate dataset and evaluation preparation activities.

Final evaluation questions should remain separate from development and debugging examples.

## Project Status

**Week 1 research prototype — development stage.**

The repository contains experimental scripts and outputs. Interfaces, model selection, and evaluation methodology may be refined in later stages.
