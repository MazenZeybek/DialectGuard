
# DialectGuard

### Adaptive Detection and Mitigation of Factual Reliability Failures in Arabic Large Language Models

DialectGuard is a research project investigating the factual reliability of Large Language Models (LLMs) across Arabic linguistic varieties and evaluating adaptive methods for detecting and mitigating potentially unreliable responses.

The project focuses on four Arabic varieties:

- Modern Standard Arabic (MSA)
- Jordanian Arabic
- Egyptian Arabic
- Saudi Arabic

The proposed framework combines multi-model disagreement analysis, dialect normalization, retrieval-augmented generation (RAG), and risk-aware intervention to investigate whether unreliable factual responses can be identified and improved without requiring retrieval for every question.

The repository provides the system implementation, experimental configurations, development and held-out inference outputs, controlled comparisons, ablation studies, and reproducibility documentation.

**Research status:** Implementation and experimental execution completed. Factual-answer evaluation includes AI-assisted annotations and unresolved evidence-verification limitations documented in this repository and the associated research manuscript.

---

## 1. Research Motivation

Although large language models have demonstrated strong capabilities in natural language understanding and generation, factual reliability remains a significant challenge.

Arabic introduces additional complexity because of substantial variation between Modern Standard Arabic and regional dialects.

Questions expressing the same factual intent may produce different model responses depending on the dialect used.

DialectGuard investigates three related problems:

1. **Dialect sensitivity:** Whether equivalent Arabic factual questions lead to different answers or factual correctness outcomes.
2. **Unreliable responses:** Whether disagreement between multiple language models can indicate possible factual errors.
3. **Adaptive mitigation:** Whether normalization and retrieved evidence can improve response reliability through selective intervention.

Rather than assuming every response requires external evidence retrieval, DialectGuard attempts to distinguish higher-risk questions from lower-risk questions before deciding whether to intervene.

---

## 2. Research Objectives

The objectives of this research are to:

- Evaluate factual question-answering behavior across four Arabic varieties.
- Investigate whether inter-model disagreement is useful as a factual-risk signal.
- Measure the effect of dialect normalization on factual responses.
- Compare baseline generation with retrieval-augmented generation.
- Evaluate adaptive intervention through the DialectGuard framework.
- Examine the computational cost of multiple reliability-improvement strategies.
- Identify limitations of selective retrieval, risk detection, and multilingual factual evaluation.

### Research Questions

**RQ1:** How does factual question-answering reliability vary across Modern Standard Arabic, Jordanian, Egyptian, and Saudi Arabic?

**RQ2:** Can disagreement between language models help identify potentially unreliable factual responses?

**RQ3:** Does normalizing dialectal Arabic questions into Modern Standard Arabic improve factual answer correctness?

**RQ4:** How does retrieval-augmented generation affect the factual reliability of Arabic LLM responses?

**RQ5:** Can an adaptive risk-aware strategy provide a favorable factual-quality and computational-cost trade-off compared with always-on retrieval?

---

## 3. DialectGuard Framework

DialectGuard consists of two main components:

1. **Risk detection:** Identify potentially unreliable answers using model disagreement and non-answer signals.
2. **Adaptive mitigation:** Apply dialect normalization and evidence-grounded regeneration to higher-risk cases.

### System Architecture

```text
                    Arabic Question
                           |
                           v
                Identify Input Variety
                           |
                           v
                  Baseline Generation
                        (Qwen)
                           |
                           v
                  Risk Assessment
                /       |        \
             Qwen     Gemma     Granite
                \       |        /
                 Model Disagreement
                           |
                           v
                   Risk Decision
                     /          \
                    /            \
                Low Risk       High Risk
                   |              |
                   |              v
                   |      Dialect Normalization
                   |              |
                   |              v
                   |       Evidence Retrieval
                   |              |
                   |              v
                   |        Evidence Reranking
                   |              |
                   |              v
                   |      Grounded Regeneration
                   |           (Granite)
                   |              |
                   v              v
             Baseline Answer   Grounded Answer
                    \             /
                     \           /
                      Final Answer
                           |
                           v
                 Experimental Logging
```

The framework uses a heuristic risk score. This score is not a calibrated probability that a generated answer is factually incorrect.

### Risk Detection

The risk-assessment component considers:

- Similarity and disagreement between model-generated answers.
- Non-answer behavior from the baseline model.
- A configurable risk threshold.

Responses classified as higher risk are routed to the mitigation pipeline.

### Adaptive Mitigation

The mitigation process consists of:

1. Transforming the original Arabic question into MSA.
2. Retrieving potentially relevant evidence passages.
3. Reranking retrieved passages.
4. Generating an evidence-conditioned answer.
5. Recording the final response and intervention metadata.

Retrieved evidence is not assumed to be factually correct solely because it was retrieved.

---

## 4. Language Models and Retrieval Components

The implementation uses the following models.

| Component | Model |
|---|---|
| Primary baseline generator | `mlx-community/Qwen3-4B-4bit` |
| Risk-assessment comparison model | `mlx-community/gemma-3-4b-it-4bit` |
| Risk comparison and grounded generation | `ibm-research/granite-4.0-h-3b-ar` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Passage reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |

Model revision information is documented in:

`config/model_revisions.json`

Experiment configuration information is documented in:

`config/final_experiment.json`

The experiments were developed primarily for an Apple Silicon environment using locally executed models.

---

## 5. Experimental Dataset

The study uses Arabic factual question-answering examples adapted into four linguistic varieties.

### Development Dataset

The development set contains:

- 65 base factual questions.
- Four Arabic variants per question.
- 260 total question variants.

The development set was used for implementation testing, pipeline improvements, and initial experimental comparisons.

### Held-Out Dataset

The held-out evaluation contains:

- 100 base factual questions.
- Four Arabic variants per question.
- 400 total question variants.

The final evaluation compares four experimental conditions for each question variant, producing 1,600 held-out experimental records.

The held-out dataset was maintained separately from development experiments to reduce direct test-set tuning.

The dataset underwent review and filtering, but independent dialect-validation and complete semantic-overlap certification have not been documented.

### Evidence Corpus

The frozen development retrieval corpus contains 37 relevance-screened passages collected from 31 Arabic Wikipedia URLs.

Passage relevance screening should not be interpreted as independent factual verification.

---

## 6. Experimental Conditions

Four primary experimental conditions were evaluated.

| Condition | Description |
|---|---|
| `baseline` | Direct question answering using Qwen without retrieval. |
| `normalization` | Normalize the input question to MSA and generate an answer using Qwen. |
| `always_rag` | Retrieve and rerank evidence, then generate an answer with Granite. |
| `dialectguard` | Apply risk detection and selectively normalize, retrieve, and regenerate when the response is classified as high risk. |

### Controlled Experiment

A separate controlled-Qwen RAG comparison investigates retrieval effects while holding the answer-generator family more consistent.

This is important because the primary always-RAG and baseline configurations use different generators.

### Ablation Studies

Ablation experiments investigate selected components of the pipeline and their contribution to system behavior.

### Efficiency Benchmark

A separate efficiency benchmark examines processing time and computational overhead.

---

## 7. Completed Experiments

The repository contains the following experimental outputs.

| Experiment | Records |
|---|---:|
| Development experiments | 1,040 |
| Controlled Qwen RAG comparison | 520 |
| Ablation studies | 780 |
| Efficiency benchmark | 80 |
| Held-out evaluation | 1,600 |
| **Total** | **4,020** |

### Main Result Files

```text
results/
├── development_experiment_results.csv
├── development_baseline_responses.csv
├── development_evidence_corpus.csv
├── heldout_experiment_results.csv
├── heldout_run_manifest.json
├── heldout_quality_control.json
├── controlled_rag_results.csv
├── ablation_results.csv
├── efficiency_results.csv
├── initial_baseline_results.csv
├── normalization_results.csv
├── rag_pilot_results.csv
├── pipeline_pilot_results.csv
├── initial_evidence_passages.csv
├── expanded_evidence_passages.csv
└── model_tests/
```

The held-out inference run was generated using frozen experimental code from commit:

`12cd956c5a1afc11b3a156c1e64f8d863e97a09a`

The frozen results should be preserved when making subsequent documentation or repository organization changes.

---

## 8. Experimental Findings

### Risk-Detection Behavior

During held-out evaluation, DialectGuard classified:

- 378 of 400 question variants as high risk.
- 22 of 400 question variants as low risk.

This corresponds to a **94.5% intervention rate**.

The high intervention frequency suggests that the detector is highly sensitive but insufficiently selective under the tested configuration.

The intervention rate is not equivalent to the factual error rate.

### Computational Performance

Observed median processing times in the held-out experiment were approximately:

| Condition | Median latency |
|---|---:|
| Baseline | 1.95 seconds |
| Normalization | 4.42 seconds |
| Always-RAG | 7.71 seconds |
| DialectGuard | 16.49 seconds |

DialectGuard was slower than always-on RAG in 399 of 400 paired held-out evaluations.

Consequently, the current configuration **does not demonstrate a computational efficiency advantage** over always-on RAG.

The reported latency measurements depend on the local hardware, execution environment, model loading, and caching behavior.

### Factual Reliability Evaluation

A separate analysis of generated responses was conducted using AI-assisted correctness annotations and additional ambiguity-resolution rules.

The evaluation identified important differences among conditions, but some factual judgments remained ambiguous and independent evidence-support verification was incomplete.

Accordingly, the research does not claim that DialectGuard has conclusively demonstrated superior factual accuracy or a verified reduction in hallucinations.

Detailed statistical analysis, annotation provenance, unresolved cases, and uncertainty bounds are documented in the accompanying research analysis materials.

### Output Quality

The held-out run recorded zero reported execution errors and no duplicate `(question_id, condition)` records.

However, eight Granite-generated final responses were empty:

- Five in `always_rag`.
- Three in `dialectguard`.

These cases must be retained or explicitly accounted for when evaluating answer quality.

---

## 9. Repository Structure

```text
DialectGuard/
│
├── README.md
├── METHODOLOGY.md
├── EXPERIMENTAL_RESULTS.md
├── MODEL_COMPARISON.md
├── PROTOTYPE_EVALUATION.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── final_experiment.json
│   └── model_revisions.json
│
├── data/
│   ├── initial_questions.csv
│   ├── pilot_questions.csv
│   ├── development_questions.csv
│   ├── experiment_template.csv
│   └── dialect_revisions.csv
│
├── src/
│   ├── __init__.py
│   ├── dialectguard_pipeline.py
│   ├── development_pipeline.py
│   ├── heldout_experiment.py
│   ├── controlled_rag_experiment.py
│   ├── ablation_experiment.py
│   ├── efficiency_benchmark.py
│   ├── collect_development_evidence.py
│   ├── failure_detector.py
│   ├── mitigation_policy.py
│   ├── rag_pipeline.py
│   ├── retrieve_evidence.py
│   ├── evidence_verifier.py
│   └── other supporting Python modules
│
└── results/
    ├── development_experiment_results.csv
    ├── development_baseline_responses.csv
    ├── development_evidence_corpus.csv
    ├── heldout_experiment_results.csv
    ├── heldout_run_manifest.json
    ├── heldout_quality_control.json
    ├── controlled_rag_results.csv
    ├── ablation_results.csv
    ├── efficiency_results.csv
    ├── initial_baseline_results.csv
    ├── normalization_results.csv
    ├── rag_pilot_results.csv
    ├── pipeline_pilot_results.csv
    ├── initial_evidence_passages.csv
    ├── expanded_evidence_passages.csv
    └── model_tests/
```

The directory tree summarizes the primary research files. Additional existing utilities and scripts should be retained.

---

## 10. Installation

### Requirements

The implementation was developed using:

- Python 3
- Apple Silicon-compatible machine-learning libraries
- MLX-compatible language models
- Dependencies specified in `requirements.txt`

The development machine was an Apple Silicon Mac with 16 GB of unified memory.

Other environments may require dependency modifications.

### Clone the Repository

```bash
git clone https://github.com/MazenZeybek/DialectGuard.git
cd DialectGuard
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment using the appropriate virtual-environment activation command. The MLX-based inference setup is intended primarily for supported Apple Silicon systems.

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model downloads may require an internet connection and sufficient local storage.

---

## 11. Running the Experiments

All commands should be executed from the repository root with the environment activated.

### Development Experiment

Run a small development subset:

```bash
python -m src.development_pipeline --bases 2
```

### Controlled RAG Comparison

```bash
python -m src.controlled_rag_experiment --bases 2
```

### Ablation Study

```bash
python -m src.ablation_experiment --bases 2
```

### Efficiency Benchmark

```bash
python -m src.efficiency_benchmark --limit 2
```

### Held-Out Experiment Validation

```bash
python -m src.heldout_experiment --check
```

This checks the expected held-out input and configuration requirements.

The original locked held-out input dataset is maintained separately from the public repository.

### Reproducing Held-Out Inference

With the authorized held-out input files and frozen configuration available:

```bash
python -m src.heldout_experiment --run
```

Full inference may be computationally expensive.

**Important:** The commands above assume all planned file renames and internal Python imports have been applied successfully. They have not been independently revalidated against the renamed repository.

Do not overwrite the committed original experimental outputs when conducting reproduction tests.

---

## 12. Evaluation Methodology

The research examines the following metrics.

### Factual Correctness

Generated answers are compared with reference answers and classified using a structured annotation protocol.

Potential categories include:

- Correct
- Incorrect
- Partial
- Abstention
- Ambiguous

Arabic synonyms, transliterations, paraphrases, and answer context must be considered when assessing correctness.

### Evidence Support

Evidence-based evaluation examines whether factual claims are supported, contradicted, or left unresolved by the available sources.

Answer correctness and evidence support are separate evaluation dimensions.

### Risk Detection

The risk detector is evaluated using:

- Precision
- Recall
- Specificity
- False-positive behavior
- Intervention rate

Risk-detection metrics require ground-truth correctness judgments.

### Statistical Evaluation

Paired comparisons should account for the fact that four dialect variants originate from the same base question.

The research analysis uses paired comparisons, missingness reporting, and base-question-clustered uncertainty estimation where appropriate.

Ambiguous responses should not silently be counted as correct or incorrect.

### Computational Efficiency

Efficiency analysis includes:

- Inference latency
- Retrieval frequency
- Generation counts
- Model-processing overhead
- Intervention frequency

---

## 13. Reproducibility

Reproducibility relies on preserving:

1. Source code and frozen experimental settings.
2. Model identifiers and revision information.
3. Retrieval-corpus versions and hashes.
4. Prompt templates and risk thresholds.
5. Development and held-out dataset separation.
6. Complete experimental outputs.
7. Evaluation and annotation provenance.

The historical frozen experimental results should remain available through Git history and the committed experimental artifacts.

Renaming files or reorganizing folders does not retroactively change the experimental configuration used to generate the original results.

The final experiment manifest is located at:

`results/heldout_run_manifest.json`

The held-out quality-control record is located at:

`results/heldout_quality_control.json`

---

## 14. Research Limitations

The study has several important limitations.

### Dataset Size and Representation

The held-out benchmark contains 100 base factual questions. It is not necessarily representative of all Arabic factual question-answering domains.

### Dialect Validation

The dialect adaptations were reviewed during dataset construction, but independent expert validation was not comprehensively documented.

### Evidence Quality

Retrieved passages were relevance-screened rather than independently certified as factually correct.

### Generator Confounding

Some conditions use different answer generators. Consequently, differences between conditions cannot always be attributed solely to retrieval or normalization.

### Risk-Detector Selectivity

The high intervention rate reduces the potential efficiency benefits of selective mitigation.

### Hardware-Specific Timing

Latency results reflect the evaluated local implementation and may differ on other hardware or deployment architectures.

### Annotation Uncertainty

AI-assisted answer labels and unresolved ambiguous responses limit the strength of factual-accuracy conclusions.

### Generalizability

Results from these models, dialects, questions, and retrieval materials should not automatically be generalized to other language models or Arabic-speaking populations.

---

## 15. Responsible Use

DialectGuard is intended for research into Arabic language-model reliability.

It should not be treated as:

- A certified fact-checking service.
- A guaranteed hallucination-prevention system.
- An authoritative source of historical or scientific information.
- A production-ready reliability-monitoring platform.

Model-generated answers and retrieved evidence may contain errors.

Users should independently verify information used in high-stakes decisions.

Private reference answers, unreleased evaluation inputs, credentials, and restricted data should not be committed to the public repository.

---

## 16. Research Documentation

The repository includes the following research documents.

| Document | Purpose |
|---|---|
| `README.md` | Main project overview and usage instructions |
| `METHODOLOGY.md` | Research methods and implementation details |
| `EXPERIMENTAL_RESULTS.md` | Experimental setup, results, and limitations |
| `MODEL_COMPARISON.md` | Language-model comparison |
| `PROTOTYPE_EVALUATION.md` | Initial prototype evaluation and observations |

Experimental results are available in the `results/` directory.

The research manuscript is maintained separately from the inference implementation and should be consulted for full scientific interpretation and evaluation caveats.

---

## 17. Related Research

The project is informed by research on Arabic language-model evaluation, retrieval-augmented generation, and adaptive retrieval.

Relevant work includes:

- **DialectalArabicMMLU** — Arabic dialectal language-model evaluation.
- **AraHalluEval** — Arabic hallucination evaluation.
- **Adaptive-RAG** — Adaptive retrieval based on question complexity.
- **Self-RAG** — Retrieval-augmented generation with self-reflection.
- **FLARE** — Active retrieval to support language-model generation.

These studies provide methodological background. DialectGuard specifically investigates response-risk-triggered mitigation for Arabic factual question answering.

---