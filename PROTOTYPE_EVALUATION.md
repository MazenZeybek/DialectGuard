# DialectGuard — Week 1 Integrated Demo Summary

## Objective

Demonstrate a working adaptive pipeline that detects potentially unreliable factual answers to Arabic questions and selectively applies dialect normalization, evidence retrieval, and grounded regeneration.

This is an exploratory development demonstration, not a final accuracy evaluation.

## Setup

- **Hardware:** Apple M2 MacBook Pro, 16 GB unified memory
- **Baseline models:** Qwen3-4B, Gemma 3 4B, Granite 4.0 H 3B Arabic
- **Mitigation model:** Granite 4.0 H 3B Arabic
- **Retrieval:** Multilingual SentenceTransformers embeddings with FAISS
- **Evidence reranking:** Multilingual cross-encoder
- **Evidence collection:** 5 passages
- **Integrated demo inputs:** 25 distinct MSA questions

The larger baseline dataset contains 1,152 model responses across 96 questions, four Arabic varieties (MSA, Jordanian, Egyptian, Saudi), and three models.

## Adaptive Pipeline

1. Load existing baseline answers from the three models.
2. Compute a heuristic risk signal using cross-model semantic disagreement and non-answer indicators.
3. For low-risk cases, preserve the baseline answer.
4. For high-risk cases, normalize the question into MSA, retrieve and rerank evidence, and generate an evidence-grounded answer.
5. Record the risk decision, retrieved passage identifiers, final answer, and processing time.

## Results

| Measure | Observed result |
|---|---:|
| Questions processed | 25 |
| High-risk decisions | 18 |
| Low-risk decisions | 7 |
| Evidence passages | 5 |

The pipeline completed successfully and saved its output to `results/dialectguard_demo_results.csv`.

### Examples of Successful Corrections

| Question | Initial response | Final response |
|---|---|---|
| Where was Jamal Khashoggi born? | Incorrect birthplace | Medina |
| What is Morocco's capital? | Tunis | Rabat |
| When did World War II begin in Europe? | April 1941 | September 1, 1939 |

### Observed Limitations

**False reassurance:** The detector assigned low risk to some incorrect answers, including statements that Oman lies in the western Arabian Peninsula and that China lies in Southeast Asia. Agreement among models is not proof of factual correctness.

**Limited evidence coverage:** With only five evidence passages, several high-risk questions retrieved unrelated evidence and produced abstentions rather than corrected answers.

**Dialect scope:** Although the baseline data includes four Arabic varieties and normalization was tested separately, the 25-question integrated demonstration used MSA questions only.

**Evaluation scope:** The risk counts are routing decisions, not correctness scores. No formal accuracy improvement or hallucination reduction percentage is claimed.

## Reproduction

From the repository root, with dependencies installed:

```bash
python -m src.dialectguard_pipeline
```

Required inputs:

- `results/all_baseline_results.csv`
- `results/evidence_passages.csv`

Generated output:

- `results/dialectguard_demo_results.csv`

The initial run may require downloading embedding, reranking, and language model weights.

## Week 1 Conclusion

The integrated prototype demonstrates the intended adaptive workflow: lower-risk responses can bypass retrieval, while higher-risk responses undergo normalization, evidence retrieval, and grounded regeneration.

The demonstration also identifies two important research challenges for later work: detecting confidently incorrect model agreement and handling insufficient evidence coverage.

These observations will inform subsequent evaluation using separately held-out questions.
