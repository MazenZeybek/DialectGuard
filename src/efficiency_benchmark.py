
"""
DialectGuard development-only efficiency benchmark.

20 variants: 10 historically high-risk and 10 low-risk.
All four Week 2 conditions are measured on the same questions.

No answer-cache reuse. No held-out questions.
Results are saved independently of existing experiments.
"""

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from src import week2_pipeline as w


OUTPUT = Path("results/efficiency_benchmark.csv")

CONDITIONS = [
    "baseline",
    "normalization",
    "always_rag",
    "dialectguard",
]

FIELDS = [
    "question_id",
    "base_question_id",
    "variety",
    "condition",
    "historical_risk_flag",
    "measured_risk_flag",
    "answer",
    "total_ms",
    "model_switch_ms",
    "excluding_model_switch_ms",
    "generation_count",
    "retrieval_count",
    "run_status",
]

# Balanced across risk and variety.
QUOTAS = {
    "msa": 3,
    "jordanian": 3,
    "egyptian": 2,
    "saudi": 2,
}


def choose_questions():
    history = pd.read_csv(
        "results/week2_development_experiments.csv",
        dtype=str,
        keep_default_na=False,
    )

    history = history[
        (history["condition"] == "dialectguard")
        & (history["split"] == "dev")
    ].copy()

    questions = pd.read_csv(
        w.QUESTIONS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    questions = questions[
        questions["split"] == "dev"
    ].copy()

    history = history[
        ["question_id", "risk_flag"]
    ].drop_duplicates("question_id")

    data = questions.merge(
        history,
        on="question_id",
        validate="one_to_one",
    )

    high = []
    low = []

    for variety, count in QUOTAS.items():
        for risk, target in [
            ("high", high),
            ("low", low),
        ]:
            subset = data[
                (data["variety"] == variety)
                & (data["risk_flag"] == risk)
            ].sort_values("question_id")

            if len(subset) < count:
                raise RuntimeError(
                    f"Not enough {variety} {risk} questions"
                )

            target.extend(
                subset.head(count).to_dict("records")
            )

    # Interleave high/low for a useful small pilot.
    selected = []
    for h, l in zip(high, low):
        selected.extend([h, l])

    assert len(selected) == 20
    assert len({x["question_id"] for x in selected}) == 20

    return selected


def prepare_retrieval():
    print("Loading embedding model...")

    encoder = SentenceTransformer(
        w.EMBEDDING_MODEL,
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        local_files_only=True,
    )

    print("Loading reranker...")

    reranker = CrossEncoder(
        w.RERANK_MODEL,
        revision="1427fd652930e4ba29e8149678df786c240d8825",
        local_files_only=True,
    )

    evidence = pd.read_csv(w.EVIDENCE_FILE)

    evidence = evidence.dropna(
        subset=["evidence_passage"]
    ).copy()

    evidence = evidence[
        evidence["evidence_passage"]
        .astype(str)
        .str.strip()
        .ne("")
    ].reset_index(drop=True)

    print("Building FAISS index...")
    index = w.build_index(evidence, encoder)

    return encoder, reranker, evidence, index


def measure_condition(
    row,
    condition,
    manager,
    encoder,
    reranker,
    evidence,
    index,
):
    question = str(row["question"])
    variety = str(row["variety"])

    # Instrument the existing model manager.
    # This records time spent loading or switching models.
    original_get = manager.get

    tracking = {
        "switch_ms": 0.0,
        "calls": 0,
        "retrievals": 0,
    }

    original_generate = manager.generate_answer

    def measured_get(model_id):
        changing = manager.current_model_id != model_id
        start = time.perf_counter()

        result = original_get(model_id)

        if changing:
            tracking["switch_ms"] += (
                time.perf_counter() - start
            ) * 1000

        return result

    def measured_generate(*args, **kwargs):
        tracking["calls"] += 1
        return original_generate(*args, **kwargs)

    manager.get = measured_get
    manager.generate_answer = measured_generate

    measured_risk = ""
    answer = ""

    try:
        start = time.perf_counter()

        if condition == "baseline":
            answer = manager.generate_answer(
                w.QWEN,
                w.baseline_prompt(question),
            )

        elif condition == "normalization":
            normalized, _ = w.normalize_question_week2(
                question,
                variety,
                manager,
            )

            answer = manager.generate_answer(
                w.QWEN,
                w.baseline_prompt(normalized),
            )

        elif condition == "always_rag":
            tracking["retrievals"] += 1

            passages = w.retrieve_passages(
                question,
                encoder,
                index,
                evidence,
                reranker,
            )

            answer, _ = w.grounded_answer(
                question,
                passages,
                manager,
            )

        elif condition == "dialectguard":
            answers = {}

            # Fresh generation for all three risk models.
            for model_id in w.MODELS:
                answers[model_id] = (
                    manager.generate_answer(
                        model_id,
                        w.baseline_prompt(question),
                    )
                )

            group = pd.DataFrame([
                {"model_answer": answers[w.QWEN]},
                {"model_answer": answers[w.GEMMA]},
                {"model_answer": answers[w.GRANITE]},
            ])

            risk = w.calculate_risk(group, encoder)
            risk_score = int(risk["risk_score"])

            measured_risk = (
                "high"
                if risk_score >= w.RISK_THRESHOLD
                else "low"
            )

            answer = answers[w.QWEN]

            if measured_risk == "high":
                normalized, _ = (
                    w.normalize_question_week2(
                        question,
                        variety,
                        manager,
                    )
                )

                tracking["retrievals"] += 1

                passages = w.retrieve_passages(
                    normalized,
                    encoder,
                    index,
                    evidence,
                    reranker,
                )

                answer, _ = w.grounded_answer(
                    normalized,
                    passages,
                    manager,
                )

        else:
            raise ValueError(condition)

        total_ms = (
            time.perf_counter() - start
        ) * 1000

    finally:
        manager.get = original_get
        manager.generate_answer = original_generate

    return {
        "question_id": row["question_id"],
        "base_question_id": row["base_question_id"],
        "variety": variety,
        "condition": condition,
        "historical_risk_flag": row["risk_flag"],
        "measured_risk_flag": measured_risk,
        "answer": answer,
        "total_ms": round(total_ms, 2),
        "model_switch_ms": round(
            tracking["switch_ms"], 2
        ),
        "excluding_model_switch_ms": round(
            total_ms - tracking["switch_ms"], 2
        ),
        "generation_count": tracking["calls"],
        "retrieval_count": tracking["retrievals"],
        "run_status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")

    selected = choose_questions()[:args.limit]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = set()

    if OUTPUT.exists():
        previous = pd.read_csv(
            OUTPUT,
            dtype=str,
            keep_default_na=False,
        )

        if list(previous.columns) != FIELDS:
            raise RuntimeError(
                "Existing benchmark schema mismatch"
            )

        if previous.duplicated(
            ["question_id", "condition"]
        ).any():
            raise RuntimeError(
                "Duplicate rows in existing benchmark"
            )

        completed = {
            (r["question_id"], r["condition"])
            for _, r in previous.iterrows()
            if r["run_status"] == "ok"
        }

    required = {
        (r["question_id"], c)
        for r in selected
        for c in CONDITIONS
    }

    missing = required - completed

    print("Selected questions:", len(selected))
    print("Expected rows:", len(required))
    print("Already completed:", len(required & completed))
    print("Rows remaining:", len(missing))

    if not missing:
        print("Benchmark already complete.")
        return

    encoder, reranker, evidence, index = (
        prepare_retrieval()
    )

    manager = w.ModelManager()

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        if file.tell() == 0:
            writer.writeheader()

        for row in selected:
            for condition in CONDITIONS:
                key = (
                    row["question_id"],
                    condition,
                )

                if key in completed:
                    continue

                print(
                    f"\n{row['question_id']} | {condition}",
                    flush=True,
                )

                result = measure_condition(
                    row,
                    condition,
                    manager,
                    encoder,
                    reranker,
                    evidence,
                    index,
                )

                writer.writerow(result)
                file.flush()

                print(
                    f"Total: {result['total_ms']/1000:.2f}s"
                    f" | Switching:"
                    f" {result['model_switch_ms']/1000:.2f}s",
                    flush=True,
                )

    print("\nEFFICIENCY BENCHMARK COMPLETE")
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
