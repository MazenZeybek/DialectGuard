
"""
DialectGuard — Controlled same-generator RAG experiment.

Compare Qwen without retrieval against Qwen with retrieval.
Development questions only. Held-out test data is never used.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.week2_pipeline import (
    ModelManager,
    QWEN,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    GROUNDING_MAX_TOKENS,
    baseline_prompt,
    grounding_prompt,
    retrieve_passages,
)
from src.dialectguard_pipeline import build_index


QUESTIONS_FILE = Path("data/week2_development_questions.csv")
EVIDENCE_FILE = Path("results/week2_evidence_final.csv")
CONFIG_FILE = Path("config/final_experiment.json")
OUTPUT_FILE = Path("results/controlled_qwen_rag.csv")

FIELDS = [
    "question_id",
    "base_question_id",
    "variety",
    "split",
    "condition",
    "generator_model_id",
    "question",
    "answer",
    "retrieved_passage_ids",
    "retrieval_count",
    "generation_count",
    "latency_ms",
    "corpus_version",
    "run_status",
]


def verify_corpus():
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    actual = hashlib.sha256(EVIDENCE_FILE.read_bytes()).hexdigest()

    if actual != config["corpus_sha256"]:
        raise RuntimeError("Evidence corpus hash mismatch!")

    print("Frozen evidence verified.")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.bases <= 65:
        parser.error("--bases must be between 1 and 65")

    config = verify_corpus()

    questions = pd.read_csv(QUESTIONS_FILE).fillna("")
    questions = questions[questions["split"] == "dev"].copy()

    selected = (
        questions["base_question_id"].drop_duplicates().head(args.bases)
    )
    questions = questions[
        questions["base_question_id"].isin(selected)
    ].copy()

    print("Development question variants:", len(questions))
    print("Expected rows:", len(questions) * 2)

    existing = {}
    if OUTPUT_FILE.exists():
        old = pd.read_csv(
            OUTPUT_FILE, dtype=str, keep_default_na=False
        )

        if list(old.columns) != FIELDS:
            raise RuntimeError("Output schema mismatch")

        for _, record in old.iterrows():
            if (
                record["run_status"] == "ok"
                and record["generator_model_id"] == QWEN
                and record["corpus_version"] == config["corpus_version"]
            ):
                key = (record["question_id"], record["condition"])
                existing[key] = record.to_dict()

    required = {
        (str(row["question_id"]), condition)
        for _, row in questions.iterrows()
        for condition in ("qwen_no_rag", "qwen_with_rag")
    }

    if required <= set(existing):
        print("All selected rows already completed.")
        return

    print("Loading embedding model...")
    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=(
            "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
        ),
        local_files_only=True,
    )

    print("Loading reranker...")
    reranker = CrossEncoder(
        RERANK_MODEL,
        revision=(
            "1427fd652930e4ba29e8149678df786c240d8825"
        ),
        local_files_only=True,
    )

    evidence = pd.read_csv(EVIDENCE_FILE)
    evidence = evidence.dropna(subset=["evidence_passage"])
    evidence = evidence[
        evidence["evidence_passage"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)

    index = build_index(evidence, encoder)
    manager = ModelManager()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Rewrite successful checkpoints; failed rows are retried.
    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for key, record in existing.items():
            if key in required:
                writer.writerow(record)

        file.flush()

        for _, row in questions.iterrows():
            question = str(row["question"])
            question_id = str(row["question_id"])

            for condition in ("qwen_no_rag", "qwen_with_rag"):
                if (question_id, condition) in existing:
                    continue

                print(question_id, condition, flush=True)

                retrieved_ids = []
                retrieval_count = 0

                try:
                    start = time.perf_counter()

                    if condition == "qwen_no_rag":
                        prompt = baseline_prompt(question)
                        token_limit = GROUNDING_MAX_TOKENS
                    else:
                        passages = retrieve_passages(
                            question, encoder, index, evidence, reranker
                        )
                        retrieval_count = 1
                        retrieved_ids = [
                            str(item["source_id"]) for item in passages
                        ]
                        prompt = grounding_prompt(question, passages)
                        token_limit = GROUNDING_MAX_TOKENS

                    answer = manager.generate_answer(
                        QWEN, prompt, max_tokens=token_limit
                    )

                    latency = round(
                        (time.perf_counter() - start) * 1000, 2
                    )
                    status = "ok"

                except Exception as exc:
                    print("ERROR:", repr(exc), flush=True)
                    answer = ""
                    latency = ""
                    status = f"error:{type(exc).__name__}"

                writer.writerow({
                    "question_id": question_id,
                    "base_question_id": row["base_question_id"],
                    "variety": row["variety"],
                    "split": row["split"],
                    "condition": condition,
                    "generator_model_id": QWEN,
                    "question": question,
                    "answer": answer,
                    "retrieved_passage_ids": ";".join(retrieved_ids),
                    "retrieval_count": retrieval_count,
                    "generation_count": 1,
                    "latency_ms": latency,
                    "corpus_version": config["corpus_version"],
                    "run_status": status,
                })
                file.flush()

    print("\nCONTROLLED RAG RUN COMPLETE")
    print("Results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
