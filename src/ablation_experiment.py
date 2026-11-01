"""DialectGuard development-only component ablations.

Reuse prior Week 2 answers only when the resulting response is identical by
construction. Generation counts describe *logical* calls, not new calls made
while producing this CSV. Latency is omitted for reused rows because comparable
end-to-end timing is unavailable.

Conditions:
  full_dialectguard - Existing Week 2 risk-routed output.
  no_normalization  - Risk routing retained; high-risk inputs go directly to RAG.
  no_risk_routing   - Always normalize (unless MSA) and RAG, no risk ensemble.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.week2_pipeline import (
    ModelManager, QWEN, GRANITE, EMBEDDING_MODEL, RERANK_MODEL,
    GROUNDING_MAX_TOKENS, normalize_question_week2,
    retrieve_passages, grounding_prompt,
)
from src.dialectguard_pipeline import build_index

DEVELOPMENT_FILE = Path("data/week2_development_questions.csv")
WEEK2_RESULTS = Path("results/week2_development_experiments.csv")
EVIDENCE_FILE = Path("results/week2_evidence_final.csv")
CONFIG_FILE = Path("config/final_experiment.json")
REVISIONS_FILE = Path("config/model_revisions.json")
OUTPUT_FILE = Path("results/ablation_experiments.csv")

CONDITIONS = ("full_dialectguard", "no_normalization", "no_risk_routing")
# Deliberately a NEW schema and output version: old pilot cannot be reused.
ABLATION_VERSION = "ablation-v2-logical-counts"
FIELDS = [
    "question_id", "base_question_id", "variety", "condition",
    "original_question", "normalized_question", "risk_flag", "final_answer",
    "retrieved_passage_ids", "retrieval_count", "generation_count",
    "latency_ms", "generator_model_id", "result_origin", "run_status",
    "ablation_version", "corpus_sha256",
]


def pinned_path(repo_id, revisions):
    matching = [entry for entry in revisions.values() if entry["id"] == repo_id]
    if len(matching) != 1:
        raise ValueError(f"Missing/duplicate revision: {repo_id}")
    return snapshot_download(
        repo_id=repo_id,
        revision=matching[0]["revision"],
        local_files_only=True,
    )


def as_int(value):
    return int(float(value)) if str(value).strip() else 0


def ids_from_week2(row):
    return str(row["retrieved_passage_ids"]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.bases <= 65:
        parser.error("--bases must be between 1 and 65")

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    revisions = json.loads(REVISIONS_FILE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(EVIDENCE_FILE.read_bytes()).hexdigest()
    if digest != config["corpus_sha256"]:
        raise RuntimeError("Evidence corpus changed from the frozen manifest")

    questions = pd.read_csv(DEVELOPMENT_FILE, dtype=str, keep_default_na=False)
    questions = questions[questions["split"] == "dev"].copy()
    chosen = questions["base_question_id"].drop_duplicates().head(args.bases)
    questions = questions[questions["base_question_id"].isin(chosen)]
    if len(questions) != 4 * args.bases:
        raise RuntimeError("Development questions are incomplete")

    previous = pd.read_csv(WEEK2_RESULTS, dtype=str, keep_default_na=False)
    full = previous[previous["condition"] == "dialectguard"].copy()
    baseline = previous[previous["condition"] == "baseline"].copy()
    full = full.merge(
        baseline[["question_id", "final_answer"]].rename(
            columns={"final_answer": "baseline_answer"}),
        on="question_id", validate="one_to_one", how="inner",
    )
    full = full.merge(
        questions[["question_id", "question", "base_question_id", "variety"]],
        on="question_id", validate="one_to_one", how="inner",
        suffixes=("", "_dev"),
    )
    if len(full) != len(questions):
        raise RuntimeError("Missing or duplicated development Week 2 results")
    if full["run_status"].str.startswith("error").any():
        raise RuntimeError("Source Week 2 rows contain failures")
    if set(full["risk_flag"].unique()) - {"high", "low"}:
        raise RuntimeError("Missing or invalid risk flags")
    # Ensure the source comparison uses the documented frozen settings.
    for field, expected in (
        ("prompt_version", config["prompt_version"]),
        ("corpus_version", config["corpus_version"]),
        ("pipeline_version", config["pipeline_version"]),
    ):
        if not (full[field] == expected).all():
            raise RuntimeError(f"Week 2 {field} differs from frozen settings")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if OUTPUT_FILE.exists():
        old = pd.read_csv(OUTPUT_FILE, dtype=str, keep_default_na=False)
        if list(old.columns) != FIELDS:
            raise RuntimeError(
                "An older ablation CSV exists. Back it up first using: "
                "mv results/ablation_experiments.csv "
                "results/ablation_experiments_old_pilot.csv"
            )
        for entry in old.to_dict("records"):
            if (
                entry["run_status"] == "ok"
                and entry["ablation_version"] == ABLATION_VERSION
                and entry["corpus_sha256"] == digest
            ):
                completed[(entry["question_id"], entry["condition"])] = entry

    needed = {(qid, c) for qid in full["question_id"] for c in CONDITIONS}
    print("Question variants:", len(full))
    print("Expected rows:", len(needed))
    print("Reusable rows:", len(needed & set(completed)))
    if needed <= set(completed):
        print("All selected rows completed")
        return

    # Write a clean checkpoint atomically, then append+flush after each row.
    temp = OUTPUT_FILE.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key, record in completed.items():
            if key in needed:
                writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, OUTPUT_FILE)

    state = {}
    def prepare_rag():
        if state:
            return state
        print("Loading frozen retrieval models and evidence...", flush=True)
        encoder = SentenceTransformer(str(pinned_path(EMBEDDING_MODEL, revisions)))
        reranker = CrossEncoder(str(pinned_path(RERANK_MODEL, revisions)))
        evidence = pd.read_csv(EVIDENCE_FILE)
        evidence = evidence.dropna(subset=["evidence_passage"])
        evidence = evidence[
            evidence["evidence_passage"].astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        state.update(
            encoder=encoder, reranker=reranker, evidence=evidence,
            index=build_index(evidence, encoder), manager=ModelManager(),
        )
        return state

    new_count = 0
    errors = 0
    with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        for _, row in full.iterrows():
            qid = str(row["question_id"])
            question = str(row["question"])
            variety = str(row["variety"])
            high = row["risk_flag"] == "high"
            # 3 ensemble baseline generations are logically required only
            # when risk routing is being evaluated.
            norm_calls = 0 if variety == "msa" else 1

            for condition in CONDITIONS:
                key = (qid, condition)
                if key in completed:
                    continue
                print(qid, condition, flush=True)
                data = dict(
                    question_id=qid,
                    base_question_id=str(row["base_question_id"]),
                    variety=variety,
                    condition=condition,
                    original_question=question,
                    normalized_question="",
                    risk_flag=str(row["risk_flag"]),
                    final_answer="",
                    retrieved_passage_ids="",
                    retrieval_count=0,
                    generation_count=0,
                    latency_ms="",  # intentionally missing for reused work
                    generator_model_id="",
                    result_origin="",
                    run_status="ok",
                    ablation_version=ABLATION_VERSION,
                    corpus_sha256=digest,
                )
                try:
                    if condition == "full_dialectguard":
                        # Existing full risk-routed treatment. Preserve the
                        # original logical model-call count and generator.
                        data.update(
                            normalized_question=str(row["normalized_question"]),
                            final_answer=str(row["final_answer"]),
                            retrieved_passage_ids=ids_from_week2(row),
                            retrieval_count=as_int(row["retrieval_count"]),
                            generation_count=as_int(row["generation_count"]),
                            generator_model_id=str(row["generator_model_id"]),
                            result_origin="reused_week2",
                        )
                    elif condition == "no_normalization" and not high:
                        # Low-risk routing leaves the original Qwen answer.
                        data.update(
                            final_answer=str(row["baseline_answer"]),
                            generation_count=3,
                            generator_model_id=QWEN,
                            result_origin="reused_low_risk",
                        )
                    elif condition == "no_normalization" and high and variety == "msa":
                        # Normalization already identity on MSA: exact reuse.
                        data.update(
                            final_answer=str(row["final_answer"]),
                            retrieved_passage_ids=ids_from_week2(row),
                            retrieval_count=as_int(row["retrieval_count"]),
                            generation_count=3 + (1 if as_int(row["generation_count"]) > 3 else 0),
                            generator_model_id=str(row["generator_model_id"]),
                            result_origin="reused_msa_identity",
                        )
                    elif condition == "no_risk_routing" and high:
                        # High-risk full treatment already did normalize+RAG.
                        # The no-risk method eliminates the 3 ensemble calls.
                        data.update(
                            normalized_question=str(row["normalized_question"]),
                            final_answer=str(row["final_answer"]),
                            retrieved_passage_ids=ids_from_week2(row),
                            retrieval_count=as_int(row["retrieval_count"]),
                            generation_count=norm_calls + (1 if str(row["generator_model_id"]) == GRANITE else 0),
                            generator_model_id=str(row["generator_model_id"]),
                            result_origin="reused_high_risk",
                        )
                    else:
                        s = prepare_rag()
                        start = time.perf_counter()
                        query = question
                        # No-normalization: high risk uses original dialect.
                        # No-risk-routing: all questions use normalization.
                        if condition == "no_risk_routing":
                            query, executed_norm_calls = normalize_question_week2(
                                question, variety, s["manager"]
                            )
                            data["normalized_question"] = query
                            data["generation_count"] += executed_norm_calls
                        passages = retrieve_passages(
                            query, s["encoder"], s["index"],
                            s["evidence"], s["reranker"],
                        )
                        data["retrieval_count"] = 1
                        data["retrieved_passage_ids"] = ";".join(
                            str(p["source_id"]) for p in passages
                        )
                        if passages:
                            data["final_answer"] = s["manager"].generate_answer(
                                GRANITE, grounding_prompt(query, passages),
                                max_tokens=GROUNDING_MAX_TOKENS,
                            )
                            data["generation_count"] += 1
                            data["generator_model_id"] = GRANITE
                        else:
                            data["final_answer"] = "لا أعرف."
                        if condition == "no_normalization":
                            data["generation_count"] += 3
                        data["latency_ms"] = round(
                            (time.perf_counter() - start) * 1000, 2
                        )
                        data["result_origin"] = "generated"
                except Exception as exc:
                    print("ERROR:", repr(exc), flush=True)
                    data["run_status"] = f"error:{type(exc).__name__}"
                    errors += 1
                writer.writerow(data)
                handle.flush()
                new_count += 1

    print("\nABLATION RUN COMPLETE")
    print("New rows written:", new_count)
    print("Errors in new rows:", errors)
    print("Output:", OUTPUT_FILE)
    print("Expected selected rows:", len(needed))
    print("Note: logical call counts include cached/reused work; "
          "latencies are not full end-to-end comparable estimates.")


if __name__ == "__main__":
    main()
