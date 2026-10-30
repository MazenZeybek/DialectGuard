
"""
DialectGuard final held-out experiment runner.

Usage:
    python -m src.heldout_experiment --check
    python -m src.heldout_experiment --run

The --check mode performs no model inference.
The --run mode requires an execution-ready Git commit.

Held-out questions are never used for prompt tuning.
No development answer caches or private gold labels are loaded.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from src import week2_pipeline as w


ZIP_PATH = Path(
    "data/heldout_locked/"
    "DialectGuard_Final_Experiment_Handoff_FOR_MAZEN.zip"
)

CONFIG_PATH = Path("config/final_experiment.json")
REVISIONS_PATH = Path("config/model_revisions.json")

OUTPUT = Path("results/heldout_experiment_results.csv")
RUN_MANIFEST = Path("results/heldout_run_manifest.json")

TEST_FILENAME = "01_HELDOUT_TEST_QUESTIONS_400_NO_GOLD.csv"
TEMPLATE_FILENAME = "02_EXPERIMENT_RESULTS_TEMPLATE_25_COLUMNS.csv"

FINAL_PIPELINE_VERSION = "heldout-final-v1"

CONDITIONS = [
    "baseline",
    "normalization",
    "always_rag",
    "dialectguard",
]

CRITICAL_FILES = [
    "src/heldout_experiment.py",
    "src/week2_pipeline.py",
    "src/dialectguard_pipeline.py",
    "config/final_experiment.json",
    "config/model_revisions.json",
    "results/week2_evidence_final.csv",
]


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def git(*arguments):
    return subprocess.check_output(
        ["git", *arguments],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def read_inputs():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        manifest = json.loads(
            archive.read("MANIFEST.json")
        )

        for filename, expected_hash in manifest["sha256"].items():
            actual_hash = sha256_bytes(
                archive.read(filename)
            )

            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Handoff hash mismatch: {filename}"
                )

        data = archive.read(TEST_FILENAME)
        template = archive.read(TEMPLATE_FILENAME)

    questions = list(
        csv.DictReader(
            io.StringIO(data.decode("utf-8-sig"))
        )
    )

    columns = next(
        csv.reader(
            io.StringIO(template.decode("utf-8-sig"))
        )
    )

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    revisions = json.loads(
        REVISIONS_PATH.read_text(encoding="utf-8")
    )

    return questions, columns, manifest, config, revisions, data


def validate_inputs():
    questions, columns, manifest, config, revisions, raw = (
        read_inputs()
    )

    if len(questions) != 400:
        raise RuntimeError("Expected exactly 400 test variants")

    ids = [q["question_id"] for q in questions]

    if len(set(ids)) != 400:
        raise RuntimeError("Duplicate held-out question IDs")

    bases = {q["base_question_id"] for q in questions}

    if len(bases) != 100:
        raise RuntimeError("Expected exactly 100 base IDs")

    expected_varieties = set(w.VARIETIES)

    if {q["variety"] for q in questions} != expected_varieties:
        raise RuntimeError("Incorrect dialect varieties")

    counts = pd.Series(
        [q["variety"] for q in questions]
    ).value_counts()

    if any(counts.get(v, 0) != 100 for v in w.VARIETIES):
        raise RuntimeError("Expected 100 rows per variety")

    for q in questions:
        if q["split"] != "heldout_review_locked":
            raise RuntimeError("Unexpected held-out split")

        if not q["question"].strip():
            raise RuntimeError("Empty question")

    # No answer labels may be supplied as runner inputs.
    forbidden = (
        "gold_answer",
        "accepted_answers",
        "correct_answer",
    )

    if any(
        any(x in field.lower() for x in forbidden)
        for field in questions[0]
    ):
        raise RuntimeError("Potential gold-answer columns detected")

    expected_columns = list(
        pd.read_csv(
            w.TEMPLATE_FILE,
            nrows=0,
        ).columns
    )

    if columns != expected_columns or len(columns) != 25:
        raise RuntimeError(
            "Handoff template differs from Week 2 schema"
        )

    if config["conditions"] != CONDITIONS:
        raise RuntimeError("Frozen conditions mismatch")

    if config["varieties"] != w.VARIETIES:
        raise RuntimeError("Frozen varieties mismatch")

    checks = {
        "temperature": w.TEMPERATURE,
        "max_new_tokens": w.MAX_NEW_TOKENS,
        "normalization_max_tokens": w.NORMALIZATION_MAX_TOKENS,
        "grounding_max_tokens": w.GROUNDING_MAX_TOKENS,
        "risk_threshold": w.RISK_THRESHOLD,
        "prompt_version": w.PROMPT_VERSION,
        "corpus_version": w.CORPUS_VERSION,
    }

    for key, actual in checks.items():
        if config[key] != actual:
            raise RuntimeError(
                f"Configuration mismatch: {key}"
            )

    corpus_hash = sha256_file(w.EVIDENCE_FILE)

    if corpus_hash != config["corpus_sha256"]:
        raise RuntimeError("Evidence corpus hash mismatch")

    if manifest["expected_result_rows"] != 1600:
        raise RuntimeError("Handoff expected-row mismatch")

    print("Handoff SHA-256 hashes: PASS")
    print("Corpus SHA-256: PASS")
    print("Configuration: PASS")
    print("Question variants:", len(questions))
    print("Base questions:", len(bases))
    print("Output columns:", len(columns))
    print("Expected results:", len(questions) * len(CONDITIONS))
    print("Dataset version:", manifest["dataset_version"])

    return questions, columns, manifest, config, revisions, raw


def verify_frozen_commit():
    commit = git("rev-parse", "HEAD")

    for filename in CRITICAL_FILES:
        current = Path(filename)

        if not current.exists():
            raise RuntimeError(
                f"Missing critical file: {filename}"
            )

        tracked = git("ls-files", "--", filename)

        if filename not in tracked.splitlines():
            raise RuntimeError(
                f"Not committed: {filename}"
            )

        status = git(
            "status",
            "--porcelain",
            "--",
            filename,
        )

        if status:
            raise RuntimeError(
                f"Uncommitted critical file: {filename}"
            )

    print("Frozen Git commit:", commit)
    print("Critical files committed: PASS")

    return commit


def prepare_retrieval(revisions):
    print("Loading pinned embedding model...")

    encoder = SentenceTransformer(
        w.EMBEDDING_MODEL,
        revision=revisions["embedding"]["revision"],
        local_files_only=True,
    )

    print("Loading pinned reranker...")

    reranker = CrossEncoder(
        w.RERANK_MODEL,
        revision=revisions["reranker"]["revision"],
        local_files_only=True,
    )

    evidence = pd.read_csv(w.EVIDENCE_FILE)

    evidence = evidence.dropna(
        subset=["evidence_passage"]
    )

    evidence = evidence[
        evidence["evidence_passage"]
        .astype(str)
        .str.strip()
        .ne("")
    ].reset_index(drop=True)

    if evidence.empty:
        raise RuntimeError("Empty frozen evidence corpus")

    index = w.build_index(evidence, encoder)

    return encoder, reranker, evidence, index


def result_record(
    row,
    condition,
    raw_answer,
    final_answer,
    latency_ms,
    risk=None,
    intervention=False,
    normalized="",
    retrieved_ids=None,
    retrieval_count=0,
    generation_count=0,
    generator_model=None,
):
    risk_score = ""
    risk_flag = ""

    if condition == "dialectguard" and risk is not None:
        risk_score = int(risk["risk_score"])
        risk_flag = (
            "high"
            if risk_score >= w.RISK_THRESHOLD
            else "low"
        )

    generator_model = generator_model or w.QWEN

    return {
        "run_id": (
            f"{FINAL_PIPELINE_VERSION}__"
            f"{row['question_id']}__{condition}"
        ),
        "question_id": row["question_id"],
        "base_question_id": row["base_question_id"],
        "variety": row["variety"],
        # Documented transformation of the original split.
        "split": "test",
        "condition": condition,
        "target_model_id": w.QWEN,
        "generator_model_id": generator_model,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "risk_score": risk_score,
        "risk_flag": risk_flag,
        "intervention_triggered": intervention,
        "normalized_question": normalized,
        "retrieved_passage_ids": ";".join(
            str(x) for x in (retrieved_ids or [])
        ),
        "retrieval_count": retrieval_count,
        "generation_count": generation_count,
        "latency_ms": round(latency_ms, 2),
        "prompt_version": w.PROMPT_VERSION,
        "temperature": w.TEMPERATURE,
        "max_new_tokens": (
            w.GROUNDING_MAX_TOKENS
            if generator_model == w.GRANITE
            else w.MAX_NEW_TOKENS
        ),
        "seed": "",
        "corpus_version": w.CORPUS_VERSION,
        "pipeline_version": FINAL_PIPELINE_VERSION,
        "run_status": "ok",
    }


def generate_question(
    row,
    manager,
    encoder,
    reranker,
    evidence,
    index,
):
    question = row["question"]
    variety = row["variety"]

    # Fresh three-model answers for each held-out question.
    answers = {}
    timings = {}

    for model_id in w.MODELS:
        start = time.perf_counter()

        answers[model_id] = manager.generate_answer(
            model_id,
            w.baseline_prompt(question),
        )

        timings[model_id] = (
            time.perf_counter() - start
        ) * 1000

    raw = answers[w.QWEN]

    risk_start = time.perf_counter()

    risk = w.calculate_risk(
        pd.DataFrame([
            {"model_answer": answers[w.QWEN]},
            {"model_answer": answers[w.GEMMA]},
            {"model_answer": answers[w.GRANITE]},
        ]),
        encoder,
    )

    risk_ms = (
        time.perf_counter() - risk_start
    ) * 1000

    rows = {}

    # 1. Baseline: Qwen only.
    rows["baseline"] = result_record(
        row,
        "baseline",
        raw,
        raw,
        timings[w.QWEN],
        generation_count=1,
    )

    # 2. Normalize question, then Qwen generation.
    start = time.perf_counter()

    normalized, norm_calls = w.normalize_question_week2(
        question,
        variety,
        manager,
    )

    normalized_answer = manager.generate_answer(
        w.QWEN,
        w.baseline_prompt(normalized),
    )

    rows["normalization"] = result_record(
        row,
        "normalization",
        raw,
        normalized_answer,
        (time.perf_counter() - start) * 1000,
        intervention=norm_calls > 0,
        normalized=normalized,
        generation_count=norm_calls + 1,
    )

    # 3. Always RAG: original question, Granite generator.
    start = time.perf_counter()

    passages = w.retrieve_passages(
        question,
        encoder,
        index,
        evidence,
        reranker,
    )

    rag_answer, rag_calls = w.grounded_answer(
        question,
        passages,
        manager,
    )

    rows["always_rag"] = result_record(
        row,
        "always_rag",
        raw,
        rag_answer,
        (time.perf_counter() - start) * 1000,
        intervention=True,
        retrieved_ids=[
            p["source_id"] for p in passages
        ],
        retrieval_count=1,
        generation_count=rag_calls,
        generator_model=(
            w.GRANITE if rag_calls else w.QWEN
        ),
    )

    # 4. DialectGuard: three-model risk detector
    # followed by conditional normalization and RAG.
    start = time.perf_counter()

    dg_answer = raw
    dg_normalized = ""
    dg_ids = []
    dg_retrievals = 0
    dg_calls = 3
    dg_generator = w.QWEN

    high_risk = (
        int(risk["risk_score"]) >= w.RISK_THRESHOLD
    )

    if high_risk:
        dg_normalized, calls = (
            w.normalize_question_week2(
                question,
                variety,
                manager,
            )
        )

        dg_calls += calls

        dg_passages = w.retrieve_passages(
            dg_normalized,
            encoder,
            index,
            evidence,
            reranker,
        )

        dg_retrievals = 1
        dg_ids = [p["source_id"] for p in dg_passages]

        dg_answer, calls = w.grounded_answer(
            dg_normalized,
            dg_passages,
            manager,
        )

        dg_calls += calls

        if calls:
            dg_generator = w.GRANITE

    intervention_ms = (
        time.perf_counter() - start
    ) * 1000

    # Include every actual detector-model call,
    # risk computation, and intervention.
    total_dg_ms = (
        sum(timings.values())
        + risk_ms
        + intervention_ms
    )

    rows["dialectguard"] = result_record(
        row,
        "dialectguard",
        raw,
        dg_answer,
        total_dg_ms,
        risk=risk,
        intervention=high_risk,
        normalized=dg_normalized,
        retrieved_ids=dg_ids,
        retrieval_count=dg_retrievals,
        generation_count=dg_calls,
        generator_model=dg_generator,
    )

    return rows


def load_completed(columns, expected_keys):
    completed = {}

    if not OUTPUT.exists():
        return completed

    previous = pd.read_csv(
        OUTPUT,
        dtype=str,
        keep_default_na=False,
    )

    if list(previous.columns) != columns:
        raise RuntimeError("Existing output schema mismatch")

    for record in previous.to_dict("records"):
        key = (
            record["question_id"],
            record["condition"],
        )

        if key not in expected_keys:
            raise RuntimeError(
                "Unexpected question-condition in output"
            )

        if key in completed:
            raise RuntimeError(
                "Duplicate existing result key"
            )

        if record["run_status"] == "ok":
            completed[key] = record

    return completed


def save_manifest(record):
    RUN_MANIFEST.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs without running inference.",
    )

    mode.add_argument(
        "--run",
        action="store_true",
        help="Execute all frozen held-out experiments.",
    )

    args = parser.parse_args()

    (
        questions,
        columns,
        handoff,
        config,
        revisions,
        raw_test,
    ) = validate_inputs()

    if args.check:
        print("\nVALIDATION ONLY — NO INFERENCE EXECUTED")
        return

    commit = verify_frozen_commit()

    expected_keys = {
        (q["question_id"], c)
        for q in questions
        for c in CONDITIONS
    }

    # Prevent resuming results with a different freeze.
    if RUN_MANIFEST.exists():
        previous_manifest = json.loads(
            RUN_MANIFEST.read_text(encoding="utf-8")
        )

        if (
            previous_manifest["git_commit"] != commit
            or previous_manifest["test_sha256"]
            != sha256_bytes(raw_test)
            or previous_manifest["corpus_sha256"]
            != config["corpus_sha256"]
        ):
            raise RuntimeError(
                "Existing results belong to a different freeze"
            )

    completed = load_completed(
        columns,
        expected_keys,
    )

    print("Reusable successful results:", len(completed))
    print("Remaining results:", 1600 - len(completed))

    if len(completed) == 1600:
        print("All held-out results already complete.")
        return

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT.exists():
        shutil.copy2(
            OUTPUT,
            OUTPUT.with_suffix(".pre_resume.bak.csv"),
        )

    # Atomically retain successful rows only.
    temporary = OUTPUT.with_suffix(".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for record in completed.values():
            writer.writerow(record)

        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary, OUTPUT)

    manifest = {
        "experiment": "DialectGuard final held-out",
        "dataset_version": handoff["dataset_version"],
        "git_commit": commit,
        "test_sha256": sha256_bytes(raw_test),
        "handoff_zip_sha256": sha256_file(ZIP_PATH),
        "corpus_sha256": config["corpus_sha256"],
        "configuration": config,
        "model_revisions": revisions,
        "conditions": CONDITIONS,
        "input_split": "heldout_review_locked",
        "output_split": "test",
        "expected_rows": 1600,
        "timing_method": (
            "Per-question timing includes model inference "
            "and model switching. DialectGuard sums three "
            "fresh detector generations, risk calculation, "
            "and conditional intervention. Embedding and "
            "reranker initialization and FAISS index build "
            "are excluded. Shared detector answers are "
            "not regenerated for the baseline condition."
        ),
        "limitations": [
            "Held-out dataset is review-locked v0.1.",
            "Independent linguistic signoff not established.",
            "Final semantic overlap documentation pending.",
            "Evidence was development-relevance screened "
            "but not independently fact-verified.",
            "Standard Qwen-versus-Granite comparison "
            "confounds retrieval with generator identity.",
            "Sequential local model loading affects latency.",
        ],
        "started_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "running",
    }

    save_manifest(manifest)

    encoder, reranker, evidence, index = (
        prepare_retrieval(revisions)
    )

    manager = w.ModelManager()

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
        )

        for row in questions:
            qid = row["question_id"]

            missing = [
                c for c in CONDITIONS
                if (qid, c) not in completed
            ]

            if not missing:
                continue

            print(
                f"\n{qid} | remaining: {len(missing)}",
                flush=True,
            )

            try:
                generated = generate_question(
                    row,
                    manager,
                    encoder,
                    reranker,
                    evidence,
                    index,
                )

                for condition in missing:
                    record = generated[condition]
                    writer.writerow(record)
                    completed[(qid, condition)] = record

                    print(
                        f"  {condition}: ok",
                        flush=True,
                    )

            except Exception as exc:
                print(
                    f"  ERROR: {type(exc).__name__}: {exc}",
                    flush=True,
                )

                for condition in missing:
                    record = {
                        col: "" for col in columns
                    }

                    record.update({
                        "run_id": (
                            f"{FINAL_PIPELINE_VERSION}__"
                            f"{qid}__{condition}"
                        ),
                        "question_id": qid,
                        "base_question_id": row["base_question_id"],
                        "variety": row["variety"],
                        "split": "test",
                        "condition": condition,
                        "target_model_id": w.QWEN,
                        "pipeline_version": FINAL_PIPELINE_VERSION,
                        "run_status": (
                            f"error:{type(exc).__name__}"
                        ),
                    })

                    writer.writerow(record)

            file.flush()

    final = pd.read_csv(OUTPUT)

    successful = final[
        final["run_status"] == "ok"
    ]

    manifest["successful_rows"] = len(successful)
    manifest["error_rows"] = len(final) - len(successful)
    manifest["finished_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    if (
        len(final) == 1600
        and len(successful) == 1600
        and not final.duplicated(
            ["question_id", "condition"]
        ).any()
    ):
        manifest["status"] = "complete"
    else:
        manifest["status"] = "incomplete"

    manifest["output_sha256"] = sha256_file(OUTPUT)

    save_manifest(manifest)

    print("\nHELD-OUT RUN FINISHED")
    print("Rows:", len(final))
    print("Successful:", len(successful))
    print("Status:", manifest["status"])
    print("Output:", OUTPUT)
    print("Manifest:", RUN_MANIFEST)


if __name__ == "__main__":
    main()
