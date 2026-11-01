
"""
DialectGuard — Week 2 development experiments.

Conditions:
    baseline
    normalization
    always_rag
    dialectguard

Input: Development questions only.
Output: Abdallah's 25-column experiment template.

The held-out test dataset is never loaded.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.dialectguard_pipeline import (
    calculate_risk,
    build_index,
    retrieve_evidence,
)


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = Path(
    "data/week2_development_questions.csv"
)

OLD_BASELINES_FILE = Path(
    "results/all_baseline_results.csv"
)

NEW_BASELINES_FILE = Path(
    "results/week2_generated_baselines.csv"
)

EVIDENCE_FILE = Path(
    "results/week2_evidence_final.csv"
)

TEMPLATE_FILE = Path(
    "data/week2_experiment_template.csv"
)

OUTPUT_FILE = Path(
    "results/week2_development_experiments.csv"
)

QWEN = "mlx-community/Qwen3-4B-4bit"
GEMMA = "mlx-community/gemma-3-4b-it-4bit"
GRANITE = "ibm-research/granite-4.0-h-3b-ar"

MODELS = [QWEN, GEMMA, GRANITE]

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANK_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)

CONDITIONS = [
    "baseline",
    "normalization",
    "always_rag",
    "dialectguard",
]

VARIETIES = [
    "msa",
    "jordanian",
    "egyptian",
    "saudi",
]

TEMPERATURE = 0.0
MAX_NEW_TOKENS = 80
GROUNDING_MAX_TOKENS = 120
NORMALIZATION_MAX_TOKENS = 100

RISK_THRESHOLD = 1

PROMPT_VERSION = "week2-prompts-v2"
PIPELINE_VERSION = "week2-development-v2"
CORPUS_VERSION = "week2-37-passages-screened-v1"


# ============================================================
# PROMPT TEMPLATES
# ============================================================

def baseline_prompt(question):
    return (
        "أجب عن السؤال التالي بإجابة قصيرة ومباشرة "
        "باللغة العربية: " + question
    )



def normalization_prompt(question):
    return (
        "أنت متخصص في تحويل اللهجات العربية إلى العربية "
        "الفصحى الحديثة.\n"
        "مهمتك هي إعادة كتابة السؤال باللغة العربية الفصحى "
        "مع الحفاظ الكامل على معناه.\n"
        "لا تجب عن السؤال.\n\n"

        "قواعد صارمة:\n"
        "1. حافظ على نوع المعلومة المطلوبة تماماً.\n"
        "2. إذا كان السؤال يسأل عن مدينة، يجب أن يبقى "
        "السؤال عن مدينة، وليس عن تاريخ أو سنة.\n"
        "3. إذا كان السؤال عن سنة محددة، احتفظ بطلب السنة.\n"
        "4. إذا كان السؤال عن تاريخ محدد، احتفظ بطلب التاريخ.\n"
        "5. لا تستبدل أين بمتى أو متى بأين.\n"
        "6. لا تضف معلومات أو تحذف أسماء أو أرقاماً.\n"
        "7. استبدل جميع الكلمات العامية بالفصحى.\n"
        "8. أخرج السؤال النهائي فقط دون مقدمة أو شرح.\n\n"

        "أمثلة:\n"
        "اللهجة: جمال خاشقجي انولد في أي مدينة؟\n"
        "الفصحى: في أي مدينة وُلد جمال خاشقجي؟\n\n"

        "اللهجة: صلاح الدين الأيوبي اتوفّى سنة كام؟\n"
        "الفصحى: في أي سنة توفي صلاح الدين الأيوبي؟\n\n"

        "اللهجة: الحرب العالمية الأولى بدأت يوم كام؟\n"
        "الفصحى: في أي تاريخ بدأت الحرب العالمية الأولى؟\n\n"

        "اللهجة: وش عاصمة المغرب؟\n"
        "الفصحى: ما عاصمة المغرب؟\n\n"

        f"اللهجة: {question}\n"
        "الفصحى:"
    )



def grounding_prompt(question, passages):
    context = "\n\n".join(
        f"[{i + 1}] {item['passage']}"
        for i, item in enumerate(passages)
    )

    return (
        "أجب عن السؤال باستخدام الأدلة التالية فقط.\n"
        "إذا لم تتضمن الأدلة الإجابة بوضوح، قل: لا أعرف.\n"
        "لا تستخدم معلومات خارج الأدلة.\n"
        "لا تخمّن أو تخلط بين الأحداث.\n"
        "أجب بإيجاز.\n\n"
        f"الأدلة:\n{context}\n\n"
        f"السؤال: {question}\n"
        "الإجابة:"
    )


# ============================================================
# MODEL MANAGEMENT
# ============================================================


class ModelManager:
    def __init__(self):
        self.current_model_id = None
        self.current_model = None
        self.current_tokenizer = None

    def get(self, model_id):
        if self.current_model_id != model_id:
            import gc
            import mlx.core as mx

            if self.current_model is not None:
                print(
                    f"Unloading model: {self.current_model_id}",
                    flush=True,
                )

                self.current_model = None
                self.current_tokenizer = None
                self.current_model_id = None

                gc.collect()
                mx.clear_cache()

            print(
                f"Loading model: {model_id}",
                flush=True,
            )

            from huggingface_hub import snapshot_download

            with open(
                "config/model_revisions.json",
                encoding="utf-8",
            ) as f:
                revisions = json.load(f)

            model_config = next(
                item for item in revisions.values()
                if item["id"] == model_id
            )

            model_path = snapshot_download(
                repo_id=model_id,
                revision=model_config["revision"],
                local_files_only=True,
            )

            model, tokenizer = load(model_path)

            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_id = model_id

        return self.current_model, self.current_tokenizer

    def generate_answer(
        self,
        model_id,
        prompt,
        max_tokens=MAX_NEW_TOKENS,
    ):
        model, tokenizer = self.get(model_id)

        if getattr(tokenizer, "chat_template", None):
            options = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            if model_id == QWEN:
                options["enable_thinking"] = False

            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                **options,
            )

        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(
                temp=TEMPERATURE,
            ),
            verbose=False,
        )

        return str(answer).strip()



# ============================================================
# BASELINE CACHE
# ============================================================

def baseline_key(base_id, variety, question, model_id):
    return (
        str(base_id),
        str(variety),
        str(question).strip(),
        str(model_id),
    )


def load_baseline_cache():
    cache = {}

    files = [
        OLD_BASELINES_FILE,
        NEW_BASELINES_FILE,
    ]

    for filename in files:
        if not filename.exists():
            continue

        df = pd.read_csv(filename).fillna("")

        for _, row in df.iterrows():
            key = baseline_key(
                row["question_id"],
                row["variety"],
                row["question"],
                row["model_id"],
            )

            cache[key] = str(row["model_answer"])

    print(
        f"Loaded {len(cache)} cached baseline answers.",
        flush=True,
    )

    return cache


def save_new_baseline(row, model_id, answer):
    NEW_BASELINES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "question_id",
        "variety",
        "question",
        "model_id",
        "model_answer",
    ]

    exists = NEW_BASELINES_FILE.exists()

    with NEW_BASELINES_FILE.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            "question_id": row["base_question_id"],
            "variety": row["variety"],
            "question": row["question"],
            "model_id": model_id,
            "model_answer": answer,
        })


def obtain_baselines(row, cache, manager):
    answers = {}
    measurements = {}

    for model_id in MODELS:
        key = baseline_key(
            row["base_question_id"],
            row["variety"],
            row["question"],
            model_id,
        )

        if key in cache:
            answers[model_id] = cache[key]

            measurements[model_id] = {
                "cached": True,
                "latency_ms": None,
            }

        else:
            start = time.perf_counter()

            answer = manager.generate_answer(
                model_id,
                baseline_prompt(str(row["question"])),
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            answers[model_id] = answer
            cache[key] = answer

            measurements[model_id] = {
                "cached": False,
                "latency_ms": elapsed,
            }

            save_new_baseline(
                row, model_id, answer
            )

    return answers, measurements


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_question_week2(
    question,
    variety,
    manager,
):
    if variety == "msa":
        return question, 0

    result = manager.generate_answer(
        GRANITE,
        normalization_prompt(question),
        max_tokens=NORMALIZATION_MAX_TOKENS,
    )

    result = result.strip().split("\n")[0].strip()

    return (result or question), 1


# ============================================================
# RAG
# ============================================================

def retrieve_passages(
    question,
    encoder,
    index,
    evidence,
    reranker,
):
    return retrieve_evidence(
        question,
        encoder,
        index,
        evidence,
        reranker,
    )


def grounded_answer(
    question,
    passages,
    manager,
):
    if not passages:
        return "لا أعرف.", 0

    answer = manager.generate_answer(
        GRANITE,
        grounding_prompt(question, passages),
        max_tokens=GROUNDING_MAX_TOKENS,
    )

    return answer, 1


# ============================================================
# EXPERIMENT CONDITIONS
# ============================================================

def run_condition(
    row,
    condition,
    answers,
    measurements,
    risk,
    risk_latency_ms,
    manager,
    encoder,
    index,
    evidence,
    reranker,
):
    question = str(row["question"])
    variety = str(row["variety"])

    raw_answer = answers[QWEN]
    final_answer = raw_answer

    normalized = ""
    retrieved_ids = []

    risk_score = ""
    risk_flag = ""

    intervention = False
    retrieval_count = 0

    # These are logical calls required by the condition,
    # including calls that were satisfied by cached answers.
    generation_count = 0

    generator_model = QWEN
    status = "ok"

    start = time.perf_counter()

    if condition == "baseline":
        generation_count = 1

        target_measurement = measurements[QWEN]

        if target_measurement["cached"]:
            latency_ms = ""
            status = "cached_baseline_unmeasured"
        else:
            latency_ms = round(
                target_measurement["latency_ms"],
                2,
            )

    elif condition == "normalization":
        normalized, norm_calls = normalize_question_week2(
            question,
            variety,
            manager,
        )

        final_answer = manager.generate_answer(
            QWEN,
            baseline_prompt(normalized),
        )

        generation_count = norm_calls + 1
        intervention = norm_calls > 0

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

    elif condition == "always_rag":
        passages = retrieve_passages(
            question,
            encoder,
            index,
            evidence,
            reranker,
        )

        retrieval_count = 1
        intervention = True

        retrieved_ids = [
            item["source_id"]
            for item in passages
        ]

        final_answer, grounded_calls = grounded_answer(
            question,
            passages,
            manager,
        )

        generation_count = grounded_calls

        if grounded_calls:
            generator_model = GRANITE

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

    elif condition == "dialectguard":
        risk_score = int(risk["risk_score"])

        risk_flag = (
            "high"
            if risk_score >= RISK_THRESHOLD
            else "low"
        )

        # Qwen + Gemma + Granite are required
        # for disagreement-based risk detection.
        generation_count = 3

        if risk_score >= RISK_THRESHOLD:
            intervention = True

            normalized, norm_calls = normalize_question_week2(
                question,
                variety,
                manager,
            )

            generation_count += norm_calls

            passages = retrieve_passages(
                normalized,
                encoder,
                index,
                evidence,
                reranker,
            )

            retrieval_count = 1

            retrieved_ids = [
                item["source_id"]
                for item in passages
            ]

            final_answer, grounded_calls = grounded_answer(
                normalized,
                passages,
                manager,
            )

            generation_count += grounded_calls

            if grounded_calls:
                generator_model = GRANITE

        intervention_latency = (
            time.perf_counter() - start
        ) * 1000

        # A true end-to-end time requires measured
        # times for ALL three baseline model calls.
        if all(
            not measurements[m]["cached"]
            for m in MODELS
        ):
            baseline_total = sum(
                measurements[m]["latency_ms"]
                for m in MODELS
            )

            latency_ms = round(
                baseline_total
                + risk_latency_ms
                + intervention_latency,
                2,
            )
        else:
            latency_ms = ""
            status = "cached_partial_latency"

    else:
        raise ValueError(
            f"Unsupported condition: {condition}"
        )

    return {
        "run_id": (
            f"{PIPELINE_VERSION}__"
            f"{row['question_id']}__{condition}"
        ),
        "question_id": row["question_id"],
        "base_question_id": row["base_question_id"],
        "variety": variety,
        "split": row["split"],
        "condition": condition,
        "target_model_id": QWEN,
        "generator_model_id": generator_model,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "risk_score": risk_score,
        "risk_flag": risk_flag,
        "intervention_triggered": intervention,
        "normalized_question": normalized,
        "retrieved_passage_ids": ";".join(
            retrieved_ids
        ),
        "retrieval_count": retrieval_count,
        "generation_count": generation_count,
        "latency_ms": latency_ms,
        "prompt_version": PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "max_new_tokens": (
            GROUNDING_MAX_TOKENS
            if generator_model == GRANITE
            else MAX_NEW_TOKENS
        ),
        "seed": "",
        "corpus_version": CORPUS_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_status": status,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    import os
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bases",
        type=int,
        default=2,
        help="Number of development base questions (1-65).",
    )
    args = parser.parse_args()

    if not 1 <= args.bases <= 65:
        parser.error("--bases must be between 1 and 65")

    print("\n=== DialectGuard Week 2 ===")

    questions = pd.read_csv(QUESTIONS_FILE).fillna("")
    questions = questions[
        questions["split"] == "dev"
    ].copy()

    selected_bases = (
        questions["base_question_id"]
        .drop_duplicates()
        .head(args.bases)
    )

    questions = questions[
        questions["base_question_id"].isin(selected_bases)
    ].copy()

    questions["variety"] = pd.Categorical(
        questions["variety"],
        categories=VARIETIES,
        ordered=True,
    )

    questions = questions.sort_values(
        ["base_question_id", "variety"]
    )

    expected = len(questions) * len(CONDITIONS)

    columns = list(pd.read_csv(TEMPLATE_FILE).columns)

    print("Base questions:", len(selected_bases))
    print("Question variants:", len(questions))
    print("Expected output rows:", expected)
    print("Required CSV columns:", len(columns))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load completed rows only if they match the
    # CURRENT experiment configuration.
    completed_records = {}

    if OUTPUT_FILE.exists():
        old = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            keep_default_na=False,
        )

        if list(old.columns) != columns:
            raise ValueError(
                "Existing CSV schema does not match template. "
                "Back up the CSV before changing configuration."
            )

        valid_statuses = {
            "ok",
            "cached_baseline_unmeasured",
            "cached_partial_latency",
        }

        for _, record in old.iterrows():
            record = record.to_dict()

            if record["run_status"] not in valid_statuses:
                continue

            if record["prompt_version"] != PROMPT_VERSION:
                continue

            if record["corpus_version"] != CORPUS_VERSION:
                continue

            if record["pipeline_version"] != PIPELINE_VERSION:
                continue

            if str(record["temperature"]) != str(TEMPERATURE):
                continue

            key = (
                record["question_id"],
                record["condition"],
            )

            completed_records[key] = record

        print(
            "Reusable completed rows:",
            len(completed_records),
        )

        backup = OUTPUT_FILE.with_suffix(".pre_resume.bak.csv")
        shutil.copy2(OUTPUT_FILE, backup)
        print("Previous output backed up:", backup)

    # Retain successful rows, remove failed rows,
    # and write a clean checkpoint before processing.
    temporary = OUTPUT_FILE.with_suffix(".resume.tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
        )
        writer.writeheader()

        for record in completed_records.values():
            writer.writerow(record)

        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary, OUTPUT_FILE)

    # Skip expensive model loading when all selected
    # question-condition pairs are already complete.
    required_keys = {
        (row["question_id"], condition)
        for _, row in questions.iterrows()
        for condition in CONDITIONS
    }

    missing_keys = required_keys - set(completed_records)

    print("Rows still needed:", len(missing_keys))

    if not missing_keys:
        print("\nAll selected rows are already complete.")
        return

    cache = load_baseline_cache()
    manager = ModelManager()

    print("Loading embedding model...")
    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        local_files_only=True,
    )

    print("Loading reranker...")
    reranker = CrossEncoder(
        RERANK_MODEL,
        revision="1427fd652930e4ba29e8149678df786c240d8825",
        local_files_only=True,
    )

    evidence = pd.read_csv(EVIDENCE_FILE)
    evidence = evidence.dropna(
        subset=["evidence_passage"]
    ).copy()

    evidence = evidence[
        evidence["evidence_passage"]
        .astype(str)
        .str.strip()
        .ne("")
    ].reset_index(drop=True)

    if evidence.empty:
        raise ValueError("Evidence corpus is empty.")

    print("Evidence passages:", len(evidence))
    print("Building FAISS index...")

    index = build_index(evidence, encoder)

    new_rows = 0
    errors = 0

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
        )

        for _, row in questions.iterrows():
            question_id = row["question_id"]

            missing_conditions = [
                condition
                for condition in CONDITIONS
                if (question_id, condition)
                not in completed_records
            ]

            if not missing_conditions:
                continue

            print(
                f"\nQuestion: {question_id}",
                flush=True,
            )

            try:
                answers, measurements = obtain_baselines(
                    row,
                    cache,
                    manager,
                )

                risk_start = time.perf_counter()

                group = pd.DataFrame([
                    {"model_answer": answers[QWEN]},
                    {"model_answer": answers[GEMMA]},
                    {"model_answer": answers[GRANITE]},
                ])

                risk = calculate_risk(group, encoder)

                risk_latency_ms = (
                    time.perf_counter() - risk_start
                ) * 1000

                print(
                    f"Risk: {risk['risk_score']} "
                    f"({risk['risk_level']})"
                )

            except Exception as exc:
                print(
                    "Baseline/risk error:",
                    repr(exc),
                    flush=True,
                )
                raise

            for condition in missing_conditions:
                try:
                    record = run_condition(
                        row,
                        condition,
                        answers,
                        measurements,
                        risk,
                        risk_latency_ms,
                        manager,
                        encoder,
                        index,
                        evidence,
                        reranker,
                    )

                except Exception as exc:
                    print(
                        f"ERROR [{condition}]: {exc}",
                        flush=True,
                    )

                    record = {
                        name: ""
                        for name in columns
                    }

                    record.update({
                        "run_id": (
                            f"{PIPELINE_VERSION}__"
                            f"{question_id}__{condition}"
                        ),
                        "question_id": question_id,
                        "base_question_id": row[
                            "base_question_id"
                        ],
                        "variety": str(row["variety"]),
                        "split": row["split"],
                        "condition": condition,
                        "target_model_id": QWEN,
                        "run_status": (
                            f"error:{type(exc).__name__}"
                        ),
                    })

                    errors += 1

                writer.writerow(record)
                file.flush()

                new_rows += 1

                print(
                    f"  {condition}: "
                    f"{record['run_status']}",
                    flush=True,
                )

    print("\n=== Run completed ===")
    print("Rows reused:", len(required_keys & set(completed_records)))
    print("New rows written:", new_rows)
    print("Execution errors:", errors)
    print("Expected selected rows:", expected)
    print("Results:", OUTPUT_FILE)
    print("New baseline cache:", NEW_BASELINES_FILE)

    print(
        "\nNote: generation_count records logical model "
        "calls. Historical cached inference times "
        "remain unavailable and are not fabricated."
    )


if __name__ == "__main__":
    main()
