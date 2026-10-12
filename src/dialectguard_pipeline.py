
from pathlib import Path
import re
import time

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate


# =====================================================
# CONFIGURATION
# =====================================================

BASELINE_FILE = Path("results/all_baseline_results.csv")
EVIDENCE_FILE = Path("results/evidence_passages.csv")
OUTPUT_FILE = Path("results/dialectguard_demo_results.csv")

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANK_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)

GRANITE_MODEL = "ibm-research/granite-4.0-h-3b-ar"

MAX_QUESTIONS = 25
TOP_K = 1
RERANK_CANDIDATES = 5

HIGH_RISK_THRESHOLD = 1
SEMANTIC_SIMILARITY_THRESHOLD = 0.80

ABSTENTION_PHRASES = [
    "لا أعرف",
    "لا اعرف",
    "لست متأكدا",
    "لست متأكد",
    "لا أستطيع",
    "لا استطيع",
    "ليس لدي معلومات",
    "غير معروف",
    "i don't know",
    "i do not know",
    "not sure",
]


# =====================================================
# TEXT UTILITIES
# =====================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = text.replace("ى", "ي")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_non_answer(answer):
    normalized = normalize_text(answer)

    if not normalized or normalized in {"nan", "none"}:
        return True

    return any(
        normalize_text(phrase) in normalized
        for phrase in ABSTENTION_PHRASES
    )


# =====================================================
# SEMANTIC RISK DETECTION
# =====================================================

def calculate_risk(group, encoder):
    """
    Prototype pre-retrieval risk detector.

    +1: Semantic disagreement between model answers.
    +2: Selected baseline answer is a non-answer.

    No gold answers or retrieved evidence are used.
    """

    answers = (
        group["model_answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    selected_answer = answers[0]
    non_answer = is_non_answer(selected_answer)

    if len(answers) < 2:
        disagreement = False
        minimum_similarity = None

    else:
        embeddings = encoder.encode(
            answers,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        similarities = embeddings @ embeddings.T

        pairwise_scores = [
            float(similarities[i, j])
            for i in range(len(answers))
            for j in range(i + 1, len(answers))
        ]

        minimum_similarity = min(pairwise_scores)

        disagreement = (
            minimum_similarity
            < SEMANTIC_SIMILARITY_THRESHOLD
        )

    score = int(disagreement) + 2 * int(non_answer)

    high_risk = score >= HIGH_RISK_THRESHOLD

    return {
        "risk_score": score,
        "risk_level": "high" if high_risk else "low",
        "model_disagreement": disagreement,
        "minimum_similarity": minimum_similarity,
        "baseline_non_answer": non_answer,
        "baseline_answer": selected_answer,
    }


# =====================================================
# GRANITE GENERATION
# =====================================================

def run_granite(
    prompt,
    model,
    tokenizer,
    max_tokens=120,
):
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    ).strip()


# =====================================================
# DIALECT NORMALIZATION
# =====================================================

def normalize_question(
    question,
    variety,
    model,
    tokenizer,
):
    if variety == "msa":
        return question

    prompt = (
        "أنت محرر لغوي متخصص في تحويل اللهجات العربية "
        "إلى العربية الفصحى الحديثة.\n"
        "مهمتك إعادة صياغة السؤال فقط، وليس الإجابة عنه.\n\n"

        "قواعد إلزامية:\n"
        "1. حافظ على المعنى الأصلي للسؤال تماماً.\n"
        "2. لا تضف أو تحذف أي معلومة.\n"
        "3. لا تغيّر نوع السؤال: سؤال المدينة يبقى عن المدينة، "
        "وسؤال التاريخ يبقى عن التاريخ.\n"
        "4. لا تضف كلمات مثل متى أو أين إذا كانت ستغيّر المطلوب.\n"
        "5. حافظ على الأسماء والأرقام والأحداث.\n"
        "6. أخرج سؤالاً واحداً بالفصحى فقط دون شرح.\n\n"

        "أمثلة:\n"
        "اللهجة: شو عاصمة المغرب؟\n"
        "الفصحى: ما عاصمة المغرب؟\n\n"

        "اللهجة: أمريكا فيها كم ولاية؟\n"
        "الفصحى: كم عدد الولايات في أمريكا؟\n\n"

        "اللهجة: الحرب العالمية الأولى بدأت يوم كام؟\n"
        "الفصحى: في أي تاريخ بدأت الحرب العالمية الأولى؟\n\n"

        f"اللهجة: {question}\n"
        "الفصحى:"
    )

    result = run_granite(
        prompt,
        model,
        tokenizer,
        max_tokens=100,
    )

    result = result.strip().split("\n")[0].strip()

    return result if result else question


# =====================================================
# BUILD FAISS INDEX
# =====================================================

def build_index(evidence, encoder):
    embeddings = encoder.encode(
        evidence["evidence_passage"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# =====================================================
# RETRIEVAL + RERANKING
# =====================================================

def retrieve_evidence(
    question,
    encoder,
    index,
    evidence,
    reranker,
):
    query_embedding = encoder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    candidate_count = min(
        RERANK_CANDIDATES,
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        candidate_count,
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = evidence.iloc[int(idx)]

        candidates.append({
            "source_id": str(row["question_id"]),
            "source_url": str(row["source_url"]),
            "passage": str(row["evidence_passage"]),
            "faiss_score": float(score),
        })

    if not candidates:
        return []

    pairs = [
        (question, item["passage"])
        for item in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for item, score in zip(
        candidates,
        rerank_scores,
    ):
        item["rerank_score"] = float(score)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidates[:TOP_K]


# =====================================================
# GROUNDED GENERATION
# =====================================================

def generate_grounded_answer(
    question,
    retrieved,
    model,
    tokenizer,
):
    if not retrieved:
        return "لا أعرف."

    context = "\n\n".join(
        f"[{i + 1}] {item['passage']}"
        for i, item in enumerate(retrieved)
    )

    prompt = (
        "أجب عن السؤال باستخدام الأدلة التالية فقط.\n"
        "إذا لم تتضمن الأدلة الإجابة بوضوح، قل: لا أعرف.\n"
        "لا تستخدم معلومات خارج الأدلة.\n"
        "لا تخمّن أو تخلط بين الأحداث.\n"
        "أجب بإيجاز.\n\n"
        f"الأدلة:\n{context}\n\n"
        f"السؤال: {question}\n"
        "الإجابة:"
    )

    return run_granite(
        prompt,
        model,
        tokenizer,
        max_tokens=120,
    )


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():
    print("\n=== DialectGuard Week 1 Integrated Demo ===")

    baselines = pd.read_csv(BASELINE_FILE)

    required = {
        "question_id",
        "variety",
        "question",
        "model_id",
        "model_answer",
    }

    missing = required - set(baselines.columns)

    if missing:
        raise ValueError(
            f"Missing baseline columns: {sorted(missing)}"
        )

    question_keys = (
    baselines[["question_id", "variety"]]
    .drop_duplicates()
    .query("variety == 'msa'")
    .head(MAX_QUESTIONS)
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
        raise ValueError("Evidence collection is empty.")

    print("Selected question-variety pairs:", len(question_keys))
    print("Evidence passages:", len(evidence))

    print("\nLoading embedding model...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Building FAISS index...")
    index = build_index(evidence, encoder)

    print("Loading multilingual reranker...")
    reranker = CrossEncoder(RERANK_MODEL)

    print("Loading Granite...")
    model, tokenizer = load(GRANITE_MODEL)

    records = []

    for _, key in question_keys.iterrows():
        qid = key["question_id"]
        variety = key["variety"]

        group = baselines[
            (baselines["question_id"] == qid)
            & (baselines["variety"] == variety)
        ]

        question = str(group.iloc[0]["question"])

        start = time.perf_counter()

        risk = calculate_risk(
            group,
            encoder,
        )

        normalized = ""
        retrieved = []
        intervention = "none"

        if risk["risk_level"] == "high":
            intervention = "normalize_retrieve_regenerate"

            normalized = normalize_question(
                question,
                variety,
                model,
                tokenizer,
            )

            retrieved = retrieve_evidence(
                normalized,
                encoder,
                index,
                evidence,
                reranker,
            )

            final_answer = generate_grounded_answer(
                normalized,
                retrieved,
                model,
                tokenizer,
            )

        else:
            final_answer = risk["baseline_answer"]

        elapsed = time.perf_counter() - start

        record = {
            "question_id": qid,
            "variety": variety,
            "question": question,
            "baseline_answer": risk["baseline_answer"],
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "model_disagreement": risk["model_disagreement"],
            "minimum_similarity": risk["minimum_similarity"],
            "baseline_non_answer": risk["baseline_non_answer"],
            "intervention": intervention,
            "normalized_question": normalized,
            "retrieved_source_ids": "|".join(
                item["source_id"]
                for item in retrieved
            ),
            "retrieved_source_urls": "|".join(
                item["source_url"]
                for item in retrieved
            ),
            "final_answer": final_answer,
            "intervention_seconds": round(elapsed, 3),
            "risk_policy": (
                "semantic_disagreement=1; "
                "baseline_non_answer=2; "
                "similarity_threshold=0.80; "
                "high_threshold=1"
            ),
            "generation_model": GRANITE_MODEL,
        }

        records.append(record)

        print("\n" + "=" * 60)
        print(f"{qid} [{variety}]")
        print("Question:", question)
        print("Baseline:", risk["baseline_answer"])
        print(
            "Minimum similarity:",
            (
                round(risk["minimum_similarity"], 3)
                if risk["minimum_similarity"] is not None
                else "N/A"
            ),
        )
        print("Risk:", risk["risk_score"], risk["risk_level"])
        print("Action:", intervention)

        if normalized:
            print("Normalized:", normalized)

        if retrieved:
            print(
                "Retrieved:",
                [item["source_id"] for item in retrieved],
            )

        print("Final:", final_answer)
        print("Seconds:", round(elapsed, 2))

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("Integrated demo completed!")
    print("Saved:", OUTPUT_FILE)

    print(
        "High-risk cases:",
        sum(r["risk_level"] == "high" for r in records),
    )

    print(
        "Low-risk cases:",
        sum(r["risk_level"] == "low" for r in records),
    )


if __name__ == "__main__":
    main()
