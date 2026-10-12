
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate


# ==========================================
# CONFIGURATION
# ==========================================

EVIDENCE_FILE = Path("results/evidence_passages.csv")
QUESTIONS_FILE = Path("data/new qs.csv")
OUTPUT_FILE = Path("results/rag_test_results.csv")

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANK_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)

GENERATION_MODEL = "ibm-research/granite-4.0-h-3b-ar"

TOP_K = 1
RERANK_CANDIDATES = 5
MAX_QUESTIONS = 5


# ==========================================
# BUILD FAISS INDEX
# ==========================================

def build_index(documents, encoder):
    embeddings = encoder.encode(
        documents,
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


# ==========================================
# RETRIEVAL + RERANKING
# ==========================================

def retrieve(question, encoder, index, evidence, reranker):
    query_vector = encoder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    )

    candidate_count = min(
        RERANK_CANDIDATES,
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        candidate_count,
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = evidence.iloc[int(idx)]

        candidates.append({
            "question_id": row["question_id"],
            "source_url": row["source_url"],
            "passage": row["evidence_passage"],
            "faiss_score": float(score),
        })

    if not candidates:
        return [], []

    pairs = [
        (question, candidate["passage"])
        for candidate in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for candidate, score in zip(
        candidates,
        rerank_scores,
    ):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidates[:TOP_K], candidates


# ==========================================
# GROUNDED ANSWER GENERATION
# ==========================================

def grounded_answer(question, matches, model, tokenizer):
    if not matches:
        return "لا أعرف."

    context = "\n\n".join(
        f"[{i + 1}] {item['passage']}"
        for i, item in enumerate(matches)
    )

    user_prompt = (
        "أجب عن السؤال اعتماداً على الأدلة التالية فقط.\n"
        "إذا لم تتضمن الأدلة الإجابة بشكل واضح، قل: لا أعرف.\n"
        "لا تستخدم معلومات خارج الأدلة.\n"
        "لا تخلط بين الأحداث أو التواريخ المختلفة.\n"
        "أجب بإيجاز ودقة.\n\n"
        f"الأدلة:\n{context}\n\n"
        f"السؤال: {question}\n"
        "الإجابة:"
    )

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = user_prompt

    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=120,
        verbose=False,
    )

    return answer.strip()


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n=== DialectGuard RAG Prototype ===")

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
        raise ValueError("No evidence passages found.")

    questions = pd.read_csv(QUESTIONS_FILE)

    questions = questions[
        questions["variety"] == "msa"
    ].head(MAX_QUESTIONS)

    print("Evidence passages:", len(evidence))
    print("Questions selected:", len(questions))

    # Load embedding model
    print("\nLoading embedding model...")

    encoder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Build FAISS index
    print("\nBuilding FAISS index...")

    index = build_index(
        evidence["evidence_passage"].tolist(),
        encoder,
    )

    print("FAISS index size:", index.ntotal)

    # Load reranker
    print("\nLoading multilingual reranker...")

    reranker = CrossEncoder(
        RERANK_MODEL
    )

    # Load generation model
    print("\nLoading Granite...")

    model, tokenizer = load(
        GENERATION_MODEL
    )

    records = []

    # Process questions
    for _, row in questions.iterrows():
        qid = row["question_id"]
        question = row["question"]

        matches, candidates = retrieve(
            question,
            encoder,
            index,
            evidence,
            reranker,
        )

        answer = grounded_answer(
            question,
            matches,
            model,
            tokenizer,
        )

        retrieved_ids = [
            item["question_id"]
            for item in matches
        ]

        candidate_ids = [
            item["question_id"]
            for item in candidates
        ]

        hit = qid in retrieved_ids

        top_score = (
            matches[0]["rerank_score"]
            if matches
            else None
        )

        top_faiss_score = (
            matches[0]["faiss_score"]
            if matches
            else None
        )

        records.append({
            "question_id": qid,
            "question": question,
            "retrieved_ids": "|".join(retrieved_ids),
            "candidate_ids": "|".join(candidate_ids),
            "retrieval_hit_at_1": hit,
            "grounded_answer": answer,
            "top_rerank_score": top_score,
            "top_faiss_score": top_faiss_score,
            "retrieved_source_url": (
                matches[0]["source_url"]
                if matches
                else ""
            ),
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANK_MODEL,
            "generation_model": GENERATION_MODEL,
        })

        print("\n" + "=" * 55)
        print(f"{qid}: {question}")
        print("FAISS + reranking candidates:", candidate_ids)
        print("Final retrieved:", retrieved_ids)
        print("Hit@1:", hit)
        print("Rerank score:", top_score)
        print("Grounded answer:", answer)

    # Save results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 55)
    print("RAG test completed!")
    print("Saved:", OUTPUT_FILE)

    if records:
        hits = sum(
            int(record["retrieval_hit_at_1"])
            for record in records
        )

        print(
            f"Passage ID matches: {hits}/{len(records)}"
        )

        print(
            "Note: ID matches are a development-set "
            "diagnostic, not a factual accuracy score."
        )


if __name__ == "__main__":
    main()
