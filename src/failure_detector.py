
from pathlib import Path
import re
import unicodedata

import pandas as pd


INPUT = Path("results/all_baseline_evaluation_draft.csv")
DISAGREEMENTS = Path("results/dialect_disagreements.csv")
OUTPUT = Path("results/failure_detection_results.csv")


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = text.translate(
        str.maketrans(
            "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
            "01234567890123456789"
        )
    )

    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")

    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )

    return " ".join(text.split())


def main():
    df = pd.read_csv(INPUT)
    disagreements = pd.read_csv(DISAGREEMENTS)

    # Match model IDs to the names used by our
    # cross-dialect disagreement dataset.
    model_names = {
        "mlx-community/Qwen3-4B-4bit": "Qwen3",
        "mlx-community/gemma-3-4b-it-4bit": "Gemma3",
        "ibm-research/granite-4.0-h-3b-ar": "Granite",
    }

    df["model"] = df["model_id"].map(model_names)

    if df["model"].isna().any():
        raise ValueError("Unknown model ID detected.")

    disagreement_columns = [
        "question_id",
        "model",
        "text_disagreement",
        "unique_answer_count",
    ]

    df = df.merge(
        disagreements[disagreement_columns],
        on=["question_id", "model"],
        how="left",
        validate="many_to_one",
    )

    if df["text_disagreement"].isna().any():
        raise ValueError("Missing disagreement data.")

    # Reference matching is only meaningful
    # when the gold answer has been source-checked.
    df["reference_checked"] = (
        df["verification_status"] == "source_checked"
    )

    df["reference_match"] = (
        df["reference_checked"]
        & (df["evaluation_label"] == "correct")
    )

    df["reference_nonmatch"] = (
        df["reference_checked"]
        & (df["evaluation_label"] == "needs_review")
    )

    # Basic heuristic: empty answers or answers
    # that simply repeat the input question.
    def possible_non_answer(row):
        answer = normalize(row["model_answer"])
        question = normalize(row["question"])

        if not answer:
            return True

        if answer == question:
            return True

        return False

    df["possible_non_answer"] = df.apply(
        possible_non_answer,
        axis=1
    )

    # Risk signals, not confirmed factual errors.
    df["risk_score"] = (
        df["text_disagreement"].astype(int)
        + 2 * df["reference_nonmatch"].astype(int)
        + 2 * df["possible_non_answer"].astype(int)
    )

    def risk_level(score):
        if score >= 3:
            return "high"
        if score >= 1:
            return "medium"
        return "low"

    df["risk_level"] = df["risk_score"].apply(risk_level)

    # Preserve the distinction between risk
    # and verified correctness.
    df["detection_status"] = "unverified_risk"

    df.loc[
        df["reference_match"],
        "detection_status"
    ] = "reference_match"

    df.loc[
        df["reference_nonmatch"],
        "detection_status"
    ] = "reference_nonmatch_review"

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Failure detection completed!")
    print("Responses processed:", len(df))
    print("Unique questions:", df["question_id"].nunique())

    print("\nRisk levels:")
    print(df["risk_level"].value_counts().to_string())

    print("\nDetection statuses:")
    print(df["detection_status"].value_counts().to_string())

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
