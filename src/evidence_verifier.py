
from pathlib import Path
import re
import unicodedata

import pandas as pd


INPUT = Path("results/failure_detection_results.csv")
OUTPUT = Path("results/evidence_verification_results.csv")


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

    # Normalize common Arabic name variants.
    text = re.sub(r"\bابن\b", "بن", text)

    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("P")
    )

    return " ".join(text.split())


def get_accepted_answers(row):
    answers = [row["gold_answer_draft"]]

    if pd.notna(row["accepted_answers_draft"]):
        answers.extend(
            str(row["accepted_answers_draft"]).split(";")
        )

    return [
        normalize(answer)
        for answer in answers
        if pd.notna(answer) and str(answer).strip()
    ]


def verify_answer(row):
    # Only use source-checked references.
    if row["verification_status"] != "source_checked":
        return "unverified_reference"

    answer = normalize(row["model_answer"])

    if not answer:
        return "no_answer"

    accepted = get_accepted_answers(row)

    for candidate in accepted:
        if candidate and candidate in answer:
            return "reference_supported_match"

    return "reference_nonmatch_needs_review"


def main():
    df = pd.read_csv(INPUT)

    df["evidence_status"] = df.apply(
        verify_answer,
        axis=1
    )

    df["evidence_review_flag"] = (
        df["evidence_status"]
        == "reference_nonmatch_needs_review"
    )

    df["consistent_nonmatch_flag"] = (
        df["evidence_review_flag"]
        & (~df["text_disagreement"].astype(bool))
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Evidence verification completed!")
    print("Responses processed:", len(df))

    print("\nEvidence statuses:")
    print(df["evidence_status"].value_counts().to_string())

    print(
        "\nConsistent reference non-matches:",
        int(df["consistent_nonmatch_flag"].sum())
    )

    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()
