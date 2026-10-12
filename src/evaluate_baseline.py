
import os
import re
import pandas as pd

RESULTS_PATH = "results/all_baseline_results.csv"
ANSWER_KEY_PATH = "data/qs_AK.csv"
OUTPUT_PATH = "results/all_baseline_evaluation_draft.csv"

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def get_accepted_answers(row):
    answers = []

    for column in ["gold_answer_draft", "accepted_answers_draft"]:
        value = row.get(column)

        if pd.notna(value):
            answers.extend(str(value).split(";"))

    return [normalize(answer) for answer in answers if normalize(answer)]

def evaluate_answer(answer, accepted_answers):
    normalized = normalize(answer)

    if not normalized or not accepted_answers:
        return "needs_review"

    for gold in accepted_answers:
        if normalized == gold:
            return "correct"

        # Allow a short answer embedded in a longer sentence.
        if len(gold) >= 4 and re.search(
            r"(?<!\w)" + re.escape(gold) + r"(?!\w)",
            normalized
        ):
            return "correct"

    return "needs_review"

def main():
    predictions = pd.read_csv(RESULTS_PATH)
    answer_key = pd.read_csv(ANSWER_KEY_PATH)

    merged = predictions.merge(
        answer_key,
        on="question_id",
        how="left",
        validate="many_to_one"
    )

    merged["accepted_answers"] = merged.apply(
        get_accepted_answers, axis=1
    )

    merged["evaluation_label"] = merged.apply(
        lambda row: evaluate_answer(
            row["model_answer"],
            row["accepted_answers"]
        ) if row["verification_status"] == "source_checked"
        else "unverified_gold",
        axis=1
    )

    os.makedirs("results", exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nEvaluation completed.")
    print("Total responses:", len(merged))
    print("\nEvaluation labels:")
    print(merged["evaluation_label"].value_counts().to_string())

    print("\nBy dialect:")
    print(pd.crosstab(
        merged["variety"],
        merged["evaluation_label"]
    ).to_string())

    print("\nSaved to:", OUTPUT_PATH)
    print("Note: 'needs_review' is not automatically incorrect.")

if __name__ == "__main__":
    main()
